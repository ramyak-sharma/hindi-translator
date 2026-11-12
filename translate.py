#!/usr/bin/env python3
"""Translate an EPUB into Hindi with LLMs, falling back across free models when one fails.

  translate.py run BOOK.epub [--bilingual] [--limit N]   translate (resumable), then write the EPUB
  translate.py build BOOK.epub [--bilingual]             write the EPUB from what is translated so far
  translate.py status [BOOK.epub]                        progress and model availability
  translate.py models                                    show the fallback chain

Keys live in keys.env (see keys.env.example). If it has no GEMINI_API_KEY, the key
is read from the Calibre Ebook Translator plugin's config.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import os
import posixpath
import re
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from lxml import etree

ROOT = Path(__file__).resolve().parent
WORK = ROOT / 'work'
OUTPUT = ROOT / 'output'
STATE = ROOT / 'state.json'
PLUGIN_CONFIG = Path.home() / 'Library/Preferences/calibre/plugins/ebook_translator.json'

DEFAULT_CONFIG = {
    'merge_chars': 6000,           # source characters per request
    'max_segments': 40,            # paragraphs per request
    'temperature': 0.3,
    'style': 'a narrative nonfiction history book',
    'register': ('natural, fluent modern Hindi as used by quality Hindi publishers: '
                 'clear and readable, not stiff or heavily Sanskritised, without needless English words'),
    'provider_order': ['gemini', 'mistral', 'github', 'groq', 'cerebras', 'openrouter'],
    'max_models_per_provider': 10,
    'chain': None,                 # explicit ["provider:model", ...] instead of discovery
    'skip_files': [],              # regexes on file names inside the EPUB to leave untranslated
    'build_every': 10,             # rewrite the output EPUB every N minutes
    'workers': 6,                  # models translating different parts at the same time
}

DEVANAGARI = re.compile('[ऀ-ॿ]')
LETTERS = re.compile('[A-Za-z]')


def log(*a):
    print(time.strftime('%m-%d %H:%M:%S'), *a, flush=True)


def load_json(path, default):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def save_json(path, data):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    os.replace(tmp, path)


def slug(path):
    return re.sub(r'[^\w]+', '-', Path(path).stem).strip('-')[:60]


def load_config(book=None):
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(load_json(ROOT / 'config.json', {}))
    if book:
        cfg.update(load_json(WORK / slug(book) / 'settings.json', {}))
    return cfg


KEY_PREFIXES = [('github_pat_', 'GITHUB_TOKEN'), ('ghp_', 'GITHUB_TOKEN'), ('gsk_', 'GROQ_API_KEY'),
                ('sk-or-', 'OPENROUTER_API_KEY'), ('csk-', 'CEREBRAS_API_KEY'), ('AIza', 'GEMINI_API_KEY')]


def load_keys():
    keys = {}
    f = ROOT / 'keys.env'
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                keys[k.strip()] = v.strip().strip('\'"')
            elif line and not line.startswith('#'):     # bare key: recognise it by its prefix
                for prefix, env in KEY_PREFIXES:
                    if line.startswith(prefix):
                        keys[env] = line
    for p in PROVIDERS.values():
        if os.environ.get(p.key_env):
            keys[p.key_env] = os.environ[p.key_env]
    if not keys.get('GEMINI_API_KEY'):
        try:
            v = load_json(PLUGIN_CONFIG, {})['engine_preferences']['Gemini']['api_keys']
            keys['GEMINI_API_KEY'] = (v if isinstance(v, list) else v.split())[0].strip()
        except (KeyError, IndexError, TypeError, AttributeError):
            pass
    return keys


# ---------------------------------------------------------------- API calls

class ApiError(Exception):
    """kind: minute (wait, retry same model), exhausted (quota gone until `wait`), transient,
    dead (model unusable), auth (key rejected), blocked (refused content), unsupported,
    bad (unusable output) or too_big (request/response too large)."""

    def __init__(self, kind, msg, wait=None):
        super().__init__(msg)
        self.kind, self.wait, self.model = kind, wait, None


def parse_seconds(s):
    if s is None:
        return None
    s = str(s).strip()
    try:
        return float(s)
    except ValueError:
        pass
    parts = re.findall(r'([\d.]+)\s*(ms|h|m|s)', s)
    return sum(float(n) * {'h': 3600, 'm': 60, 's': 1, 'ms': 0.001}[u] for n, u in parts) if parts else None


def classify(code, text, headers=None):
    low = text.lower()
    m = re.search(r'"message":\s*"((?:[^"\\]|\\.)*)"', text)
    short = re.sub(r'\s+', ' ', m.group(1) if m else text)[:160]
    wait = parse_seconds(headers.get('retry-after')) if headers else None
    m = re.search(r'"retryDelay":\s*"([^"]+)"', text) or re.search(r'retry in ([\d.hms]+)', text, re.I)
    if m and wait is None:
        wait = parse_seconds(m.group(1))
    if (code == 401 or code == 403 and re.search(r'\bkey\b|token|auth|credential', low)
            or 'api_key_invalid' in low or 'api key not valid' in low):
        return ApiError('auth', f'{code} {short}')
    if code == 429 or 'resource_exhausted' in low or 'rate limit' in low:
        if 'limit: 0' in low:
            return ApiError('exhausted', f'{code} not on this tier', 86400)
        if 'perday' in low or 'per day' in low or 'per-day' in low or 'daily' in low or (wait or 0) > 300:
            return ApiError('exhausted', f'{code} daily quota', wait or 3600)
        return ApiError('minute', f'{code} {short}', wait or 20)
    if code == 402 or 'insufficient' in low and ('credit' in low or 'balance' in low):
        return ApiError('exhausted', f'{code} no credit', 86400)
    if code == 413 or 'too large' in low or 'context length' in low or 'maximum context' in low:
        return ApiError('too_big', f'{code} {short}')
    if code in (404, 410) or 'no longer available' in low or 'does not exist' in low:
        return ApiError('dead', f'{code} {short}')
    if code == 400 and ('response_format' in low or 'json mode' in low or 'json_validate' in low):
        return ApiError('unsupported', f'{code} {short}')
    if code == 503 or 'overloaded' in low or 'high demand' in low:
        return ApiError('busy', f'{code} {short}')
    if code >= 500 or code in (408, 409):
        return ApiError('transient', f'{code} {short}')
    return ApiError('dead' if code in (400, 403, 422) else 'transient', f'{code} {short}')


def http_json(method, url, headers, body=None, timeout=180):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={'Content-Type': 'application/json', 'User-Agent': 'hindi-translator', **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise classify(e.code, e.read().decode('utf-8', 'replace'), e.headers) from None
    except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, ValueError) as e:
        raise ApiError('transient', f'network: {e}') from None


def version_key(model):
    return tuple(float(n) for n in re.findall(r'\d+(?:\.\d+)?', model))


def rank(ids, prefs, exclude):
    ids = [i for i in ids if not any(x in i.lower() for x in exclude)]
    out = []
    for p in prefs:
        rx = re.compile(p)
        out += sorted((i for i in ids if rx.search(i.lower()) and i not in out), key=version_key, reverse=True)
    return out


class Provider:
    name = key_env = ''
    interval = 4.0      # seconds between requests to one model
    max_chars = None    # largest batch (source characters) this provider's free tier can return in full
    per_model = True    # free limits apply per model (else to the whole account)

    def __init__(self, key, temperature):
        self.key, self.temperature = key, temperature
        self.no_json = set()


class Gemini(Provider):
    name, key_env, interval = 'gemini', 'GEMINI_API_KEY', 6.0
    base = 'https://generativelanguage.googleapis.com/v1beta'

    def hdr(self):
        return {'x-goog-api-key': self.key}

    def discover(self):
        ids, token = [], ''
        while True:
            r = http_json('GET', f'{self.base}/models?pageSize=1000' + (f'&pageToken={token}' if token else ''),
                          self.hdr(), timeout=30)
            ids += [m['name'].split('/', 1)[1] for m in r.get('models', [])
                    if 'generateContent' in m.get('supportedGenerationMethods', [])]
            token = r.get('nextPageToken')
            if not token:
                break
        ids = [i for i in ids if not re.search(r'-\d{3}$', i)]
        return rank(ids, [r'^gemini-[\d.]+-flash$', r'^gemini-[\d.]+-pro$', r'^gemini-[\d.]+-flash-lite$',
                          r'^gemini-[\d.]+-(flash|pro).*preview', r'^gemma-.*-it$'],
                    ['embedding', 'image', 'tts', 'audio', 'live', 'aqa', 'robotics', 'computer-use',
                     'learnlm', 'nano', '-latest', 'thinking', 'exp-', '-1b-', '-4b-', 'e2b', 'e4b'])

    def complete(self, model, system, user):
        gemma = model.startswith('gemma')
        body = {'contents': [{'role': 'user', 'parts': [{'text': f'{system}\n\n{user}' if gemma else user}]}],
                'generationConfig': {'temperature': self.temperature}}
        if not gemma:
            body['system_instruction'] = {'parts': [{'text': system}]}
            if model not in self.no_json:
                body['generationConfig']['responseMimeType'] = 'application/json'
        r = http_json('POST', f'{self.base}/models/{model}:generateContent', self.hdr(), body)
        if r.get('promptFeedback', {}).get('blockReason'):
            raise ApiError('blocked', f"prompt blocked: {r['promptFeedback']['blockReason']}")
        cand = (r.get('candidates') or [{}])[0]
        reason = cand.get('finishReason')
        text = ''.join(p.get('text', '') for p in cand.get('content', {}).get('parts', []) if not p.get('thought'))
        if reason == 'MAX_TOKENS':
            raise ApiError('too_big', 'output truncated')
        if reason not in (None, 'STOP') or not text:
            raise ApiError('blocked', f'finishReason={reason}')
        return text


class OpenAICompat(Provider):
    base = models_url = ''
    prefs, exclude = [], []

    def hdr(self):
        return {'Authorization': f'Bearer {self.key}'}

    def discover(self):
        r = http_json('GET', self.models_url or f'{self.base}/models', self.hdr(), timeout=30)
        data = r.get('data', []) if isinstance(r, dict) else r
        return rank([m['id'] for m in data if m.get('id')], self.prefs, self.exclude)

    def complete(self, model, system, user):
        body = {'model': model, 'temperature': self.temperature,
                'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]}
        if model not in self.no_json:
            body['response_format'] = {'type': 'json_object'}
        if 'gpt-oss' in model:
            body['reasoning_effort'] = 'low'
        try:
            r = http_json('POST', f'{self.base}/chat/completions', self.hdr(), body)
        except ApiError as e:
            if e.kind == 'unsupported' and model not in self.no_json:
                self.no_json.add(model)
                return self.complete(model, system, user)
            raise
        if r.get('error'):
            err = r['error']
            code = err.get('code') if isinstance(err, dict) and isinstance(err.get('code'), int) else 500
            raise classify(code, json.dumps(err))
        choice = (r.get('choices') or [{}])[0]
        if choice.get('finish_reason') == 'length':
            raise ApiError('too_big', 'output truncated')
        text = (choice.get('message') or {}).get('content') or ''
        if not text:
            raise ApiError('blocked', f"empty reply (finish_reason={choice.get('finish_reason')})")
        return text


class Mistral(OpenAICompat):
    name, key_env, interval = 'mistral', 'MISTRAL_API_KEY', 2.0
    per_model = False
    base = 'https://api.mistral.ai/v1'
    prefs = [r'^mistral-large-latest$', r'^mistral-medium-latest$', r'^mistral-small-latest$']


class GitHub(OpenAICompat):
    name, key_env, interval = 'github', 'GITHUB_TOKEN', 8.0
    base = 'https://models.github.ai/inference'
    models_url = 'https://models.github.ai/catalog/models'
    prefs = [r'^openai/gpt-[\d.]+$', r'^openai/gpt-4o$', r'^deepseek/deepseek-v', r'^meta/llama-4-maverick']
    exclude = ['mini', 'nano', 'embedding']


class Groq(OpenAICompat):
    name, key_env, interval, max_chars = 'groq', 'GROQ_API_KEY', 3.0, 2500
    base = 'https://api.groq.com/openai/v1'
    prefs = [r'kimi-k2', r'gpt-oss-120b', r'llama-4-maverick', r'qwen3', r'llama-3\.3-70b']
    exclude = ['whisper', 'guard', 'tts', 'compound', 'playai', 'distil']


class Cerebras(OpenAICompat):
    name, key_env, interval, max_chars = 'cerebras', 'CEREBRAS_API_KEY', 3.0, 3000
    base = 'https://api.cerebras.ai/v1'
    prefs = [r'gpt-oss-120b', r'qwen-3-235b', r'llama-4-maverick', r'llama-3\.3-70b']


class OpenRouter(OpenAICompat):
    name, key_env, interval, max_chars = 'openrouter', 'OPENROUTER_API_KEY', 4.0, 4000
    per_model = False
    base = 'https://openrouter.ai/api/v1'
    prefs = [r'deepseek.*:free$', r'gemma-4-31b.*:free$', r'nemotron-3-ultra.*:free$', r'qwen3.*:free$',
             r'gemma-4.*:free$', r'llama-4.*:free$', r'gpt-oss.*:free$']
    exclude = ['coder', 'code', 'vl', 'math', 'guard', 'embed', 'safety', 'nano', 'inkling']


PROVIDERS = {p.name: p for p in (Gemini, Mistral, GitHub, Groq, Cerebras, OpenRouter)}


class Unavailable(Exception):
    """The model can't be used right now; it has been marked and the batch should go back in the queue."""


class Chain:
    """Ordered (provider, model) list. Workers claim a free model each, so several run at once."""

    def __init__(self, cfg, keys):
        self.cfg, self.keys = cfg, keys
        self.lock = threading.RLock()
        self.state = load_json(STATE, {})
        self.last, self.interval, self.auth_dead, self.claimed = {}, {}, set(), set()
        self.entries = self.discover(cfg, keys)

    @staticmethod
    def discover(cfg, keys):
        provs, entries = {}, []
        for name in cfg['provider_order']:
            cls = PROVIDERS.get(name)
            if not cls:
                continue
            # GROQ_API_KEY, GROQ_API_KEY_2, ...: each extra account becomes its own provider (groq, groq#2, ...)
            for env in sorted(k for k in keys if keys[k] and re.fullmatch(re.escape(cls.key_env) + r'(_\d+)?', k)):
                n = env[len(cls.key_env) + 1:]
                prov = cls(keys[env], cfg['temperature'])
                prov.name = f'{name}#{n}' if n else name
                provs[prov.name] = prov
        if cfg.get('chain'):
            for item in cfg['chain']:
                name, model = item.split(':', 1)
                if name in provs:
                    entries.append((provs[name], model))
            return entries
        for name, prov in provs.items():
            try:
                models = prov.discover()[:cfg['max_models_per_provider']]
            except ApiError as e:
                log(f'{name}: skipped, could not list models ({e})')
                continue
            entries += [(prov, m) for m in models]
        return entries

    def reload_if_keys_changed(self):
        keys = load_keys()
        if keys != self.keys:
            log('keys.env changed; rebuilding the model chain')
            entries = self.discover(self.cfg, keys)
            with self.lock:
                self.keys, self.entries, self.auth_dead = keys, entries, set()
            log('model chain: ' + ', '.join(self.label(p, m) for p, m in self.entries))

    @staticmethod
    def label(prov, model):
        return f'{prov.name}:{model}'

    @staticmethod
    def slot(prov, model):
        # providers whose free limit covers the whole account get one worker at a time
        return f'{prov.name}:{model}' if prov.per_model else prov.name

    def labels(self):
        return {self.label(p, m) for p, m in self.entries}

    def until(self, lab):
        return self.state.get(lab, {}).get('until', 0)

    def usable(self, prov, model):
        return prov.name not in self.auth_dead and self.until(self.label(prov, model)) <= time.time()

    def mark(self, lab, seconds, reason):
        with self.lock:
            self.state[lab] = {'until': time.time() + seconds, 'reason': str(reason)[:160]}
            save_json(STATE, self.state)

    def acquire(self, exclude=()):
        with self.lock:
            for prov, model in self.entries:
                s = self.slot(prov, model)
                if self.label(prov, model) not in exclude and s not in self.claimed and self.usable(prov, model):
                    self.claimed.add(s)
                    return prov, model
        return None

    def release(self, prov, model):
        with self.lock:
            self.claimed.discard(self.slot(prov, model))

    def next_free(self):
        with self.lock:
            times = [self.until(self.label(p, m)) for p, m in self.entries if p.name not in self.auth_dead]
        return min(times) if times else None

    def pace(self, lab, prov):
        gap = self.interval.get(lab, prov.interval) - (time.time() - self.last.get(lab, 0))
        if gap > 0:
            time.sleep(gap)
        self.last[lab] = time.time()

    def call(self, prov, model, system, user, validate):
        """One request with retries on this model. Raises Unavailable, or ApiError(bad/too_big/blocked)."""
        lab, limited, failures = self.label(prov, model), 0, 0
        while True:
            self.pace(lab, prov)
            try:
                return validate(prov.complete(model, system, user))
            except ApiError as e:
                e.model = lab
                if e.kind == 'minute':
                    limited += 1
                    self.interval[lab] = min(self.interval.get(lab, prov.interval) * 1.5, 60)
                    if limited <= 3:
                        log(f'{lab}: rate limited, waiting {e.wait:.0f}s')
                        time.sleep(e.wait + 1)
                        continue
                    log(f'{lab}: still rate limited, resting it for 3 min')
                    self.mark(lab, 180, e)
                elif e.kind == 'transient':
                    failures += 1
                    if failures < 3:
                        log(f'{lab}: {e}; retrying')
                        time.sleep(15 * failures)
                        continue
                    log(f'{lab}: keeps failing, resting it for 10 min')
                    self.mark(lab, 600, e)
                elif e.kind == 'busy':
                    log(f'{lab}: overloaded, resting it for 5 min')
                    self.mark(lab, 300, e)
                elif e.kind == 'exhausted':
                    log(f'{lab}: {e}, resting it until the quota resets')
                    self.mark(lab, (e.wait or 3600) + 60, e)
                elif e.kind in ('dead', 'unsupported'):
                    log(f'{lab}: unusable ({e}), dropping it for a week')
                    self.mark(lab, 7 * 86400, e)
                elif e.kind == 'auth':
                    log(f'{prov.name}: key rejected ({e}), skipping provider')
                    with self.lock:
                        self.auth_dead.add(prov.name)
                else:
                    raise   # bad / too_big / blocked: about this batch, not the model
                raise Unavailable() from e


# ---------------------------------------------------------------- EPUB handling

NS = {'opf': 'http://www.idpf.org/2007/opf', 'dc': 'http://purl.org/dc/elements/1.1/',
      'c': 'urn:oasis:names:tc:opendocument:xmlns:container', 'ncx': 'http://www.daisy.org/z3986/2005/ncx/'}
PARSER = etree.XMLParser(recover=True, huge_tree=True, resolve_entities=False)
TEXT_BLOCKS = {'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'blockquote', 'dd', 'dt', 'td', 'th',
               'figcaption', 'caption', 'div'}
ALL_BLOCKS = TEXT_BLOCKS | {'section', 'article', 'aside', 'header', 'footer', 'nav', 'table', 'tr', 'tbody',
                            'thead', 'ul', 'ol', 'dl', 'figure', 'pre', 'hr', 'body'}
SKIP_INSIDE = {'script', 'style', 'pre', 'code', 'math', 'svg', 'head'}
XML_LANG = '{http://www.w3.org/XML/1998/namespace}lang'


def local(el):
    return etree.QName(el).localname.lower() if isinstance(el.tag, str) else ''


def leaf_blocks(root):
    for el in root.iter():
        if local(el) not in TEXT_BLOCKS:
            continue
        if any(local(d) in ALL_BLOCKS for d in el.iterdescendants()):
            continue
        if any(local(a) in SKIP_INSIDE for a in el.iterancestors()):
            continue
        if LETTERS.search(''.join(el.itertext())):
            yield el


def encode(el):
    """Inner content as text with inline elements replaced by numbered <tN> tags."""
    tags = []

    def walk(node):
        out = []
        for ch in node:
            if isinstance(ch.tag, str):
                i = len(tags)
                if ''.join(ch.itertext()).strip():
                    tags.append((ch.tag, dict(ch.attrib), None))
                    out.append(f'<t{i}>{html.escape(ch.text or "", quote=False)}{walk(ch)}</t{i}>')
                else:     # anchors, images, line breaks: kept verbatim
                    tags.append((ch.tag, dict(ch.attrib), ch))
                    out.append(f'<t{i}/>')
            out.append(html.escape(ch.tail or '', quote=False))
        return ''.join(out)

    src = html.escape(el.text or '', quote=False) + walk(el)
    return re.sub(r'\s+', ' ', src).strip(), tags


def plain(s):
    return re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', '', s))).strip()


def append_text(dst, text):
    if not text:
        return
    if len(dst):
        dst[-1].tail = (dst[-1].tail or '') + text
    else:
        dst.text = (dst.text or '') + text


def rebuild(el, translated, tags):
    for ch in list(el):
        el.remove(ch)
    el.text = None
    used = set()
    try:
        frag = etree.fromstring('<x>' + re.sub(r'&(?!#?\w+;)', '&amp;', translated) + '</x>')
    except etree.XMLSyntaxError:
        frag = None

    def fill(dst, src):
        for c in src:
            m = re.fullmatch(r't(\d+)', c.tag) if isinstance(c.tag, str) else None
            idx = int(m.group(1)) if m else None
            if idx is None or idx >= len(tags) or idx in used:
                append_text(dst, c.text)
                fill(dst, c)
                append_text(dst, c.tail)
                continue
            used.add(idx)
            tag, attrib, orig = tags[idx]
            if orig is not None and not ''.join(c.itertext()).strip():
                new = copy.deepcopy(orig)
                dst.append(new)
            else:
                new = etree.SubElement(dst, tag, attrib)
                new.text = c.text
                fill(new, c)
            new.tail = c.tail

    if frag is None:
        el.text = plain(translated)
    else:
        el.text = frag.text
        fill(el, frag)
    for i, (_, _, orig) in enumerate(tags):    # never lose anchors that links point at
        if orig is not None and i not in used:
            new = copy.deepcopy(orig)
            new.tail = None
            el.append(new)


class Seg:
    __slots__ = ('doc', 'el', 'src', 'tags', 'key')

    def __init__(self, doc, el):
        self.doc, self.el = doc, el
        self.src, self.tags = encode(el)
        self.key = hashlib.sha1(self.src.encode()).hexdigest()


class Book:
    def __init__(self, path, skip_files=()):
        self.path = Path(path)
        self.zip = zipfile.ZipFile(self.path)
        container = etree.fromstring(self.zip.read('META-INF/container.xml'))
        self.opf_name = container.find('.//c:rootfile', NS).get('full-path')
        self.opf = etree.fromstring(self.zip.read(self.opf_name), PARSER)
        base = posixpath.dirname(self.opf_name)
        manifest = {i.get('id'): i for i in self.opf.iterfind('.//opf:manifest/opf:item', NS)}
        self.docs, self.ncx_name = {}, None
        for item in manifest.values():
            if item.get('media-type') == 'application/x-dtbncx+xml':
                self.ncx_name = posixpath.normpath(posixpath.join(base, urllib.parse.unquote(item.get('href'))))
        for ref in self.opf.iterfind('.//opf:spine/opf:itemref', NS):
            item = manifest.get(ref.get('idref'))
            if item is None or 'html' not in (item.get('media-type') or ''):
                continue
            name = posixpath.normpath(posixpath.join(base, urllib.parse.unquote(item.get('href'))))
            root = etree.fromstring(self.zip.read(name), PARSER)
            if root is not None:
                self.docs[name] = root
        skip = [re.compile(p) for p in skip_files]
        self.segs = [Seg(name, el) for name, root in self.docs.items()
                     if not any(p.search(name) for p in skip) for el in leaf_blocks(root)]

    def unique(self):
        seen, out = set(), []
        for s in self.segs:
            if s.key not in seen:
                seen.add(s.key)
                out.append(s)
        return out


def write_epub(book, cache, out, bilingual=False):
    """Apply cached translations to a freshly loaded book and write it. Returns untranslated count."""
    missing, headings = 0, {}
    for s in book.segs:
        t = cache.get(s.key)
        if t is None:
            missing += 1
            continue
        target = s.el
        if bilingual:
            target = copy.deepcopy(s.el)
            target.tail = None
            for d in target.iter():
                d.attrib.pop('id', None)
            s.el.addnext(target)
        rebuild(target, t, s.tags)
        headings[plain(s.src)] = re.sub(r'\s+', ' ', ''.join(target.itertext())).strip()

    files = {}
    for name, root in book.docs.items():
        root.set('lang', 'hi')
        root.set(XML_LANG, 'hi')
        files[name] = etree.tostring(root.getroottree(), xml_declaration=True, encoding='utf-8')
    for lang in book.opf.iterfind('.//dc:language', NS):
        lang.text = 'hi'
    title = book.opf.find('.//dc:title', NS)
    if title is not None and 'हिंदी' not in (title.text or ''):
        title.text = f'{title.text} (हिंदी)'
    files[book.opf_name] = etree.tostring(book.opf.getroottree(), xml_declaration=True, encoding='utf-8')
    if book.ncx_name and book.ncx_name in book.zip.namelist():
        ncx = etree.fromstring(book.zip.read(book.ncx_name), PARSER)
        for t in ncx.iterfind('.//ncx:navLabel/ncx:text', NS):
            key = re.sub(r'\s+', ' ', t.text or '').strip()
            if key in headings:
                t.text = headings[key]
        files[book.ncx_name] = etree.tostring(ncx.getroottree(), xml_declaration=True, encoding='utf-8')

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp')
    with zipfile.ZipFile(tmp, 'w') as z:
        z.writestr(zipfile.ZipInfo('mimetype'), b'application/epub+zip')
        for info in book.zip.infolist():
            if info.filename != 'mimetype':
                z.writestr(info.filename, files.get(info.filename) or book.zip.read(info.filename),
                           compress_type=zipfile.ZIP_DEFLATED)
    os.replace(tmp, out)
    return missing


# ---------------------------------------------------------------- translation

class Cache:
    def __init__(self, path):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute('pragma journal_mode=wal')
        self.db.execute('create table if not exists seg (key text primary key, src text, out text, model text, ts real)')

    def get(self, key):
        r = self.db.execute('select out from seg where key = ?', (key,)).fetchone()
        return r[0] if r else None

    def keys(self):
        return {r[0] for r in self.db.execute('select key from seg')}

    def put(self, rows):
        with self.db:
            self.db.executemany('insert or replace into seg values (?, ?, ?, ?, ?)',
                                [(s.key, s.src, t, m, time.time()) for s, t, m in rows])

    def models(self):
        return self.db.execute('select model, count(*) from seg group by model order by 2 desc').fetchall()


def read_terms(path):
    terms = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#'):
                k, _, v = line.partition('\t') if '\t' in line else line.partition('=')
                if k.strip() and v.strip():
                    terms[k.strip()] = v.strip()
    return terms


class Glossary:
    def __init__(self, user_path, auto_path):
        self.user, self.auto, self.auto_path = read_terms(user_path), read_terms(auto_path), auto_path
        self.known = {k.lower() for k in self.user} | {k.lower() for k in self.auto}

    def relevant(self, text, limit=80):
        low = text.lower()
        out = list(self.user.items()) + [(k, v) for k, v in self.auto.items() if k not in self.user]
        return [(k, v) for k, v in out if re.search(r'\b' + re.escape(k.lower()) + r'\b', low)][:limit]

    def add(self, terms):
        new = []
        for k, v in (terms if isinstance(terms, dict) else {}).items():
            k, v = str(k).strip(), str(v).strip()
            if (k.lower() in self.known or not LETTERS.search(k) or not DEVANAGARI.search(v)
                    or len(k) > 60 or re.search(r'[\t\n=]', k + v)):
                continue
            self.known.add(k.lower())
            self.auto[k] = v
            new.append(f'{k}\t{v}\n')
        if new:
            with open(self.auto_path, 'a') as f:
                f.writelines(new)


SYSTEM = """You are an expert literary translator who translates English books into Hindi for publication.
You are translating {style}. Write in {register}.

Rules:
- Translate every segment completely and faithfully: no summarising, no omissions, no additions, no notes.
- Keep the author's voice, tone and rhythm. Render idioms with natural Hindi equivalents, not word for word.
- Write names, places and titles in Devanagari. Use the glossary spellings exactly, and stay consistent with the context.
- Keep numbers, years and dates exact (use international digits, e.g. 1757).
- Segments may contain inline tags such as <t3>...</t3> or <t5/>. Keep every tag, placed around the matching
  translated words. Do not add, drop or rename tags.
- The input is JSON {{"segments": [...]}}. Reply with ONLY a JSON object:
  {{"t": [one Hindi translation per input segment, same order and count],
    "terms": {{"English name or term": "Devanagari spelling used"}}}}
  "terms" lists up to 15 proper nouns or special terms from these segments that are not in the glossary."""


def build_prompt(segs, ctx, gloss):
    parts = []
    terms = gloss.relevant(' '.join(plain(s.src) for s in segs))
    if terms:
        parts.append('Glossary (use these spellings):\n' + '\n'.join(f'{k} → {v}' for k, v in terms))
    if ctx:
        parts.append('Previous passage, for context only (do not translate it again):\n'
                     + '\n'.join(f'EN: {a[:500]}\nHI: {b[:500]}' for a, b in ctx))
    parts.append(json.dumps({'segments': [s.src for s in segs]}, ensure_ascii=False))
    return '\n\n'.join(parts)


def parse_reply(text, segs):
    t = re.sub(r'^\s*```(?:json)?\s*|\s*```\s*$', '', text.strip())
    obj = None
    for candidate in (t, (re.search(r'\{.*\}', t, re.S) or [None])[0]):
        try:
            obj = json.loads(candidate)
            break
        except (TypeError, ValueError):
            continue
    out = obj.get('t') if isinstance(obj, dict) else obj
    if not isinstance(out, list):
        raise ApiError('bad', 'reply was not the expected JSON')
    if len(out) != len(segs):
        raise ApiError('bad', f'expected {len(segs)} segments, got {len(out)}')
    out = [str(x).strip() for x in out]
    untranslated = sum(1 for s, x in zip(segs, out)
                       if len(LETTERS.findall(plain(s.src))) > 20 and not DEVANAGARI.search(x))
    if untranslated > max(1, len(segs) // 5) or any(not x for x in out):
        raise ApiError('bad', f'{untranslated} segment(s) came back untranslated or empty')
    return out, obj.get('terms') if isinstance(obj, dict) else {}


class Job:
    """A queue of segments shared by worker threads; each worker translates with its own model."""

    def __init__(self, segs, todo, cache, gloss, chain, cfg):
        self.order = {s.key: i for i, s in enumerate(segs)}
        self.segs, self.queue = segs, list(todo)
        self.cache, self.gloss, self.chain, self.cfg = cache, gloss, chain, cfg
        self.lock = threading.Lock()
        self.failed = {}            # segment key -> model labels that couldn't translate it
        self.inflight = 0
        self.finished = len(segs) - len(todo)
        self.given_up = 0

    def take(self, limit, lab):
        with self.lock:
            labels = self.chain.labels()
            batch, size, i = [], 0, 0
            while i < len(self.queue) and len(batch) < self.cfg['max_segments']:
                s = self.queue[i]
                tried = self.failed.get(s.key, set())
                if labels <= tried:
                    log('no model could translate a segment; leaving it in English')
                    del self.queue[i]
                    self.given_up += 1
                    continue
                if lab in tried or batch and size + len(s.src) > limit:
                    if batch:
                        break
                    i += 1
                    continue
                batch.append(s)
                size += len(s.src)
                del self.queue[i]
            self.inflight += bool(batch)
            return batch

    def give_back(self, segs):
        with self.lock:
            self.queue[0:0] = segs
            self.queue.sort(key=lambda s: self.order[s.key])

    def context(self, seg):
        i = self.order[seg.key]
        with self.lock:
            prev = [(s, self.cache.get(s.key)) for s in self.segs[max(0, i - 2):i]]
        return [(plain(s.src), plain(t)) for s, t in prev if t]

    def translate(self, prov, model, batch):
        lab = Chain.label(prov, model)
        system = SYSTEM.format(**self.cfg)
        while batch:
            user = build_prompt(batch, self.context(batch[0]), self.gloss)
            try:
                texts, terms = self.chain.call(prov, model, system, user, lambda t, b=batch: parse_reply(t, b))
            except Unavailable:
                self.give_back(batch)
                return
            except ApiError as e:
                if len(batch) > 1:
                    log(f'{lab}: {e}; splitting batch of {len(batch)}')
                    mid = len(batch) // 2
                    self.give_back(batch[mid:])
                    batch = batch[:mid]
                    continue
                log(f'{lab}: {e}; leaving this segment to another model')
                with self.lock:
                    self.failed.setdefault(batch[0].key, set()).add(lab)
                self.give_back(batch)
                return
            with self.lock:
                self.gloss.add(terms)
                self.cache.put([(s, t, lab) for s, t in zip(batch, texts)])
                self.finished += len(batch)
                n, total = self.finished, len(self.segs)
            log(f'{n}/{total} ({100 * n / total:.1f}%) +{len(batch)} via {lab}')
            return

    def worker(self):
        while True:
            with self.lock:
                if not self.queue and not self.inflight:
                    return
            got = self.chain.acquire()
            if not got:
                time.sleep(20)
                continue
            prov, model = got
            batch = []
            try:
                batch = self.take(min(self.cfg['merge_chars'], prov.max_chars or 10 ** 9), Chain.label(prov, model))
                if batch:
                    self.translate(prov, model, batch)
            finally:
                if batch:
                    with self.lock:
                        self.inflight -= 1
                self.chain.release(prov, model)
            if not batch:
                time.sleep(5)


# ---------------------------------------------------------------- commands

def paths(book):
    work = WORK / slug(book)
    work.mkdir(parents=True, exist_ok=True)
    return work, OUTPUT / f'{Path(book).stem} - Hindi.epub'


def notify(msg):
    subprocess.run(['osascript', '-e', f'display notification {json.dumps(msg)} with title "Hindi translator"'],
                   capture_output=True)


def cmd_run(args):
    book_path = str(Path(args.book).expanduser().resolve())
    WORK.mkdir(exist_ok=True)
    save_json(WORK / 'current.json', {'book': book_path, 'bilingual': args.bilingual})
    subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())])     # keep the Mac awake while running
    cfg = load_config(book_path)
    work, out = paths(book_path)
    book = Book(book_path, cfg['skip_files'])
    cache = Cache(work / 'cache.sqlite')
    gloss = Glossary(work / 'glossary.tsv', work / 'auto_glossary.tsv')
    done = cache.keys()
    segs = book.unique()
    todo = [s for s in segs if s.key not in done]
    if args.limit:
        todo = todo[:args.limit]
    log(f'{len(segs) - len(todo)}/{len(segs)} segments already translated; '
        f'{sum(len(s.src) for s in todo):,} characters to go')
    chain = Chain(cfg, load_keys())
    if not chain.entries:
        sys.exit('No usable models. Add keys to keys.env.')
    log('model chain: ' + ', '.join(Chain.label(p, m) for p, m in chain.entries))

    job = Job(segs, todo, cache, gloss, chain, cfg)
    workers = [threading.Thread(target=job.worker, daemon=True) for _ in range(cfg['workers'])]
    for w in workers:
        w.start()
    last_build, last_count, idle_since = time.time(), job.finished, time.time()
    while any(w.is_alive() for w in workers):
        time.sleep(60)
        chain.reload_if_keys_changed()
        if job.finished != last_count:
            last_count, idle_since = job.finished, time.time()
        elif time.time() - idle_since > 600 and not chain.claimed:
            nxt = chain.next_free()
            log('every model is busy or out of quota' +
                (f'; next one free at {time.strftime("%a %H:%M", time.localtime(nxt))}' if nxt else ''))
            idle_since = time.time()
        if time.time() - last_build > cfg['build_every'] * 60:
            write_epub(Book(book_path, cfg['skip_files']), cache, out, args.bilingual)
            last_build = time.time()

    missing = write_epub(Book(book_path, cfg['skip_files']), cache, out, args.bilingual)
    msg = f'Done: {out.name}' + (f' ({missing} paragraphs still in English)' if missing else '')
    log(msg, f'-> {out}')
    notify(msg)


def cmd_build(args):
    book_path = str(Path(args.book).expanduser().resolve())
    cfg = load_config(book_path)
    work, out = paths(book_path)
    missing = write_epub(Book(book_path, cfg['skip_files']), Cache(work / 'cache.sqlite'), out, args.bilingual)
    print(f'wrote {out} ({missing} paragraphs untranslated)')


def cmd_status(args):
    book_path = args.book or load_json(WORK / 'current.json', {}).get('book')
    if not book_path:
        sys.exit('No book yet; pass one.')
    cfg = load_config(book_path)
    work, out = paths(book_path)
    cache = Cache(work / 'cache.sqlite')
    segs = Book(book_path, cfg['skip_files']).unique()
    done = cache.keys()
    n = sum(1 for s in segs if s.key in done)
    left = sum(len(s.src) for s in segs if s.key not in done)
    print(f'{Path(book_path).name}\n  {n}/{len(segs)} segments ({100 * n / max(1, len(segs)):.1f}%), '
          f'{left:,} characters left\n  output: {out}')
    for model, count in cache.models():
        print(f'  {count:6} by {model}')
    now = time.time()
    for lab, v in sorted(load_json(STATE, {}).items()):
        if v.get('until', 0) > now:
            print(f'  {lab}: unavailable until {time.strftime("%a %H:%M", time.localtime(v["until"]))}  ({v["reason"][:70]})')


def cmd_models(args):
    chain = Chain(load_config(), load_keys())
    now = time.time()
    for prov, model in chain.entries:
        lab = Chain.label(prov, model)
        u = chain.until(lab)
        print(f'  {lab:55} ' + (f'unavailable until {time.strftime("%a %H:%M", time.localtime(u))}' if u > now else 'ready'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('run')
    p.add_argument('book')
    p.add_argument('--bilingual', action='store_true', help='keep the English paragraph above each Hindi one')
    p.add_argument('--limit', type=int, help='translate only N segments (for testing)')
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser('build')
    p.add_argument('book')
    p.add_argument('--bilingual', action='store_true')
    p.set_defaults(fn=cmd_build)
    p = sub.add_parser('status')
    p.add_argument('book', nargs='?')
    p.set_defaults(fn=cmd_status)
    sub.add_parser('models').set_defaults(fn=cmd_models)
    args = ap.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
