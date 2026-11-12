# hindi-translator

Translates an EPUB into Hindi with large language models, falling back across free-tier models from
several providers when one hits a quota, goes down or returns a bad answer. Runs unattended and
resumes where it stopped.

Built AI-assisted with Claude Code: I set the requirements and ran it on a full-length book;
the code was written in that session.

## How it works

- **Model chain.** Discovers available models from each provider that has a key (Gemini, Mistral,
  GitHub Models, Groq, Cerebras, OpenRouter) and ranks them by preference.
- **Failure handling.** Every API error is classified (rate limit, daily quota, overloaded, dead
  model, bad key, refused, too large). Each class has its own response: wait and retry, rest the
  model until its quota resets, drop it for a week, or skip the provider. Rest periods persist in
  `state.json`, so a restart doesn't hammer a model that is out of quota.
- **Parallel workers.** Several threads each claim a free model and translate a different batch.
  Providers whose free limit covers the whole account get one worker at a time.
- **Structured output and validation.** The model must reply with JSON holding one translation per
  input segment. Replies with the wrong count, empty segments or untranslated English are rejected;
  a failing batch is split in half and retried, and a segment no model can translate stays in English.
- **Grounding for consistency.** Each request carries the glossary terms that appear in the batch
  (user-supplied plus terms the model proposed earlier) and the previous passage with its translation.
- **Markup preserved.** Inline elements become numbered `<tN>` tags that the model must keep, then
  are rebuilt into the original XHTML. Link anchors are never lost.
- **Resumable.** Translations are cached in SQLite keyed by a hash of the source segment, and the
  output EPUB is rewritten every few minutes.

## Usage

```
cp keys.env.example keys.env      # add whichever keys you have; all providers have free tiers
./run.sh start BOOK.epub          # background run; add --bilingual to keep the English above each paragraph
./run.sh status | log | models | stop
```

Requires Python 3.10+ and `lxml`. `run.sh` expects a virtualenv at `.venv`. macOS only for the
`caffeinate` and notification calls.
