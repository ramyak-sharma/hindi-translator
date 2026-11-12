#!/bin/bash
# Background control for translate.py.
#   ./run.sh start BOOK.epub [--bilingual]   start (or resume) in the background
#   ./run.sh stop | status | log | models
cd "$(dirname "$0")"
PY=.venv/bin/python
PID=work/pid
running() { [ -f $PID ] && kill -0 "$(cat $PID)" 2>/dev/null; }
case "$1" in
  start)
    shift
    running && { echo "already running (pid $(cat $PID))"; exit 1; }
    mkdir -p work
    nohup "$PY" translate.py run "$@" >> work/run.log 2>&1 &
    echo $! > $PID
    echo "started (pid $!). Follow with: ./run.sh log" ;;
  stop)
    running && kill "$(cat $PID)" && echo stopped || echo "not running"
    rm -f $PID ;;
  status)
    "$PY" translate.py status
    running && echo "  running (pid $(cat $PID))" || echo "  not running"
    echo; tail -5 work/run.log 2>/dev/null ;;
  log) tail -f work/run.log ;;
  models) "$PY" translate.py models ;;
  *) sed -n '2,4p' "$0" ;;
esac
