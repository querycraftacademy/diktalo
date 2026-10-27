#!/usr/bin/env bash
# Diktáló indítása macOS-en és Linuxon.
cd "$(dirname "$0")" || exit 1
if [ ! -x ".venv/bin/python" ]; then
    echo "Előbb futtasd az ./install.sh szkriptet."
    exit 1
fi
exec .venv/bin/python app.py "$@"
