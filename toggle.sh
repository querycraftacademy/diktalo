#!/usr/bin/env bash
# A futó Diktáló felvételének indítása/leállítása. Ezt kösd az asztali környezet
# egy billentyűparancsára (Linux Wayland alatt ez az egyetlen út, lásd README).
# Más parancsok: ./toggle.sh start | stop | quit
cd "$(dirname "$0")" || exit 1
exec .venv/bin/python control.py "${1:-toggle}"
