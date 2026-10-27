"""A futó Diktáló vezérlése parancssorból: felvétel indítása, leállítása, kilépés.

Erre ott van szükség, ahol a program maga nem figyelheti a billentyűzetet
(Linux Wayland alatt), vagy ha a gyorsbillentyűt inkább az operációs rendszerre
bíznád. Az asztali környezet billentyűparancsát erre kell kötni:

  python control.py toggle     (Linuxon/Macen: ./toggle.sh)

A futó program egy helyi (127.0.0.1) portot nyit, a számát a .diktalo.port
fájlba írja; ez a szkript oda küldi a parancsot. Kívülről nem érhető el.
"""
from __future__ import annotations

import socket
import sys
import threading
from pathlib import Path
from typing import Callable

APP_DIR = Path(__file__).resolve().parent
PORT_FILE = APP_DIR / ".diktalo.port"
COMMANDS = ("toggle", "start", "stop", "quit")


class ControlServer:
    def __init__(self, handlers: dict[str, Callable[[], None]]):
        self.handlers = handlers
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(4)
        self.port = self._sock.getsockname()[1]
        PORT_FILE.write_text(str(self.port), encoding="utf-8")
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                conn, _addr = self._sock.accept()
            except OSError:
                return  # bezártuk
            with conn:
                try:
                    conn.settimeout(2)
                    cmd = conn.recv(64).decode("utf-8", "replace").strip().lower()
                    handler = self.handlers.get(cmd)
                    conn.sendall(b"ok\n" if handler else b"ismeretlen parancs\n")
                except OSError:
                    continue
            if handler:
                # külön szálon, hogy a hosszabb teendő (pl. leállítás) ne tartsa fel a következő parancsot
                threading.Thread(target=handler, daemon=True).start()

    def close(self):
        try:
            self._sock.close()
        finally:
            try:
                if PORT_FILE.read_text(encoding="utf-8").strip() == str(self.port):
                    PORT_FILE.unlink()
            except OSError:
                pass


def send(command: str, timeout: float = 3) -> str:
    """Parancs küldése a futó programnak. Hibánál RuntimeError."""
    try:
        port = int(PORT_FILE.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        raise RuntimeError("A Diktáló nem fut (nincs .diktalo.port fájl). Előbb indítsd el.") from None
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout) as s:
            s.sendall(command.encode("utf-8") + b"\n")
            return s.recv(64).decode("utf-8", "replace").strip()
    except OSError:
        raise RuntimeError("A Diktáló nem válaszol; valószínűleg nem fut. Indítsd el újra.") from None


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1 or args[0].lower() not in COMMANDS:
        print(f"Használat: python control.py {{{'|'.join(COMMANDS)}}}")
        return 2
    try:
        reply = send(args[0].lower())
    except RuntimeError as e:
        print(e)
        return 1
    return 0 if reply == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
