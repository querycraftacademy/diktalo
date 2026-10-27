"""Globális gyorsbillentyű: Windowson a keyboard csomaggal, Macen és Linuxon pynputtal.

A config "hotkey" mezője mindhárom rendszeren ugyanazt a szintaxist használja
("ctrl+windows", "f9", "right shift"). A "windows" (más néven "cmd", "super")
Macen a Command, Linuxon a Super billentyű, így az alapérték, a "ctrl+windows"
Macen Ctrl+Cmd, Linuxon Ctrl+Super lesz.

Linuxon Wayland alatt egy program nem figyelheti a billentyűzetet. Ott a
gyorsbillentyűt az asztali környezet billentyűparancsaiban kell beállítani,
a toggle.sh-ra (lásd control.py és README).
"""
from __future__ import annotations

import queue
import sys
import threading
from typing import Callable

_ALIASES = {
    "control": "ctrl",
    "option": "alt",
    "opt": "alt",
    "altgr": "alt_gr",
    "alt gr": "alt_gr",
    "windows": "cmd",
    "win": "cmd",
    "command": "cmd",
    "super": "cmd",
    "meta": "cmd",
    "escape": "esc",
    "return": "enter",
    "del": "delete",
    "pgup": "page_up",
    "pgdn": "page_down",
    "capslock": "caps_lock",
}


def parse_hotkey(text: str) -> list[str]:
    """ "ctrl+windows" -> ["ctrl", "cmd"]; "right shift" -> ["shift_r"]. """
    tokens = []
    for part in (text or "").split("+"):
        name = " ".join(part.strip().lower().split())
        if not name:
            raise ValueError(f"Érvénytelen gyorsbillentyű: {text!r}")
        side = ""
        for prefix, suffix in (("left ", "_l"), ("right ", "_r")):
            if name.startswith(prefix):
                name, side = name[len(prefix):], suffix
        name = _ALIASES.get(name, name).replace(" ", "_")
        tokens.append(name + side)
    return tokens


def key_names(key) -> set[str]:
    """Egy pynput-billentyű nevei: a Key.ctrl_r a "ctrl_r" és a "ctrl" tokent is kielégíti."""
    name = getattr(key, "name", None)
    if name:  # pynput.keyboard.Key
        names = {name}
        if name.endswith(("_l", "_r")):
            names.add(name[:-2])
        if name == "alt_gr":
            names.update({"alt", "alt_r"})
        return names
    char = getattr(key, "char", None)
    return {char.lower()} if char else set()


def _key_id(key):
    name = getattr(key, "name", None)
    if name:
        return name
    return getattr(key, "vk", None) or getattr(key, "char", None) or repr(key)


class ComboTracker:
    """Melyik billentyű van lenyomva, és ettől teljes-e a kombináció (tiszta logika, tesztelhető)."""

    def __init__(self, tokens: list[str]):
        self.tokens = list(tokens)
        self._down: dict = {}
        self.active = False

    def _complete(self) -> bool:
        held = set().union(*self._down.values()) if self._down else set()
        return all(t in held for t in self.tokens)

    def press(self, key_id, names: set[str]) -> bool:
        """True, ha ettől a lenyomástól lett teljes a kombináció (az ismétlődő lenyomás nem számít)."""
        self._down[key_id] = set(names)
        if not self.active and self._complete():
            self.active = True
            return True
        return False

    def release(self, key_id) -> bool:
        """True, ha egy kombinációbeli billentyű felengedése bontotta meg az aktív kombinációt."""
        names = self._down.pop(key_id, set())
        if self.active and any(t in names for t in self.tokens) and not self._complete():
            self.active = False
            return True
        return False


def _register_keyboard(hotkey, mode, on_start, on_stop, on_toggle) -> Callable[[], None]:
    import keyboard

    if mode == "hold":
        # hold mód: kombináció lenyomása indít, bármelyik tag felengedése leállít
        keyboard.add_hotkey(hotkey, on_start)
        for key in hotkey.split("+"):
            keyboard.on_release_key(key.strip(), lambda e: on_stop(), suppress=False)
    else:
        keyboard.add_hotkey(hotkey, on_toggle)
    return keyboard.unhook_all


def _register_pynput(hotkey, mode, on_start, on_stop, on_toggle) -> Callable[[], None]:
    from pynput import keyboard as kb

    tokens = parse_hotkey(hotkey)
    for t in tokens:
        if len(t) > 1 and not hasattr(kb.Key, t):
            raise ValueError(f"Ismeretlen billentyű a gyorsbillentyűben: {t!r} ({hotkey!r})")
    tracker = ComboTracker(tokens)
    # A figyelő szálát nem tartjuk fel (macOS letiltja a lassú eseményfigyelőt):
    # a teendők egy saját szálon, sorban futnak.
    jobs: queue.Queue = queue.Queue()

    def worker():
        while True:
            job = jobs.get()
            if job is None:
                return
            try:
                job()
            except Exception as e:  # noqa: BLE001
                print(f"Gyorsbillentyű-hiba: {e}")

    threading.Thread(target=worker, daemon=True).start()

    def on_press(key):
        if tracker.press(_key_id(key), key_names(key)):
            jobs.put(on_start if mode == "hold" else on_toggle)

    def on_release(key):
        if tracker.release(_key_id(key)) and mode == "hold":
            jobs.put(on_stop)

    listener = kb.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    def stop():
        listener.stop()
        jobs.put(None)

    return stop


def register(hotkey: str, mode: str, on_start, on_stop, on_toggle) -> Callable[[], None]:
    """Beállítja a gyorsbillentyűt; a visszaadott függvény leállítja a figyelést."""
    if sys.platform == "win32":
        return _register_keyboard(hotkey, mode, on_start, on_stop, on_toggle)
    return _register_pynput(hotkey, mode, on_start, on_stop, on_toggle)
