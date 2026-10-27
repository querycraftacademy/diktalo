"""A kész szöveg bejuttatása az aktív ablakba (vágólap + beillesztés, vagy gépelés).

Windows: keyboard csomag, Ctrl+V. macOS: pynput, Cmd+V. Linux X11: pynput, Ctrl+V.
Linux Wayland: a programok nem szimulálhatnak billentyűt, ezért a wtype vagy a
ydotool parancsot hívjuk. Ha egyik sincs, a szöveg a vágólapon marad, és a
felhasználó illeszti be.
"""
import os
import shutil
import subprocess
import sys
import time

import pyperclip

IS_WINDOWS = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"


def is_wayland() -> bool:
    return sys.platform.startswith("linux") and (
        os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland" or bool(os.environ.get("WAYLAND_DISPLAY"))
    )


def paste_shortcut() -> str:
    return "Cmd+V" if IS_MAC else "Ctrl+V"


def wayland_tool() -> str | None:
    """A Wayland alatt billentyűt szimuláló eszköz neve, ha telepítve van."""
    for tool in ("wtype", "ydotool"):
        if shutil.which(tool):
            return tool
    return None


_clipboard_ready = False


def _init_clipboard():
    """Linuxon kifejezetten a telepített vágólap-eszközt kérjük (a pyperclip találgatása
    WSL-szerű környezetben a Windows clip.exe-jére futna)."""
    global _clipboard_ready
    if _clipboard_ready or not sys.platform.startswith("linux"):
        return
    _clipboard_ready = True
    order = (("wl-copy", "wl-clipboard"),) if is_wayland() else ()
    for exe, name in order + (("xclip", "xclip"), ("xsel", "xsel")):
        if shutil.which(exe):
            try:
                pyperclip.set_clipboard(name)
                return
            except Exception:
                continue


def _run(cmd: list[str]) -> bool:
    try:
        return subprocess.run(cmd, capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _press_paste() -> bool:
    if IS_WINDOWS:
        import keyboard

        keyboard.send("ctrl+v")
        return True
    if is_wayland():
        tool = wayland_tool()
        if tool == "wtype":
            return _run(["wtype", "-M", "ctrl", "v", "-m", "ctrl"])
        if tool == "ydotool":
            # 29 = bal Ctrl, 47 = V (Linux-billentyűkódok); a régi ydotool a "ctrl+v" alakot érti
            return _run(["ydotool", "key", "29:1", "47:1", "47:0", "29:0"]) or _run(["ydotool", "key", "ctrl+v"])
        return False
    from pynput.keyboard import Controller, Key

    kb = Controller()
    with kb.pressed(Key.cmd if IS_MAC else Key.ctrl):
        kb.press("v")
        kb.release("v")
    return True


def _type(text: str) -> bool:
    if IS_WINDOWS:
        import keyboard

        keyboard.write(text, delay=0.004)
        return True
    if is_wayland():
        tool = wayland_tool()
        if tool == "wtype":
            return _run(["wtype", text])
        if tool == "ydotool":
            return _run(["ydotool", "type", text])
        return False
    from pynput.keyboard import Controller

    Controller().type(text)
    return True


def deliver(
    text: str,
    method: str = "paste",
    restore_clipboard: bool = False,
    append_space: bool = True,
) -> bool:
    """True, ha a szöveg bekerült az ablakba; False, ha csak a vágólapra került."""
    if not text:
        return True
    if append_space:
        # Szóköz a végére, hogy a következő diktálás ne ragadjon az előzőhöz
        text = text + " "
    # Várunk, hogy a gyorsbillentyű minden módosítója felengedődjön,
    # különben a Ctrl+V-ből Ctrl+valami más lehet.
    time.sleep(0.15)
    if method == "type" and _type(text):
        return True

    _init_clipboard()
    old = None
    if restore_clipboard:
        try:
            old = pyperclip.paste()
        except Exception:
            old = None
    pyperclip.copy(text)
    time.sleep(0.08)
    if not _press_paste():
        return False  # a szöveg a vágólapon marad
    if restore_clipboard and old is not None:
        time.sleep(0.4)
        pyperclip.copy(old)
    return True
