"""A kész szöveg bejuttatása az aktív ablakba (vágólap + Ctrl+V vagy gépelés)."""
import time

import keyboard
import pyperclip


def deliver(
    text: str,
    method: str = "paste",
    restore_clipboard: bool = False,
    append_space: bool = True,
):
    if not text:
        return
    if append_space:
        # Szóköz a végére, hogy a következő diktálás ne ragadjon az előzőhöz
        text = text + " "
    # Várunk, hogy a gyorsbillentyű minden módosítója felengedődjön,
    # különben a Ctrl+V-ből Ctrl+valami más lehet.
    time.sleep(0.15)
    if method == "type":
        keyboard.write(text, delay=0.004)
        return

    old = None
    if restore_clipboard:
        try:
            old = pyperclip.paste()
        except Exception:
            old = None
    pyperclip.copy(text)
    time.sleep(0.08)
    keyboard.send("ctrl+v")
    if restore_clipboard and old is not None:
        time.sleep(0.4)
        pyperclip.copy(old)
