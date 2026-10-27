"""
Diktáló – helyi, internet nélküli beszédfelismerés Windowsra, macOS-re és Linuxra.

  gyorsbillentyű nyomva tartva -> mikrofonfelvétel -> Whisper (a gépeden) ->
  beillesztés oda, ahol a kurzor áll.

A Whisper azon a hardveren fut, ami a gépedben van: NVIDIA GPU-n faster-whisperrel,
Intel GPU-n OpenVINO-val, AMD GPU-n whisper.cpp Vulkannal, Apple Silicon Macen
mlx-whisperrel, ezek híján a processzoron.

Indítás:  Windows: start.bat   macOS/Linux: ./start.sh
"""
import datetime
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import config as config_mod
import control
import hardware
import hotkeys
import output as output_mod
from backends import create_transcriber
from corrector import Corrector
from recorder import SAMPLE_RATE, Recorder, device_name, level_db, looks_like_webcam, normalize

STATE_IDLE = "idle"
STATE_RECORDING = "recording"
STATE_PROCESSING = "processing"
STATE_LABELS = {STATE_IDLE: "várakozik", STATE_RECORDING: "felvétel", STATE_PROCESSING: "átírás"}

COLORS = {
    STATE_IDLE: (90, 90, 90),
    STATE_RECORDING: (220, 50, 50),
    STATE_PROCESSING: (240, 170, 30),
}


def make_icon(color):
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((8, 8, 56, 56), fill=color + (255,))
    # kis mikrofon sziluett
    d.rounded_rectangle((27, 18, 37, 36), radius=5, fill=(255, 255, 255, 230))
    d.arc((22, 26, 42, 44), start=0, end=180, fill=(255, 255, 255, 230), width=3)
    d.line((32, 44, 32, 50), fill=(255, 255, 255, 230), width=3)
    return img


def beep(freq, dur=120):
    try:
        if sys.platform == "win32":
            import winsound

            threading.Thread(target=winsound.Beep, args=(freq, dur), daemon=True).start()
            return
        import sounddevice as sd

        rate = 22050
        t = np.arange(int(rate * dur / 1000)) / rate
        fade = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.01)  # 10 ms be- és kiúsztatás, nem pattog
        tone = (0.2 * np.sin(2 * np.pi * freq * t) * fade).astype(np.float32)
        threading.Thread(target=sd.play, args=(tone, rate), daemon=True).start()
    except Exception:
        pass


def on_main_thread(fn):
    """macOS-en a tálcaikont csak a fő szálról szabad frissíteni."""
    if sys.platform == "darwin":
        try:
            from PyObjCTools import AppHelper

            AppHelper.callAfter(fn)
            return
        except Exception:
            pass
    fn()


class App:
    def __init__(self):
        self.cfg = config_mod.load()
        self.state = STATE_IDLE
        audio_cfg = self.cfg["audio"]
        self.recorder = Recorder(
            device=self.cfg["audio_device"],
            max_seconds=self.cfg["max_seconds"],
            preroll_seconds=audio_cfg["preroll_seconds"],
            tail_seconds=audio_cfg["tail_seconds"],
            keep_mic_open=audio_cfg["keep_mic_open"],
        )
        self.corrector = Corrector(
            self.cfg["minimax"], language=self.cfg["language"], vocabulary=self.cfg["vocabulary"]
        )
        self.ai_enabled = bool(self.cfg["ai_correction"]) and self.corrector.available
        if self.cfg["ai_correction"] and not self.corrector.available:
            env = self.cfg["minimax"].get("api_key_env") or "MINIMAX_API_KEY"
            print(f"FIGYELEM: az AI-javítás be van kapcsolva, de a(z) {env} környezeti változó üres; kikapcsolva.")

        hw = hardware.detect(self.cfg)
        print(hardware.describe(hw))
        for note in hw.notes:
            print(f"Tipp: {note}")
        self.transcriber = create_transcriber(self.cfg, hw=hw)
        self.backend_label = self.transcriber.candidate.reason
        print(f"Átírás: {self.backend_label}")
        self._transcribe_lock = threading.Lock()
        # Utolsó néhány diktált mondat (opcionális kontextus a Whispernek és a javítónak)
        self.history: list[str] = []
        self.icon = None
        self._quit = threading.Event()

    def _context(self, max_chars: int) -> str:
        ctx = " ".join(self.history)
        return ctx[-max_chars:] if ctx else ""

    def _save_recording(self, audio, text: str):
        """Az utolsó N felvétel mentése (alapból kikapcsolva), hibakereséshez."""
        keep = self.cfg["save_recordings"]
        if not keep:
            return
        try:
            folder = Path(__file__).parent / "recordings"
            folder.mkdir(exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            pcm = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
            with wave.open(str(folder / f"{stamp}.wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SAMPLE_RATE)
                w.writeframes(pcm.tobytes())
            with open(folder / "log.txt", "a", encoding="utf-8") as f:
                p, _r = level_db(audio)
                f.write(f"{stamp}\t{len(audio) / SAMPLE_RATE:.1f}s\tcsúcs {p} dB\t{text}\n")
            for old in sorted(folder.glob("*.wav"))[:-keep]:
                old.unlink(missing_ok=True)
        except Exception as e:
            print(f"(felvétel mentése nem sikerült: {e})")

    # ---------- állapot / tálca ----------

    def set_state(self, state):
        self.state = state
        icon = self.icon
        if icon is not None:
            on_main_thread(lambda: setattr(icon, "icon", make_icon(COLORS[state])))

    # ---------- felvétel vezérlés ----------

    def start_recording(self):
        if self.state != STATE_IDLE:
            return
        try:
            self.recorder.start()
        except Exception as e:
            print(f"Mikrofon hiba: {e}")
            self.notify("Mikrofon hiba - részletek a konzolban")
            return
        self.set_state(STATE_RECORDING)
        if self.cfg["beep"]:
            beep(880)
        print("● Felvétel...")

    def stop_recording(self):
        if self.state != STATE_RECORDING:
            return
        self.set_state(STATE_PROCESSING)
        if self.cfg["beep"]:
            beep(520)
        # a stop() bevárja a hangkártya puffer végét, ezért a sípszó előbb megy
        audio = self.recorder.stop()
        threading.Thread(target=self._process, args=(audio,), daemon=True).start()

    def toggle_recording(self):
        if self.state == STATE_RECORDING:
            self.stop_recording()
        else:
            self.start_recording()

    # ---------- feldolgozás ----------

    def _initial_prompt(self) -> str | None:
        parts = []
        if self.cfg["vocabulary"]:
            parts.append(", ".join(self.cfg["vocabulary"]) + ".")
        if self.cfg["whisper_context"]:
            ctx = self._context(300)
            if ctx:
                parts.append(ctx)
        return " ".join(parts) or None

    def _process(self, audio):
        try:
            dur = len(audio) / SAMPLE_RATE
            if dur < 0.3:
                print("Túl rövid felvétel, eldobva.")
                return
            peak, _rms = level_db(audio)
            if self.cfg["audio"]["normalize"]:
                audio = normalize(audio)
            if peak > -1.0:
                print(f"FIGYELEM: a mikrofon bemenet túlvezérelt (csúcs {peak} dB), vedd lejjebb.")
            else:
                _p, new_rms = level_db(audio)
                if new_rms < -40.0:
                    print(
                        f"FIGYELEM: a felerősítés után is halk a felvétel ({new_rms} dB), "
                        "emeld a mikrofon szintjét a rendszer hangbeállításaiban."
                    )
            t0 = time.time()
            with self._transcribe_lock:
                raw = self.transcriber.transcribe(audio, self.cfg["language"], initial_prompt=self._initial_prompt())
            print(f'Whisper ({dur:.1f}s hang, {time.time() - t0:.2f}s alatt): "{raw}"')
            self._save_recording(audio, raw)
            if not raw:
                return
            text = raw
            if self.ai_enabled:
                t0 = time.time()
                text = self.corrector.correct(raw, context=self._context(600))
                print(f'MiniMax ({time.time() - t0:.2f}s alatt): "{text}"')
            self.history.append(text)
            del self.history[:-5]
            pasted = output_mod.deliver(
                text,
                method=self.cfg["output"]["method"],
                restore_clipboard=self.cfg["output"]["restore_clipboard"],
                append_space=self.cfg["output"].get("append_space", True),
            )
            if not pasted:
                msg = f"A szöveg a vágólapon van, illeszd be: {output_mod.paste_shortcut()}"
                print(msg + " (Waylandhez telepítsd a wtype vagy a ydotool csomagot, lásd README)")
                self.notify(msg)
        except Exception as e:
            print(f"Feldolgozási hiba: {e}")
            self.notify("Hiba történt - részletek a konzolban")
        finally:
            self.set_state(STATE_IDLE)

    def notify(self, msg):
        icon = self.icon
        if icon is not None:
            def show():
                try:
                    icon.notify(msg, "Diktáló")
                except Exception:
                    pass

            on_main_thread(show)

    # ---------- gyorsbillentyű ----------

    def setup_hotkey(self):
        """A gyorsbillentyű figyelése; a visszaadott függvény leállítja (None: nincs figyelés)."""
        hotkey = (self.cfg["hotkey"] or "").strip()
        toggle_cmd = ".venv\\Scripts\\python control.py toggle" if sys.platform == "win32" \
            else str(Path(__file__).resolve().parent / "toggle.sh")
        if not hotkey:
            print(f"Nincs gyorsbillentyű beállítva. Felvétel ki/be: {toggle_cmd}")
            return None
        if output_mod.is_wayland():
            print(
                "Wayland alatt a program nem figyelheti a billentyűzetet. Állíts be egy\n"
                "billentyűparancsot az asztali környezetben (pl. GNOME: Beállítások ->\n"
                "Billentyűzet -> Saját billentyűparancsok), ami ezt futtatja:\n"
                f"    {toggle_cmd}\n"
                "Részletek a README Linux-szakaszában."
            )
            return None
        try:
            stop = hotkeys.register(
                hotkey, self.cfg["mode"], self.start_recording, self.stop_recording, self.toggle_recording
            )
        except Exception as e:
            print(f"A gyorsbillentyű nem állítható be ({e}). Felvétel ki/be: {toggle_cmd}")
            return None
        if sys.platform == "darwin":
            print(
                "macOS: ha a gyorsbillentyű nem reagál, engedélyezd a Terminált a Rendszerbeállítások ->\n"
                "Adatvédelem és biztonság -> Bemenetfigyelés és Kisegítő lehetőségek listáján, majd indítsd újra."
            )
        if self.cfg["mode"] == "hold":
            print(f"Nyomva tartás mód: tartsd nyomva a(z) [{hotkey}] kombinációt diktáláshoz.")
        else:
            print(f"Kapcsoló mód: [{hotkey}] indít, még egyszer [{hotkey}] leállít.")
        return stop

    # ---------- tálcaikon ----------

    def toggle_ai(self, icon=None, item=None):
        self.ai_enabled = not self.ai_enabled
        print(f"AI-javítás: {'BE' if self.ai_enabled else 'KI'}")

    def quit(self):
        self._quit.set()
        icon = self.icon
        if icon is not None:
            on_main_thread(icon.stop)

    def _run_tray(self) -> bool:
        """Tálcaikonos futás. False, ha a rendszeren nem indul tálcaikon (akkor konzolos mód jön)."""
        try:
            import pystray

            menu = pystray.Menu(
                pystray.MenuItem(lambda item: f"Állapot: {STATE_LABELS[self.state]}", None, enabled=False),
                pystray.MenuItem(f"Átírás: {self.backend_label}", None, enabled=False),
                pystray.MenuItem(
                    "AI-javítás (MiniMax, felhős)",
                    self.toggle_ai,
                    checked=lambda item: self.ai_enabled,
                    enabled=self.corrector.available,
                ),
                pystray.MenuItem("Kilépés", lambda icon, item: self.quit()),
            )
            self.icon = pystray.Icon("diktalo", make_icon(COLORS[STATE_IDLE]), "Diktáló (helyi Whisper)", menu)
        except Exception as e:
            print(f"A tálcaikon nem indítható ({e}).")
            self.icon = None
            return False
        print("Kész! A program a tálcán fut.")
        self.icon.run()
        return True

    def run(self):
        stop_hotkey = self.setup_hotkey()
        server = None
        try:
            server = control.ControlServer({
                "toggle": self.toggle_recording,
                "start": self.start_recording,
                "stop": self.stop_recording,
                "quit": self.quit,
            })
        except OSError as e:
            print(f"(a parancssori vezérlés nem indult: {e})")
        mic = device_name(self.cfg["audio_device"])
        print(f"Mikrofon: {mic}")
        if looks_like_webcam(mic):
            print(
                "FIGYELEM: ez kameramikrofonnak tűnik! Az távol vesz, és sokat ront az\n"
                "          átíráson. Állítsd át a rendszer hangbeállításaiban, vagy írd be a\n"
                '          config.json "audio_device" mezőjébe a kívánt mikrofon nevét.'
            )
        self.recorder.open()  # mikrofon megnyitása, hogy a körpuffer feltöltődjön
        try:
            if not (self.cfg.get("tray", True) and self._run_tray()):
                print("Kész! A program ebben az ablakban fut. Kilépés: Ctrl+C.")
                while not self._quit.wait(0.5):
                    pass
        finally:
            self.recorder.close()
            if stop_hotkey:
                stop_hotkey()
            if server:
                server.close()
            close = getattr(self.transcriber, "close", None)
            if close:
                close()
            print("Kilépés.")


if __name__ == "__main__":
    try:
        App().run()
    except KeyboardInterrupt:
        sys.exit(0)
