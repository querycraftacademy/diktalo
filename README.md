# Diktáló – helyi diktálás internet nélkül (Windows, macOS, Linux)

Tartsd nyomva a gyorsbillentyűt, beszélj, engedd el: a szöveg megjelenik ott, ahol a
kurzor áll (Word, böngésző, e-mail, chat, bármi). A beszédfelismerést a
**Whisper large-v3** végzi **a saját gépeden**. A hangod nem megy fel sehová, és az
első modellletöltés után internet sem kell hozzá.

| Rendszer | Gyorsbillentyű (alapból) | Telepítés | Indítás |
|---|---|---|---|
| **Windows 10/11** | **Ctrl+Win** | `install.bat` | `start.bat` |
| **macOS** (Apple Silicon és Intel) | **Ctrl+Cmd** | `./install.sh` | `./start.sh` |
| **Linux** (X11) | **Ctrl+Super** | `./install.sh` | `./start.sh` |
| **Linux** (Wayland) | az asztali környezetben állítod be | `./install.sh` | `./start.sh` + `toggle.sh` |

A program felismeri, milyen hardver van a gépedben, és azon futtatja a Whispert:

| Hardver | Háttér | Rendszer | Kipróbálva |
|---|---|---|---|
| **NVIDIA** GeForce / RTX (CUDA 12) | faster-whisper (CTranslate2), float16 | Windows, Linux | nem (a választás logikája tesztelt) |
| **AMD** Radeon (Vulkan) | whisper.cpp Vulkannal, alfolyamatként | Windows, Linux | nem (a whisper.cpp-illesztés CPU-s binárissal tesztelve) |
| **AMD** Radeon RX 6800 és újabb (ROCm) | faster-whisper + CTranslate2 ROCm-build | Windows | nem, kísérleti |
| **Intel** Arc / Iris Xe | OpenVINO GenAI, GPU | Windows, Linux | igen, Windowson, Arc B580 (25 mp-es magyar diktálás 3,8 mp alatt) |
| **Apple Silicon** (M1–M4) | mlx-whisper, Metal (Apple GPU) | macOS | nem (valódi Macen nem próbáltuk) |
| **Bármilyen x86 processzor** (Intel, AMD) | OpenVINO GenAI, CPU | Windows, Linux | igen: Windows (Ryzen 5 3600) és Ubuntu 24.04 |
| **Intel-es Mac, ARM-os Linux** processzora | faster-whisper, CPU (int8) | macOS, Linux | nem |

Ha a választott háttér nem indul el (hiányzó meghajtó, kevés videómemória, hiányzó
könyvtár), a program a következőt próbálja, a lista végén mindig a **processzor** áll.
A konzolablak kiírja, mit talált és mit választott.

**Mindhárom rendszerhez kell:** Python 3.10–3.13, kb. 3 GB szabad hely, és az első
indításkor internet a modell letöltéséhez.

Az első lépés mindenhol ugyanaz: töltsd le ezt a repót (zöld **Code** gomb →
*Download ZIP*, majd csomagold ki), vagy:

```
git clone https://github.com/querycraftacademy/diktalo.git
```

---

## Windows

### Telepítés

1. Telepíts **Python 3.10–3.13**-at (https://www.python.org/downloads/, telepítéskor
   pipáld be: „Add python.exe to PATH”).
2. Csomagold ki a repót egy rövid útvonalú mappába, pl. `C:\diktalo`.
3. Futtasd az **`install.bat`**-ot (dupla kattintás). Ez:
   - létrehozza a `.venv` virtuális környezetet,
   - telepíti az alapcsomagokat (`requirements.txt`, benne az OpenVINO),
   - ha NVIDIA-kártyát talál, a faster-whisper CUDA-s csomagjait is (`requirements-nvidia.txt`),
   - létrehozza a `config.json`-t a `config.example.json` alapján,
   - végül kiírja, milyen hardvert lát, és milyen sorrendben próbálná a háttereket.

Kézi telepítés parancssorból:

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install -r requirements-nvidia.txt   (csak NVIDIA-kártyánál)
copy config.example.json config.json
.venv\Scripts\python transcribe_file.py --detect
```

### Indítás

1. Indítsd el a **`start.bat`**-ot. A konzolablak mutatja a naplót, a tálcán
   megjelenik egy mikrofonos ikon.
2. Kattints oda, ahová írni szeretnél.
3. Tartsd nyomva a **Ctrl+Win** kombinációt: sípszó, felvétel (piros ikon).
4. Beszélj, majd engedd el: átírás (sárga ikon), és a szöveg beillesztődik a kurzorhoz.

**Automatikus indulás:** tegyél egy parancsikont a `start.bat`-ról a `shell:startup`
mappába (Win+R → `shell:startup`).

---

## macOS

### Telepítés

1. Telepíts **Python 3.10–3.13**-at: https://www.python.org/downloads/macos/, vagy
   Homebrew-val: `brew install python@3.12`. (A gépen gyárilag lévő `python3` gyakran
   3.9-es, az kevés.)
2. Nyisd meg a **Terminált**, lépj a kicsomagolt mappába, és futtasd a telepítőt:
   ```
   cd ~/Downloads/diktalo
   chmod +x install.sh
   ./install.sh
   ```
   Apple Silicon Macen (M1–M4) ez az **mlx-whisper**-t is telepíti, ami a Mac saját
   GPU-ján fut. Intel-es Macen a processzoron fut (faster-whisper).
3. **Engedélyek.** A Mac csak akkor enged mikrofont és billentyűzetet egy programnak,
   ha megengeded. Rendszerbeállítások → **Adatvédelem és biztonság**, és ott a
   **Terminál** (vagy az iTerm, ha azt használod) legyen bekapcsolva itt:
   - **Mikrofon** (ezt az első felvételkor magától kérdezi),
   - **Bemenetfigyelés** (Input Monitoring): ettől látja a gyorsbillentyűt,
   - **Kisegítő lehetőségek** (Accessibility): ettől tud beilleszteni.

   Az engedélyek után a Terminált zárd be teljesen (Cmd+Q), és nyisd meg újra.

### Indítás

```
cd ~/Downloads/diktalo
./start.sh
```

A menüsorban megjelenik a mikrofonos ikon. Kattints oda, ahová írni szeretnél, tartsd
nyomva a **Ctrl+Cmd** kombinációt, beszélj, engedd el. A szöveg **Cmd+V**-vel
beillesztődik.

Ha a gyorsbillentyű nem reagál, szinte mindig az engedélyek hiányoznak (lásd fent).
Addig a felvételt a `./toggle.sh` paranccsal is indíthatod és leállíthatod egy másik
Terminál-ablakból.

**Automatikus indulás:** Rendszerbeállítások → Általános → **Bejelentkezési elemek**,
és add hozzá a `start.sh`-t (a Terminállal nyílik meg).

---

## Linux

Ubuntu, Debian, Fedora, Arch és openSUSE alatt a telepítő maga felismeri a
csomagkezelőt.

### Telepítés

```
cd ~/Letöltések/diktalo
chmod +x install.sh
./install.sh
```

A telepítő először megnézi, megvannak-e a szükséges **rendszercsomagok**, és ha nem,
kiírja a telepítő parancsot, majd megkérdezi, futtassa-e (`sudo`-val):

| Mire kell | Debian / Ubuntu | Fedora | Arch |
|---|---|---|---|
| mikrofon (PortAudio) | `libportaudio2` | `portaudio` | `portaudio` |
| virtuális környezet | `python3-venv` | (benne van) | (benne van) |
| a billentyűfigyelő csomag fordítása | `build-essential python3-dev` | `gcc python3-devel` | `gcc` |
| vágólap, X11 | `xclip` | `xclip` | `xclip` |
| vágólap, Wayland | `wl-clipboard` | `wl-clipboard` | `wl-clipboard` |

Ha NVIDIA-kártyát talál (`nvidia-smi`), a faster-whisper CUDA-s csomagjait is
telepíti. Külön CUDA Toolkit nem kell, csak az NVIDIA saját meghajtója.

Kérdések nélküli telepítés (pl. szkriptből): `./install.sh --yes`

### Indítás (X11)

```
./start.sh
```

Tartsd nyomva a **Ctrl+Super** kombinációt (a Super a Windows-logós billentyű),
beszélj, engedd el. A szöveg **Ctrl+V**-vel beillesztődik. A tálcaikonhoz GNOME alatt
az *AppIndicator* bővítmény kell (Ubuntun gyárilag megvan). Ha nincs tálca, a program
ugyanúgy működik, csak ikon nélkül.

> **Terminálba diktálsz?** A legtöbb Linux-terminál Ctrl+Shift+V-vel illeszt be, a
> Ctrl+V ott nem működik. Állítsd a `config.json`-ban: `"output": {"method": "type"}`.

### Indítás (Wayland)

Az újabb Ubuntu és Fedora alapból Waylandet használ (ellenőrzés:
`echo $XDG_SESSION_TYPE`). Wayland alatt **egy program biztonsági okból nem figyelheti
a billentyűzetet, és nem nyomhat le billentyűt** más ablakokban. Ezért itt két dolgot
kell beállítani:

1. **Gyorsbillentyű az asztali környezetben.** GNOME: Beállítások → Billentyűzet →
   Billentyűparancsok megtekintése és testreszabása → **Egyéni billentyűparancsok** →
   „+”. Név: *Diktáló*, parancs: a `toggle.sh` teljes útja, pl.
   `/home/te/Letöltések/diktalo/toggle.sh`, billentyű: amit szeretnél (pl. Ctrl+Super+D).
   KDE-ben: Rendszerbeállítások → Billentyűparancsok → Új → Parancs.

   Wayland alatt ezért **kapcsoló üzemmód** van: egy nyomás indítja a felvételt, a
   következő leállítja.
2. **Beillesztés:** telepítsd a **wtype** (KDE, Sway, Hyprland) vagy a **ydotool**
   (GNOME) csomagot. A ydotoolhoz a `ydotoold` szolgáltatásnak is futnia kell
   (`sudo systemctl enable --now ydotool`, disztribúciótól függően). Ha egyik sincs, a
   szöveg a **vágólapra** kerül, és a program szól, hogy Ctrl+V-vel illeszd be.

**Automatikus indulás:** hozz létre egy `~/.config/autostart/diktalo.desktop` fájlt:

```
[Desktop Entry]
Type=Application
Name=Diktáló
Exec=/home/te/Letöltések/diktalo/start.sh
Terminal=false
```

### Vezérlés parancssorból (minden rendszeren)

A futó programot bármikor vezérelheted egy másik ablakból. Ez minden rendszeren megy,
és jól jön, ha a gyorsbillentyű nem reagál:

```
./toggle.sh            felvétel indítása / leállítása   (Windows: .venv\Scripts\python control.py toggle)
./toggle.sh start      csak indítás
./toggle.sh stop       csak leállítás
./toggle.sh quit       kilépés
```

---

## GPU-k részletesen

### NVIDIA (Windows, Linux)

Elég egy friss NVIDIA-meghajtó. A CUDA-könyvtárak (cuBLAS, cuDNN 9) pip-csomagként
érkeznek, a telepítő magától felteszi őket. Ha mégis hibát kapsz (Windowson
`cudnn_ops64_9.dll`, `cublas64_12.dll`, Linuxon `libcudnn…so.9`), a program
processzorra vált. Ha a hiba az egész programot leállítja, állítsd a `config.json`-ban:
`"backend": "openvino", "device": "cpu"`.

### AMD (Windows, Linux)

AMD-kártyán a **whisper.cpp Vulkan-háttere** a legjárhatóbb út. Vulkant szinte minden
Radeon tud (RX 400-as sorozattól, az integrált Radeon grafikák is). A whisper.cpp
hivatalos kiadásai nem adnak Vulkanos binárist, ezért a mellékelt szkript a hivatalos
forrásból fordít egyet. A kész `whisper-server` a könyvtáraival együtt a
`bin/whispercpp/` mappába kerül, ahol a program magától megtalálja.

**Windows:** kell hozzá Git, CMake, Visual Studio 2022 Build Tools („Desktop development
with C++”) és a [Vulkan SDK](https://vulkan.lunarg.com/sdk/home#windows):

```
powershell -ExecutionPolicy Bypass -File tools\build-whispercpp-vulkan.ps1
```

**Linux:** előbb a fordítóeszközök és a Vulkan fejlesztői csomag:

```
sudo apt install git cmake build-essential libvulkan-dev glslc          (Debian / Ubuntu)
sudo dnf install git cmake gcc-c++ vulkan-loader-devel glslc            (Fedora)
sudo pacman -S git cmake base-devel vulkan-headers vulkan-icd-loader shaderc   (Arch)

tools/build-whispercpp.sh
```

A program a mellette lévő `ggml-vulkan` könyvtárból ismeri fel, hogy a bináris
Vulkanos. Ha nincs ilyen, AMD-kártyán is a **processzoron** fut.

**Saját bináris:** ha van már Vulkanos `whisper-server`-ed, tedd a könyvtáraival együtt
a `bin/whispercpp/` mappába, vagy írd az útját a `config.json` `whispercpp.binary`
mezőjébe. Csak megbízható forrásból származó binárist használj.

**Kísérleti második út Windowson (ROCm):** a CTranslate2 4.7-es verziója óta a
GitHub-kiadásokon ROCm-es Windows wheel is van (`rocm-python-wheels-Windows.zip`). Ehhez
kell az AMD ROCm futtatókönyvtára is (`rocm[libraries]` pip-csomag az AMD saját
csomagtárolójából). A buildet ezekre a kártyákra fordítják: gfx1030 (RX 6800/6900),
gfx1100–1102 (RX 7000), gfx1150/1151 (Ryzen AI APU-k), gfx1200/1201 (RX 9000). Ha ez
telepítve van, a program magától felismeri. Ezt az utat nem próbáltuk ki, a részletek a
[CTranslate2-kiadásoknál](https://github.com/OpenNMT/CTranslate2/releases) vannak.

### Intel (Windows, Linux)

Windowson nincs teendő: az OpenVINO alapból települ, és ha lát Intel GPU-t, azt
használja (több kártyánál a diszkrétet, pl. Arc). Linuxon az Intel GPU-hoz a
*compute runtime* is kell (Ubuntu: `sudo apt install intel-opencl-icd`), nélküle az
OpenVINO csak a processzort látja. Az első GPU-s indítás 1–2 percig tarthat
(kernelfordítás), utána a `.ov_cache` miatt pár másodperc. Gyenge integrált Intel
grafikán a large-v3 lassabb lehet, mint a processzor. Ilyenkor állítsd a
`"device": "cpu"` vagy a `"model": "large-v3-turbo"` beállítást.

### Apple Silicon (macOS)

Nincs teendő: az `install.sh` telepíti az mlx-whispert, és a program a Mac GPU-ját
használja. Az első indításkor letölti a modellt (large-v3: kb. 3 GB). Lassabb, régebbi
gépen (8 GB memória) a `"model": "large-v3-turbo"` jobb választás.

Második lehetőség a whisper.cpp Metallal: `tools/build-whispercpp.sh` (kell hozzá
`xcode-select --install` és `brew install cmake`). A Homebrew-féle `whisper-cpp`
csomagban nincs `whisper-server`, ezért az nem elég.

---

## Kipróbálás mikrofon nélkül

```
python transcribe_file.py --detect                 (mit lát a gép, mit választana)
python transcribe_file.py felvetel.wav             (átírás a config szerint)
python transcribe_file.py felvetel.wav --backend openvino --device cpu --model small
python -m unittest discover -s tests -v            (a háttérválasztás tesztjei)
```

A `python` helyére Windowson `.venv\Scripts\python`, macOS-en és Linuxon
`.venv/bin/python` kell.

## Beállítások (`config.json`)

| Mező | Alapérték | Jelentés |
|---|---|---|
| `hotkey` | `"ctrl+windows"` | Gyorsbillentyű, pl. `"ctrl+windows"`, `"f9"`, `"right shift"`. A `windows` Macen a Command, Linuxon a Super billentyű. Üres (`""`): nincs gyorsbillentyű, csak a `toggle.sh` |
| `mode` | `"hold"` | `"hold"`: nyomva tartás alatt vesz fel. `"toggle"`: egy nyomás indít, a következő leállít |
| `language` | `"hu"` | Diktálás nyelve (ISO-kód). Üres string = automatikus felismerés |
| `backend` | `"auto"` | `"auto"`, `"openvino"`, `"faster-whisper"`, `"whispercpp"` vagy `"mlx"` |
| `device` | `"auto"` | `"auto"`, `"gpu"`, `"cpu"`, vagy a háttér saját eszközneve: `"GPU.1"` (OpenVINO), `"cuda:1"` (faster-whisper), `"gpu:1"` (whisper.cpp) |
| `model` | `"large-v3"` | `"large-v3"`, `"large-v3-turbo"`, `"medium"`, `"small"`, `"base"`, `"tiny"`. Minden háttér a neki való változatot tölti le |
| `threads` | `0` | Processzorszálak CPU-s futásnál (0 = automatikus) |
| `openvino.model` | `""` | Saját OpenVINO-modell (Hugging Face-repó vagy helyi mappa) |
| `faster_whisper.model` | `""` | Saját CTranslate2-modell (név vagy helyi mappa) |
| `faster_whisper.compute_type` | `"auto"` | GPU-n `float16`, CPU-n `int8`. Kevés videómemóriánál: `"int8_float16"` |
| `whispercpp.binary` | `""` | A `whisper-server` útja. Üresen a `bin/whispercpp/` mappát, majd a PATH-t nézi |
| `whispercpp.model` | `""` | ggml-modell neve vagy útja, pl. `"ggml-large-v3.bin"` (alapból `ggml-large-v3-q5_0.bin`) |
| `whispercpp.extra_args` | `[]` | További kapcsolók a whisper-servernek, pl. `["--beam-size", "5"]` |
| `mlx.model` | `""` | Saját MLX-modell (Hugging Face-repó vagy helyi mappa), csak Macen |
| `audio_device` | `null` | Mikrofon indexe vagy neve (listázás: `python -m sounddevice` a `.venv`-ből) |
| `audio.preroll_seconds` | `0.6` | A gombnyomás előtti ennyi másodperc is bekerül, így nem vész el az első szótag |
| `audio.tail_seconds` | `0.25` | Elengedés után ennyit vár, így nem vész el az utolsó szótag |
| `audio.normalize` | `true` | Halk felvétel felerősítése (legfeljebb 8x) |
| `output.method` | `"paste"` | `"paste"`: vágólap + Ctrl+V (Macen Cmd+V). `"type"`: karakterenkénti gépelés, ha egy program tiltja a beillesztést |
| `output.restore_clipboard` | `false` | Beillesztés után visszaállítja a vágólap korábbi tartalmát |
| `vocabulary` | `[]` | Gyakran elrontott nevek, szakszavak. A Whisper súgásába (és az AI-javítóba) kerülnek |
| `save_recordings` | `0` | Az utolsó N felvétel mentése a `recordings/` mappába hibakereséshez |
| `ai_correction` | `false` | Opcionális **felhős** utójavítás (MiniMax), lásd lent |
| `beep` | `true` | Hangjelzés a felvétel indításakor és leállításakor |
| `tray` | `true` | Tálcaikon (Macen menüsor-ikon). `false`: a program csak a konzolablakban fut |

### Opcionális AI-javítás (felhős, alapból kikapcsolva)

A nyers Whisper-átiratot egy nyelvi modell (MiniMax) javíthatja: írásjelek, ékezetek,
nyilvánvaló félrehallások. Ez az egyetlen rész, ami internetet használ, és a szöveged
ilyenkor elmegy a szolgáltatóhoz. Bekapcsolás:

1. állíts be egy környezeti változót:
   - Windows: `setx MINIMAX_API_KEY "a-te-kulcsod"` (új ablakban él),
   - macOS / Linux: `export MINIMAX_API_KEY="a-te-kulcsod"` a `~/.zshrc` vagy
     `~/.bashrc` fájlba,
2. a `config.json`-ban: `"ai_correction": true`.

A kulcsot ne írd a `config.json`-ba. A program csak a környezeti változóból olvassa.
Ha a javító a szavak több mint 25%-át átírná, a program eldobja a javítást, és a nyers
átirat megy tovább.

## Mennyire gyors?

A szerző gépén mérve (AMD Ryzen 5 3600, 4 szál, közben egy másik nehéz folyamat is
futott), 25 másodperces magyar diktálás:

| Háttér, modell | Átírás ideje | Eredmény |
|---|---|---|
| OpenVINO, Intel Arc B580 GPU, **large-v3** | 3,8 s | szinte hibátlan |
| OpenVINO CPU, **large-v3** | 27 s | szinte hibátlan (egy szó: „szájtási” a „szállítási” helyett) |
| OpenVINO CPU, small | 5 s | sok hiba, magyarul gyenge |
| faster-whisper CPU, small | 5 s | sok hiba, magyarul gyenge |
| whisper.cpp CPU, small | 6,5 s | sok hiba, magyarul gyenge |

Processzoron a large-v3 tehát nagyjából valós időben ír: 10 másodperc beszéd
kb. 10 másodperc várakozás. Ha ez lassú, a `"model": "large-v3-turbo"` jó kompromisszum
(Linuxon, processzoron mérve egy 12 másodperces felvétel 8,4 s alatt lett kész).
GPU-n a large-v3 jellemzően 1–4 másodperc alatt végez.

## Ismert korlátok

- **Csak egy diktálóprogram figyeljen ugyanarra a billentyűre.** Ha két példány fut
  (vagy egy másik diktálóeszköz is ugyanazt a kombinációt használja), mindkettő felvesz
  és beilleszt.
- Windowson rendszergazdaként futó ablakba a rendszer nem enged billentyűt szimulálni.
  Ilyenkor a `start.bat`-ot is rendszergazdaként kell indítani.
- **macOS-en** a program valódi Macen még nincs kipróbálva. A kód és a választási
  logika tesztelt, de ha valami nem működik, nyiss egy issue-t.
- **Linux Wayland** alatt nincs nyomva tartós mód, csak kapcsoló (lásd fent). A
  Wayland-utat nem próbáltuk ki valódi asztalon. Az X11-utat Ubuntu 24.04-en, virtuális
  kijelzőn teszteltük.
- ARM-os Windows (Snapdragon) nem támogatott: az OpenVINO és a CUDA ott nem fut.
- A NVIDIA- és AMD-utak kódja végig van gondolva, és a választás logikája tesztelt, de
  valódi NVIDIA- vagy AMD-kártyán **nem próbáltuk ki**. Ha hibát találsz, nyiss egy issue-t.
- A Whisper néha a csendre is „hallucinál” egy rövid mondatot (pl. „Köszönöm szépen!”).
  Rövid, néma felvételeknél ez előfordulhat.

## Fájlok

| Fájl | Szerep |
|---|---|
| `app.py` | felvétel, átírás, beillesztés, tálcaikon |
| `hotkeys.py` | gyorsbillentyű (Windows: keyboard, macOS/Linux: pynput) |
| `control.py` | a futó program vezérlése parancssorból (`toggle.sh`) |
| `hardware.py` | GPU- és háttérfelismerés |
| `backends/` | a négy háttér (OpenVINO, faster-whisper, whisper.cpp, mlx) és a választás logikája |
| `recorder.py` | mikrofonfelvétel körpufferrel (16 kHz mono) |
| `output.py` | beillesztés a kurzorhoz (Windows, macOS, Linux X11 és Wayland) |
| `corrector.py` | opcionális MiniMax-javítás |
| `transcribe_file.py` | WAV-fájl átírása és hardver-felismerés parancssorból |
| `install.bat`, `start.bat` | telepítés és indítás Windowson |
| `install.sh`, `start.sh`, `toggle.sh` | telepítés, indítás és kapcsoló macOS-en és Linuxon |
| `tools/build-whispercpp-vulkan.ps1` | whisper.cpp Vulkanos fordítása Windowsra (AMD-kártyához) |
| `tools/build-whispercpp.sh` | whisper.cpp fordítása Linuxra (Vulkan) és macOS-re (Metal) |
| `tests/` | a háttérválasztás, a gyorsbillentyű és a vezérlés egységtesztjei |

## Licenc

MIT, lásd [LICENSE](LICENSE). A modellek a saját licencük alatt állnak (Whisper: MIT,
OpenAI). A felhasznált projektek: [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai),
[faster-whisper](https://github.com/SYSTRAN/faster-whisper),
[whisper.cpp](https://github.com/ggml-org/whisper.cpp),
[mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper),
[pynput](https://github.com/moses-palmer/pynput).

---

## English summary

**Diktáló** is a local, offline push-to-talk dictation tool for **Windows, macOS and
Linux**. Hold the hotkey, speak, release, and the text is pasted at the cursor. Speech
recognition runs **on your own machine** with Whisper large-v3 (Hungarian by default,
configurable). Nothing is uploaded, except when you explicitly turn on the optional
cloud post-correction (MiniMax).

| OS | Default hotkey | Install | Run |
|---|---|---|---|
| Windows 10/11 | Ctrl+Win | `install.bat` | `start.bat` |
| macOS | Ctrl+Cmd | `./install.sh` | `./start.sh` |
| Linux (X11) | Ctrl+Super | `./install.sh` | `./start.sh` |
| Linux (Wayland) | bind `toggle.sh` to a shortcut in your desktop settings | `./install.sh` | `./start.sh` |

**Hardware is detected automatically**, and every backend falls back to the CPU if it
fails to start:

| Hardware | Backend | Tested |
|---|---|---|
| NVIDIA (CUDA 12), Windows/Linux | faster-whisper / CTranslate2 | selection logic only |
| AMD (Vulkan), Windows/Linux | whisper.cpp `whisper-server` (Vulkan build) | integration tested with a CPU build |
| Intel Arc / Iris Xe, Windows/Linux | OpenVINO GenAI (GPU) | yes, Arc B580 on Windows |
| Apple Silicon | mlx-whisper (Metal) | not on a real Mac yet |
| Any x86 CPU, Windows/Linux | OpenVINO GenAI (CPU) | yes, Windows and Ubuntu 24.04 |
| Intel Mac / ARM Linux CPU | faster-whisper (int8) | no |

**macOS:** grant the Terminal *Microphone*, *Input Monitoring* and *Accessibility*
permissions (System Settings → Privacy & Security), then restart the Terminal.
**Linux:** `install.sh` offers to install the needed system packages (PortAudio, venv,
a C compiler for evdev, xclip / wl-clipboard). On **Wayland**, apps cannot read global
keys or synthesize key presses: bind `toggle.sh` to a custom shortcut in your desktop
environment (toggle mode), and install `wtype` or `ydotool` for pasting; without them
the text is left on the clipboard. `toggle.sh start|stop|quit` controls a running
instance on every OS (Windows: `.venv\Scripts\python control.py toggle`).

**GPU builds of whisper.cpp:** `tools\build-whispercpp-vulkan.ps1` (Windows) and
`tools/build-whispercpp.sh` (Linux Vulkan, macOS Metal) build `whisper-server` from the
official source into `bin/whispercpp/`.

**Override detection** in `config.json`: `"backend": "auto" | "openvino" |
"faster-whisper" | "whispercpp" | "mlx"` and `"device": "auto" | "gpu" | "cpu"`.
**Test without a microphone:** `transcribe_file.py --detect` or `transcribe_file.py some.wav`.

License: MIT © 2026 Query Craft Academy.
