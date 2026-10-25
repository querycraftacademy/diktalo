# Diktáló – helyi diktálás Windowsra, internet nélkül

Tartsd nyomva a **Ctrl+Win** billentyűt, beszélj, engedd el: a szöveg megjelenik ott,
ahol a kurzor áll (Word, böngésző, e-mail, chat, bármi). A beszédfelismerést a
**Whisper large-v3** végzi **a saját gépeden**. A hangod nem megy fel sehová, és az
első modellletöltés után internet sem kell hozzá.

A program felismeri, milyen hardver van a gépedben, és azon futtatja a Whispert:

| Hardver | Háttér | Telepítés | Kipróbálva |
|---|---|---|---|
| **NVIDIA** GeForce / RTX (CUDA 12) | faster-whisper (CTranslate2), float16 | az `install.bat` magától telepíti | nem (a kód és a választás logikája tesztelt) |
| **AMD** Radeon (Vulkan) | whisper.cpp Vulkan-háttérrel, alfolyamatként | `tools\build-whispercpp-vulkan.ps1` vagy saját bináris, lásd lent | nem (a whisper.cpp-illesztés CPU-s binárissal tesztelve) |
| **AMD** Radeon RX 6800 és újabb (ROCm) | faster-whisper + CTranslate2 ROCm-build | kísérleti, kézi, lásd lent | nem |
| **Intel** Arc / Iris Xe | OpenVINO GenAI, GPU | alapból | részben: ugyanez az OpenVINO-út a szerző gépén naponta fut Arc B580-on, de ebben a kiadásban a GPU-s futást nem ismételtük meg |
| **Bármilyen x86 processzor** (Intel, AMD) | OpenVINO GenAI, CPU | alapból | igen, AMD Ryzen 5 3600-on |

Ha a választott háttér nem indul el (hiányzó meghajtó, kevés videómemória, hiányzó
DLL), a program a következőt próbálja, a lista végén mindig a **processzor** áll. A
konzolablak kiírja, mit talált és mit választott.

## Telepítés

1. Telepíts **Python 3.10–3.13**-at (https://www.python.org/downloads/, telepítéskor
   pipáld be: „Add python.exe to PATH”).
2. Töltsd le ezt a repót (zöld **Code** gomb → *Download ZIP*, vagy `git clone`), és
   csomagold ki egy rövid útvonalú mappába, pl. `C:\diktalo`.
3. Futtasd az **`install.bat`**-ot. Ez:
   - létrehozza a `.venv` virtuális környezetet,
   - telepíti az alapcsomagokat (`requirements.txt`, benne az OpenVINO),
   - ha NVIDIA-kártyát talál, a faster-whisper CUDA-s csomagjait is (`requirements-nvidia.txt`),
   - létrehozza a `config.json`-t a `config.example.json` alapján,
   - végül kiírja, milyen hardvert lát, és milyen sorrendben próbálná a háttereket.

Kézi telepítés, ha a `.bat` helyett parancssort használnál:

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install -r requirements-nvidia.txt   (csak NVIDIA-kártyánál)
copy config.example.json config.json
.venv\Scripts\python transcribe_file.py --detect
```

Az első indításkor a program letölti a modellt a `models\` mappába. Large-v3-nál ez
OpenVINO-val kb. 1,5 GB, faster-whisperrel kb. 3 GB, whisper.cpp-vel kb. 1,1 GB. Utána
minden helyben fut.

### NVIDIA GPU

Elég egy friss NVIDIA-meghajtó. A CUDA-könyvtárak (cuBLAS, cuDNN 9) pip-csomagként
érkeznek, külön CUDA Toolkit nem kell. Ha mégis hibát kapsz (`cudnn_ops64_9.dll`,
`cublas64_12.dll`), a program processzorra vált. Ha a hiba az egész programot
leállítja, állítsd a `config.json`-ban: `"backend": "openvino", "device": "cpu"`.

### AMD GPU

AMD-kártyán Windowson a **whisper.cpp Vulkan-háttere** a legjárhatóbb út. Vulkant
szinte minden Radeon tud (RX 400-as sorozattól, az integrált Radeon grafikák is). A
whisper.cpp hivatalos kiadásai (ggml-org/whisper.cpp) Windowsra **nem adnak Vulkanos
binárist**, csak CPU-s, BLAS-os és CUDA-sat. Két lehetőséged van:

1. **Fordítsd le a hivatalos forrásból** a mellékelt szkripttel. Kell hozzá Git,
   CMake, Visual Studio 2022 Build Tools („Desktop development with C++”) és a
   [Vulkan SDK](https://vulkan.lunarg.com/sdk/home#windows):
   ```
   powershell -ExecutionPolicy Bypass -File tools\build-whispercpp-vulkan.ps1
   ```
   A kész `whisper-server.exe` és a DLL-jei a `bin\whispercpp\` mappába kerülnek.
2. **Saját bináris:** ha van már Vulkanos `whisper-server.exe`-d, tedd a DLL-jeivel
   együtt a `bin\whispercpp\` mappába, vagy írd az útját a `config.json`
   `whispercpp.binary` mezőjébe. Csak megbízható forrásból származó binárist használj.

A program a `bin\whispercpp\` mappában lévő `ggml-vulkan.dll`-ből ismeri fel, hogy
a bináris Vulkanos. Ha nincs ilyen, AMD-kártyán is a **processzoron** fut (OpenVINO
CPU, ami AMD processzoron is működik).

**Kísérleti második út (ROCm):** a CTranslate2 4.7-es verziója óta a GitHub-kiadásokon
ROCm-es Windows wheel is van (`rocm-python-wheels-Windows.zip`). Ehhez kell az AMD
ROCm futtatókönyvtára is (`rocm[libraries]` pip-csomag az AMD saját csomagtárolójából).
A buildet ezekre a kártyákra fordítják: gfx1030 (RX 6800/6900), gfx1100–1102 (RX 7000),
gfx1150/1151 (Ryzen AI APU-k), gfx1200/1201 (RX 9000). Ha ez telepítve van, a program
magától felismeri (faster-whisper „cuda” eszközként látja az AMD-kártyát). Ezt az utat
nem próbáltuk ki, a részletek a [CTranslate2-kiadásoknál](https://github.com/OpenNMT/CTranslate2/releases) vannak.

### Intel GPU

Nincs teendő: az OpenVINO alapból települ, és ha lát Intel GPU-t, azt használja (több
kártyánál a diszkrétet, pl. Arc). Az első GPU-s indítás 1–2 percig tarthat
(kernelfordítás), utána a `.ov_cache` miatt pár másodperc. Gyenge integrált Intel
grafikán a large-v3 lassabb lehet, mint a processzor. Ilyenkor állítsd a
`"device": "cpu"` vagy a `"model": "large-v3-turbo"` beállítást.

## Indítás és használat

1. Indítsd el a **`start.bat`**-ot. A konzolablak mutatja a naplót, a tálcán
   megjelenik egy mikrofonos ikon.
2. Kattints oda, ahová írni szeretnél.
3. Tartsd nyomva a **Ctrl+Win** kombinációt: sípszó, felvétel (piros ikon).
4. Beszélj, majd engedd el: átírás (sárga ikon), és a szöveg beillesztődik a kurzorhoz.

Kapcsolós működéshez (egy nyomás indít, még egy leállít) állítsd a `config.json`-ban a
`"mode": "toggle"` és pl. a `"hotkey": "f9"` beállítást.

Automatikus indulás a Windowsszal: tegyél egy parancsikont a `start.bat`-ról a
`shell:startup` mappába (Win+R → `shell:startup`).

### Kipróbálás mikrofon nélkül

```
.venv\Scripts\python transcribe_file.py --detect                 (mit lát a gép, mit választana)
.venv\Scripts\python transcribe_file.py felvetel.wav             (átírás a config szerint)
.venv\Scripts\python transcribe_file.py felvetel.wav --backend openvino --device cpu --model small
.venv\Scripts\python -m unittest discover -s tests -v            (a háttérválasztás tesztjei)
```

## Beállítások (`config.json`)

| Mező | Alapérték | Jelentés |
|---|---|---|
| `hotkey` | `"ctrl+windows"` | Gyorsbillentyű, pl. `"ctrl+windows"`, `"f9"`, `"right shift"` |
| `mode` | `"hold"` | `"hold"`: nyomva tartás alatt vesz fel. `"toggle"`: egy nyomás indít, a következő leállít |
| `language` | `"hu"` | Diktálás nyelve (ISO-kód). Üres string = automatikus felismerés |
| `backend` | `"auto"` | `"auto"`, `"openvino"`, `"faster-whisper"` vagy `"whispercpp"` |
| `device` | `"auto"` | `"auto"`, `"gpu"`, `"cpu"`, vagy a háttér saját eszközneve: `"GPU.1"` (OpenVINO), `"cuda:1"` (faster-whisper), `"gpu:1"` (whisper.cpp) |
| `model` | `"large-v3"` | `"large-v3"`, `"large-v3-turbo"`, `"medium"`, `"small"`, `"base"`, `"tiny"`. Minden háttér a neki való változatot tölti le |
| `threads` | `0` | Processzorszálak CPU-s futásnál (0 = automatikus) |
| `openvino.model` | `""` | Saját OpenVINO-modell (Hugging Face-repó vagy helyi mappa) |
| `faster_whisper.model` | `""` | Saját CTranslate2-modell (név vagy helyi mappa) |
| `faster_whisper.compute_type` | `"auto"` | GPU-n `float16`, CPU-n `int8`. Kevés videómemóriánál: `"int8_float16"` |
| `whispercpp.binary` | `""` | A `whisper-server.exe` útja. Üresen a `bin\whispercpp\` mappát, majd a PATH-t nézi |
| `whispercpp.model` | `""` | ggml-modell neve vagy útja, pl. `"ggml-large-v3.bin"` (alapból `ggml-large-v3-q5_0.bin`) |
| `whispercpp.extra_args` | `[]` | További kapcsolók a whisper-servernek, pl. `["--beam-size", "5"]` |
| `audio_device` | `null` | Mikrofon indexe vagy neve (listázás: `.venv\Scripts\python -m sounddevice`) |
| `audio.preroll_seconds` | `0.6` | A gombnyomás előtti ennyi másodperc is bekerül, így nem vész el az első szótag |
| `audio.tail_seconds` | `0.25` | Elengedés után ennyit vár, így nem vész el az utolsó szótag |
| `audio.normalize` | `true` | Halk felvétel felerősítése (legfeljebb 8x) |
| `output.method` | `"paste"` | `"paste"`: vágólap + Ctrl+V. `"type"`: karakterenkénti gépelés, ha egy program tiltja a beillesztést |
| `output.restore_clipboard` | `false` | Beillesztés után visszaállítja a vágólap korábbi tartalmát |
| `vocabulary` | `[]` | Gyakran elrontott nevek, szakszavak. A Whisper súgásába (és az AI-javítóba) kerülnek |
| `save_recordings` | `0` | Az utolsó N felvétel mentése a `recordings\` mappába hibakereséshez |
| `ai_correction` | `false` | Opcionális **felhős** utójavítás (MiniMax), lásd lent |
| `beep` | `true` | Hangjelzés a felvétel indításakor és leállításakor |

### Opcionális AI-javítás (felhős, alapból kikapcsolva)

A nyers Whisper-átiratot egy nyelvi modell (MiniMax) javíthatja: írásjelek, ékezetek,
nyilvánvaló félrehallások. Ez az egyetlen rész, ami internetet használ, és a szöveged
ilyenkor elmegy a szolgáltatóhoz. Bekapcsolás:

1. állíts be egy környezeti változót: `setx MINIMAX_API_KEY "a-te-kulcsod"` (új ablakban él),
2. a `config.json`-ban: `"ai_correction": true`.

A kulcsot ne írd a `config.json`-ba. A program csak a környezeti változóból olvassa.
Ha a javító a szavak több mint 25%-át átírná, a program eldobja a javítást, és a nyers
átirat megy tovább.

## Mennyire gyors?

A szerző gépén mérve (AMD Ryzen 5 3600, 4 szál, közben egy másik nehéz folyamat is
futott), 25 másodperces magyar diktálás:

| Háttér, modell | Átírás ideje | Eredmény |
|---|---|---|
| OpenVINO CPU, **large-v3** | 27 s | szinte hibátlan (egy szó: „szájtási” a „szállítási” helyett) |
| OpenVINO CPU, small | 5 s | sok hiba, magyarul gyenge |
| faster-whisper CPU, small | 5 s | sok hiba, magyarul gyenge |
| whisper.cpp CPU, small | 6,5 s | sok hiba, magyarul gyenge |

Processzoron a large-v3 tehát nagyjából valós időben ír: 10 másodperc beszéd
kb. 10 másodperc várakozás. Ha ez lassú, a `"model": "large-v3-turbo"` jó kompromisszum.
GPU-n a large-v3 jellemzően 1–2 másodperc alatt végez.

## Ismert korlátok

- **Csak egy diktálóprogram figyeljen ugyanarra a billentyűre.** Ha két példány fut
  (vagy egy másik diktálóeszköz is a Ctrl+Win-t használja), mindkettő felvesz és beilleszt.
- Rendszergazdaként futó ablakba a Windows nem enged billentyűt szimulálni. Ilyenkor a
  `start.bat`-ot is rendszergazdaként kell indítani.
- A `keyboard` csomag Linuxon root jogot kér. A program Windowsra készült, máshol nem teszteltük.
- ARM-os Windows (Snapdragon) nem támogatott: az OpenVINO és a CUDA ott nem fut.
- A NVIDIA- és AMD-utak kódja végig van gondolva, és a választás logikája tesztelt, de
  valódi NVIDIA- vagy AMD-kártyán **nem próbáltuk ki**. Ha hibát találsz, nyiss egy issue-t.
- A Whisper néha a csendre is „hallucinál” egy rövid mondatot (pl. „Köszönöm szépen!”).
  Rövid, néma felvételeknél ez előfordulhat.

## Fájlok

| Fájl | Szerep |
|---|---|
| `app.py` | gyorsbillentyű, felvétel, átírás, beillesztés, tálcaikon |
| `hardware.py` | GPU- és háttérfelismerés |
| `backends\` | a három háttér (OpenVINO, faster-whisper, whisper.cpp) és a választás logikája |
| `recorder.py` | mikrofonfelvétel körpufferrel (16 kHz mono) |
| `output.py` | beillesztés a kurzorhoz |
| `corrector.py` | opcionális MiniMax-javítás |
| `transcribe_file.py` | WAV-fájl átírása és hardver-felismerés parancssorból |
| `tools\build-whispercpp-vulkan.ps1` | whisper.cpp Vulkanos fordítása AMD-kártyához |
| `tests\` | a háttérválasztás egységtesztjei kitalált hardverrel |

## Licenc

MIT, lásd [LICENSE](LICENSE). A modellek a saját licencük alatt állnak (Whisper: MIT,
OpenAI). A felhasznált projektek: [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai),
[faster-whisper](https://github.com/SYSTRAN/faster-whisper),
[whisper.cpp](https://github.com/ggml-org/whisper.cpp).

---

## English summary

**Diktáló** is a local, offline push-to-talk dictation tool for Windows. Hold
**Ctrl+Win**, speak, release, and the text is pasted at the cursor. Speech recognition
runs **on your own machine** with Whisper large-v3 (Hungarian by default, configurable).
Nothing is uploaded, except when you explicitly turn on the optional cloud
post-correction (MiniMax).

**Hardware is detected automatically**, and every backend falls back to the CPU if it
fails to start:

| Hardware | Backend | Tested |
|---|---|---|
| NVIDIA (CUDA 12) | faster-whisper / CTranslate2 | selection logic only, not on real NVIDIA hardware |
| AMD (Vulkan) | whisper.cpp `whisper-server` (Vulkan build) as a subprocess | integration tested with the official CPU binary, not on real AMD hardware |
| AMD RX 6800+ (ROCm) | faster-whisper + CTranslate2 ROCm wheel | experimental, untested |
| Intel Arc / Iris Xe | OpenVINO GenAI (GPU) | same pipeline used daily by the author on an Arc B580, not re-run for this release |
| Any x86 CPU (Intel, AMD) | OpenVINO GenAI (CPU) | yes, AMD Ryzen 5 3600 |

Official whisper.cpp releases do not ship a Windows Vulkan binary.
`tools\build-whispercpp-vulkan.ps1` builds one from the official source (needs Git,
CMake, VS 2022 Build Tools and the Vulkan SDK). You can also drop a trusted
`whisper-server.exe` and its DLLs into `bin\whispercpp\`.

**Install:** Python 3.10–3.13, then run `install.bat`. **Run:** `start.bat`.
**Override detection** in `config.json`: `"backend": "auto" | "openvino" |
"faster-whisper" | "whispercpp"` and `"device": "auto" | "gpu" | "cpu"` (or a native
device name such as `"GPU.1"`, `"cuda:1"`, `"gpu:1"`).
**Test without a microphone:** `.venv\Scripts\python transcribe_file.py --detect` or
`transcribe_file.py some.wav`.

**Known limitations:** only one dictation tool should listen to the same hotkey;
pasting into elevated (admin) windows requires running as admin; CPU-only large-v3 is
roughly real-time (use `large-v3-turbo` on slower machines); Windows on ARM is not
supported.

License: MIT © 2026 Query Craft Academy.
