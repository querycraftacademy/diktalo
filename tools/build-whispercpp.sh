#!/usr/bin/env bash
# whisper.cpp szerver (whisper-server) fordítása macOS-re és Linuxra a hivatalos forrásból.
#
#   macOS:  Metal (Apple GPU) – alapból
#   Linux:  Vulkan (AMD, Intel és NVIDIA kártyán is fut), ha a Vulkan fejlesztői csomag
#           telepítve van; különben csak processzoros build
#
# A kész whisper-server és a könyvtárai a bin/whispercpp mappába kerülnek, ahol a
# diktáló magától megtalálja.
#
# Előfeltételek:
#   macOS:  Xcode Command Line Tools (xcode-select --install), git, cmake (brew install cmake)
#   Linux:  git, cmake, C++ fordító; Vulkanhoz a Vulkan fejlesztői csomag és a glslc:
#           Debian/Ubuntu: sudo apt install git cmake build-essential libvulkan-dev glslc
#           Fedora:        sudo dnf install git cmake gcc-c++ vulkan-loader-devel glslc
#           Arch:          sudo pacman -S git cmake base-devel vulkan-headers vulkan-icd-loader shaderc
#
# Használat:
#   tools/build-whispercpp.sh            (automatikus: Metal / Vulkan / CPU)
#   tools/build-whispercpp.sh --cpu      (csak processzor)
#   tools/build-whispercpp.sh --cuda     (NVIDIA CUDA, Linux; CUDA Toolkit kell)
set -eu

TAG="${WHISPERCPP_TAG:-v1.9.5}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/bin/whispercpp"
WORK="${TMPDIR:-/tmp}/whispercpp-build"
MODE="${1:-auto}"
OS="$(uname -s)"

need() { command -v "$1" >/dev/null 2>&1 || { echo "Hiányzik: $1. $2"; exit 1; }; }
need git "Telepítsd a Gitet."
need cmake "Telepítsd a CMake-et (macOS: brew install cmake; Linux: a csomagkezelőddel)."

ARGS=(-DBUILD_SHARED_LIBS=ON -DWHISPER_BUILD_TESTS=OFF -DWHISPER_SDL2=OFF -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON)
if [ "$OS" = "Darwin" ]; then
    # A könyvtárakat a bináris mellett keresse
    ARGS+=("-DCMAKE_INSTALL_RPATH=@loader_path")
    if [ "$MODE" = "--cpu" ]; then
        ARGS+=(-DGGML_METAL=OFF)
        echo "Fordítás: csak processzor"
    else
        ARGS+=(-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON)
        echo "Fordítás: Metal (Apple GPU)"
    fi
else
    ARGS+=("-DCMAKE_INSTALL_RPATH=\$ORIGIN")
    case "$MODE" in
        --cpu) echo "Fordítás: csak processzor" ;;
        --cuda) ARGS+=(-DGGML_CUDA=ON); echo "Fordítás: CUDA (NVIDIA)" ;;
        *)
            if command -v glslc >/dev/null 2>&1 && { [ -f /usr/include/vulkan/vulkan.h ] || [ -n "${VULKAN_SDK:-}" ]; }; then
                ARGS+=(-DGGML_VULKAN=ON)
                echo "Fordítás: Vulkan (GPU)"
            else
                echo "FIGYELEM: nincs Vulkan fejlesztői csomag vagy glslc, csak processzoros build készül."
                echo "          GPU-hoz telepítsd őket (lásd a szkript elején), és futtasd újra."
            fi
            ;;
    esac
fi

SRC="$WORK/whisper.cpp-$TAG"
if [ ! -d "$SRC" ]; then
    mkdir -p "$WORK"
    echo "whisper.cpp $TAG letöltése..."
    git clone --depth 1 --branch "$TAG" https://github.com/ggml-org/whisper.cpp "$SRC"
fi

BUILD="$SRC/build-diktalo"
rm -rf "$BUILD"
cmake -S "$SRC" -B "$BUILD" "${ARGS[@]}"
echo "Fordítás (néhány perc)..."
JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
cmake --build "$BUILD" --config Release --target whisper-server -j "$JOBS"

EXE="$(find "$BUILD" -type f -name whisper-server -perm -u+x | head -n 1)"
[ -n "$EXE" ] || { echo "A whisper-server nem található a build mappában."; exit 1; }

mkdir -p "$OUT"
cp -f "$EXE" "$OUT/"
# A megosztott könyvtárak (a szimbolikus linkekkel együtt) a bináris mellé
find "$BUILD" \( -name "*.so" -o -name "*.so.*" -o -name "*.dylib" \) -exec cp -Pf {} "$OUT/" \;

echo "Kész: $OUT"
ls "$OUT"
echo "Ellenőrzés: .venv/bin/python transcribe_file.py --detect"
