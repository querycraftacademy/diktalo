#!/usr/bin/env bash
# Diktáló telepítő macOS-re és Linuxra: virtuális környezet + csomagok a géped hardveréhez.
#   ./install.sh          kérdez, mielőtt rendszercsomagot telepítene (sudo)
#   ./install.sh --yes    nem kérdez, mindenre igent mond
set -u
cd "$(dirname "$0")" || exit 1

ASSUME_YES=0
[ "${1:-}" = "--yes" ] && ASSUME_YES=1
OS="$(uname -s)"

fail() {
    echo
    echo "[HIBA] $1"
    echo "A telepítés megszakadt."
    exit 1
}

ask() {  # igen/nem kérdés, alapértelmezés: igen
    [ "$ASSUME_YES" = 1 ] && return 0
    [ -t 0 ] || return 1
    local reply
    read -r -p "$1 [I/n] " reply
    case "$reply" in [nN]*) return 1 ;; *) return 0 ;; esac
}

is_wayland() {
    [ "${XDG_SESSION_TYPE:-}" = "wayland" ] || [ -n "${WAYLAND_DISPLAY:-}" ]
}

# --- 1) Python keresése (3.10–3.13) ---
PY=""
for cand in python3.12 python3.11 python3.13 python3.10 python3; do
    if command -v "$cand" >/dev/null 2>&1 &&
        "$cand" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)' 2>/dev/null; then
        PY="$cand"
        break
    fi
done
if [ -z "$PY" ]; then
    if [ "$OS" = "Darwin" ]; then
        fail "Nem találtam Python 3.10–3.13-at. Telepítsd: https://www.python.org/downloads/macos/ (vagy: brew install python@3.12)"
    fi
    fail "Nem találtam Python 3.10–3.13-at. Telepítsd a disztribúciód csomagkezelőjével (pl. sudo apt install python3)."
fi
echo "Python: $PY ($("$PY" -c 'import sys; print(sys.version.split()[0])'))"

# --- 2) Linux: rendszercsomagok (hangkártya-könyvtár, vágólap, fordító a pynput evdev-függőségéhez) ---
if [ "$OS" = "Linux" ]; then
    need=()
    { ldconfig -p 2>/dev/null || /sbin/ldconfig -p 2>/dev/null; } | grep -q 'libportaudio\.so' || need+=(portaudio)
    "$PY" -c 'import ensurepip, venv' 2>/dev/null || need+=(venv)
    command -v gcc >/dev/null 2>&1 || need+=(compiler)
    "$PY" -c 'import os, sys, sysconfig; sys.exit(0 if os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h")) else 1)' \
        2>/dev/null || need+=(pydev)
    if is_wayland; then
        command -v wl-copy >/dev/null 2>&1 || need+=(wlclip)
    else
        command -v xclip >/dev/null 2>&1 || command -v xsel >/dev/null 2>&1 || need+=(xclip)
    fi

    if [ ${#need[@]} -gt 0 ]; then
        pkgs=()
        if command -v apt-get >/dev/null 2>&1; then
            mgr="apt-get install -y"
            for n in "${need[@]}"; do
                case "$n" in
                    portaudio) pkgs+=(libportaudio2) ;;
                    venv) pkgs+=(python3-venv); [ "$PY" != "python3" ] && pkgs+=("$PY-venv") ;;
                    compiler) pkgs+=(build-essential) ;;
                    pydev) pkgs+=(python3-dev); [ "$PY" != "python3" ] && pkgs+=("$PY-dev") ;;
                    xclip) pkgs+=(xclip) ;;
                    wlclip) pkgs+=(wl-clipboard) ;;
                esac
            done
        elif command -v dnf >/dev/null 2>&1; then
            mgr="dnf install -y"
            for n in "${need[@]}"; do
                case "$n" in
                    portaudio) pkgs+=(portaudio) ;;
                    compiler) pkgs+=(gcc) ;;
                    pydev) pkgs+=(python3-devel) ;;
                    xclip) pkgs+=(xclip) ;;
                    wlclip) pkgs+=(wl-clipboard) ;;
                esac
            done
        elif command -v pacman >/dev/null 2>&1; then
            mgr="pacman -S --needed --noconfirm"
            for n in "${need[@]}"; do
                case "$n" in
                    portaudio) pkgs+=(portaudio) ;;
                    compiler) pkgs+=(gcc) ;;
                    xclip) pkgs+=(xclip) ;;
                    wlclip) pkgs+=(wl-clipboard) ;;
                esac
            done
        elif command -v zypper >/dev/null 2>&1; then
            mgr="zypper install -y"
            for n in "${need[@]}"; do
                case "$n" in
                    portaudio) pkgs+=(libportaudio2) ;;
                    compiler) pkgs+=(gcc) ;;
                    pydev) pkgs+=(python3-devel) ;;
                    xclip) pkgs+=(xclip) ;;
                    wlclip) pkgs+=(wl-clipboard) ;;
                esac
            done
        else
            mgr=""
        fi

        echo
        echo "Hiányzó rendszerkellékek: ${need[*]}"
        if [ -z "$mgr" ]; then
            echo "Nem ismerem a csomagkezelődet. Telepítsd kézzel: PortAudio, python3-venv,"
            echo "gcc, Python-fejlécek, és xclip (X11) vagy wl-clipboard (Wayland)."
        elif [ ${#pkgs[@]} -gt 0 ]; then
            SUDO=""
            [ "$(id -u)" -ne 0 ] && SUDO="sudo"
            cmd="$SUDO $mgr ${pkgs[*]}"
            echo "Ezzel telepíthetők:  $cmd"
            if ask "Telepítsem most?"; then
                if [ "${mgr%% *}" = "apt-get" ]; then
                    $SUDO apt-get update || true
                fi
                $cmd || echo "[FIGYELEM] A rendszercsomagok telepítése nem sikerült, folytatom."
            else
                echo "Kihagyva. Ha a telepítés később elakad, futtasd a fenti parancsot."
            fi
        fi
    fi
fi

# --- 3) Virtuális környezet ---
if [ ! -x ".venv/bin/python" ]; then
    echo "Virtuális környezet létrehozása (.venv)..."
    "$PY" -m venv .venv || fail "A virtuális környezet nem jött létre (Debian/Ubuntu: sudo apt install python3-venv)."
fi
VPY=".venv/bin/python"
"$VPY" -m pip install --upgrade pip >/dev/null

# --- 4) Alapcsomagok (a requirements.txt a rendszeredhez valót választja) ---
echo "Alapcsomagok telepítése..."
"$VPY" -m pip install -r requirements.txt || fail "A csomagok telepítése nem sikerült (lásd fent)."

# --- 5) NVIDIA GPU esetén faster-whisper + CUDA-könyvtárak (Linux) ---
if [ "$OS" = "Linux" ] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo "NVIDIA GPU található: faster-whisper telepítése CUDA-val..."
    "$VPY" -m pip install -r requirements-nvidia.txt ||
        echo "[FIGYELEM] A faster-whisper telepítése nem sikerült, a program processzoron fog futni."
fi

# --- 6) Beállítások, futtatási jog ---
if [ ! -f "config.json" ]; then
    cp config.example.json config.json
    echo "config.json létrehozva a config.example.json alapján."
fi
chmod +x start.sh toggle.sh tools/build-whispercpp.sh 2>/dev/null

# --- 7) Mit lát a gép? ---
echo
"$VPY" transcribe_file.py --detect
echo
echo "Kész. Indítás: ./start.sh"
if [ "$OS" = "Darwin" ]; then
    echo
    echo "macOS: az első indításkor engedélyezd a Terminálnak (vagy az iTermnek) a"
    echo "Rendszerbeállítások -> Adatvédelem és biztonság alatt:"
    echo "  - Mikrofon,"
    echo "  - Bemenetfigyelés (a gyorsbillentyűhöz),"
    echo "  - Kisegítő lehetőségek (a beillesztéshez)."
    echo "Utána a Terminált zárd be és nyisd meg újra."
elif is_wayland; then
    echo
    echo "Wayland: a gyorsbillentyűt az asztali környezetben kell beállítani erre a parancsra:"
    echo "  $(pwd)/toggle.sh"
    echo "A beillesztéshez telepítsd a wtype (KDE, Sway, Hyprland) vagy a ydotool (GNOME) csomagot."
    echo "Részletek a README Linux-szakaszában."
fi
echo "AMD GPU esetén lásd a README \"AMD GPU\" szakaszát."
