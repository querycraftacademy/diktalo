<#
.SYNOPSIS
  whisper.cpp fordítása Vulkan-háttérrel (AMD GPU-hoz, de Intel és NVIDIA kártyán is fut).

.DESCRIPTION
  A ggml-org/whisper.cpp hivatalos kiadásai Windowsra CPU-s, BLAS-os és CUDA-s
  binárist adnak, Vulkanosat nem. Ez a szkript a hivatalos forrásból fordít
  egyet, és a whisper-server.exe-t a DLL-jeivel a bin\whispercpp mappába teszi,
  ahol a diktáló magától megtalálja.

  Előfeltételek (egyszer kell telepíteni):
    - Git                         https://git-scm.com
    - CMake 3.21+                 https://cmake.org  (vagy: pip install cmake)
    - Visual Studio 2022 Build Tools, "Desktop development with C++" munkaterhelés
    - Vulkan SDK (LunarG)         https://vulkan.lunarg.com/sdk/home#windows
    - friss GPU-meghajtó (AMD Adrenalin / Intel Arc / NVIDIA)

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools\build-whispercpp-vulkan.ps1
#>
param(
    [string]$Tag = "v1.9.5",
    [string]$WorkDir = (Join-Path $env:TEMP "whispercpp-vulkan-build"),
    [string]$OutDir = (Join-Path (Split-Path -Parent $PSScriptRoot) "bin\whispercpp")
)
$ErrorActionPreference = "Stop"

function Need($cmd, $hint) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        throw "Hiányzik: $cmd. $hint"
    }
}
Need git "Telepítsd a Gitet: https://git-scm.com"
Need cmake "Telepítsd a CMake-et: https://cmake.org (vagy: pip install cmake)"
if (-not $env:VULKAN_SDK) {
    throw "A VULKAN_SDK környezeti változó üres. Telepítsd a Vulkan SDK-t (https://vulkan.lunarg.com), majd nyiss új ablakot."
}

$src = Join-Path $WorkDir "whisper.cpp-$Tag"
if (-not (Test-Path $src)) {
    New-Item -ItemType Directory -Force $WorkDir | Out-Null
    Write-Host "whisper.cpp $Tag letöltése..."
    git clone --depth 1 --branch $Tag https://github.com/ggml-org/whisper.cpp $src
    if ($LASTEXITCODE -ne 0) { throw "git clone sikertelen" }
}

$build = Join-Path $src "build-vulkan"
Write-Host "Konfigurálás (GGML_VULKAN=ON)..."
cmake -S $src -B $build -DGGML_VULKAN=ON -DBUILD_SHARED_LIBS=ON -DWHISPER_BUILD_TESTS=OFF -DWHISPER_SDL2=OFF -DCMAKE_BUILD_TYPE=Release
if ($LASTEXITCODE -ne 0) { throw "cmake konfigurálás sikertelen" }

Write-Host "Fordítás (néhány perc)..."
cmake --build $build --config Release --target whisper-server --parallel
if ($LASTEXITCODE -ne 0) { throw "fordítás sikertelen" }

$exe = Get-ChildItem -Path $build -Recurse -Filter "whisper-server.exe" | Select-Object -First 1
if (-not $exe) { throw "A whisper-server.exe nem található a build mappában." }

New-Item -ItemType Directory -Force $OutDir | Out-Null
Copy-Item $exe.FullName $OutDir -Force
Get-ChildItem -Path $build -Recurse -Filter "*.dll" | ForEach-Object { Copy-Item $_.FullName $OutDir -Force }

if (-not (Get-ChildItem $OutDir -Filter "ggml-vulkan*.dll")) {
    Write-Warning "Nem találtam ggml-vulkan.dll-t: lehet, hogy a Vulkan statikusan fordult be. Ilyenkor a configban állítsd: backend=whispercpp, device=gpu."
}
Write-Host "Kész: $OutDir"
Write-Host "Ellenőrzés: .venv\Scripts\python transcribe_file.py --detect"
