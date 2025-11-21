#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "=== Creating/using build_env virtual environment ==="
if [ ! -d "build_env" ]; then
  python3 -m venv build_env
fi

source build_env/bin/activate

echo "=== Upgrading pip ==="
pip install -U pip

echo "=== Installing dependencies ==="
pip install -r requirements.txt

echo "=== Building FaceSearch.app with PyInstaller ==="
pyinstaller --noconfirm \
  --name "FaceSearch" \
  --windowed \
  --collect-all insightface \
  --collect-all onnxruntime \
  --collect-submodules onnxruntime \
  --hidden-import onnxruntime \
  desktop_app.py

echo
echo "✅ Build complete! Your app is here:"
echo "   $(pwd)/dist/FaceSearch.app"
echo
echo "You can drag it into /Applications if you like."
