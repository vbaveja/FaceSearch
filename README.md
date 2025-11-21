# FaceSearch (go-PAL ETHOS Innovation Inc)

Offline face search desktop application built using Python, InsightFace, FAISS, and PySide6.

## Features
- Offline, local-only face recognition
- Multi-folder indexing (SQLite + FAISS)
- N-person co-occurrence search
- Thumbnail generation
- Export matching photos
- macOS app bundling + DMG packaging
- Fully private — no cloud processing

## Setup

### Create Virtual Environment
```bash
python3 -m venv build_env
source build_env/bin/activate
pip install -r requirements.txt
```

### Run the App
```bash
python desktop_app.py
```

### Build macOS .app
```bash
rm -rf build dist FaceSearch.spec
pyinstaller --windowed --name "FaceSearch" --icon=FaceSearch.icns desktop_app.py
```

### Build DMG
```bash
dmgbuild -s dmg_settings.py "FaceSearch" FaceSearch.dmg
```

## Data Storage
Indexes stored under:
```
~/FaceSearchData/
```

## License
© 2025 go-PAL ETHOS Innovation Inc
# FaceSearch
# FaceSearch
