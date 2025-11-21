# Developer Notes – FaceSearch App

## Architecture
- UI: PySide6
- Detection/Recognition: InsightFace buffalo_l
- allowed_modules=["detection","recognition"]
- FAISS for embeddings
- SQLite metadata
- Pillow thumbnails

## Index Structure
Each index stored at:
```
~/FaceSearchData/indexes/<id>/
  index.faiss
  metadata.json
  faces.db
  thumbnails/
```

## V1 Decisions
- No webcam capture (PyInstaller instability)
- HEIC optional via pillow-heif
- Multiprocessing only in script mode

## V2 Roadmap
- Settings dialog
- Face clustering (auto-labeling)
- Dark mode UI
- Improved grid/wall view
- Faster thumbnails
- Signed & notarized macOS build
- App Store–ready changes

© 2025 go-PAL ETHOS Innovation Inc
