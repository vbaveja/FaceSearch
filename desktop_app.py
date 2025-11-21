import os
import sys
import json
import shutil
import sqlite3
import time
import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import faiss
import insightface
from PIL import Image

# Optional HEIC support (recommended). If pillow-heif is not installed, HEIC
# files will be skipped during loading, and you'll get a warning in the UI.
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:
    HEIC_SUPPORTED = False

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QListWidget, QListWidgetItem,
    QLabel, QSpinBox, QMessageBox, QProgressBar, QComboBox,
    QInputDialog
)
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtCore import QSize, Qt

# ================== STORAGE LAYOUT ==================
#   ~/FaceSearchData/
#       indexes.json
#       indexes/
#         <index_id>/
#             faces.db
#             metadata.json
#             index.faiss
#             thumbnails/face_1.jpg ...

BASE_DATA_DIR = os.path.join(os.path.expanduser("~"), "FaceSearchData")
INDEXES_DIR = os.path.join(BASE_DATA_DIR, "indexes")
os.makedirs(INDEXES_DIR, exist_ok=True)

FAISS_INDEX = None
METADATA = []
CURRENT_INDEX_ID = None
CURRENT_INDEX_DIR = None
CURRENT_THUMB_DIR = None

# InsightFace models (global + worker)
_model = None
_worker_model = None


# ================== MODEL / IMAGE HELPERS ==================

def get_model():
    """Global search-time model (single instance in main process)."""
    global _model
    if _model is None:
        _model = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        _model.prepare(ctx_id=0)
    return _model


def is_image_file(path: Path) -> bool:
    ext = path.suffix.lower()
    return ext in [".jpg", ".jpeg", ".png", ".heic"]


def load_image_to_bgr(path_str: str):
    """
    Loads an image via Pillow and returns a BGR numpy array
    (InsightFace expects BGR).
    """
    img = Image.open(path_str).convert("RGB")
    arr = np.array(img)  # RGB
    return arr[:, :, ::-1]


def get_worker_model():
    """Model instance used in worker processes for indexing."""
    global _worker_model
    if _worker_model is None:
        _worker_model = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        _worker_model.prepare(ctx_id=0)
    return _worker_model


# ================== INDEXING HELPERS ==================

def process_image(path_str: str):
    """
    Worker function: run in a subprocess.
    Returns a list of face dicts:
    { "image_path": str, "bbox": [x1,y1,x2,y2], "embedding": list[float] }
    """
    try:
        img_bgr = load_image_to_bgr(path_str)
    except Exception as e:
        print(f"[worker] Failed to load {path_str}: {e}")
        return []

    app = get_worker_model()
    try:
        faces = app.get(img_bgr)
    except Exception as e:
        # Some images may cause internal failures in InsightFace – skip them
        print(f"[worker] InsightFace failed on {path_str}: {e}")
        return []

    results = []
    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        emb = face.normed_embedding.astype("float32")
        results.append({
            "image_path": path_str,
            "bbox": [x1, y1, x2, y2],
            "embedding": emb.tolist()
        })
    return results


def create_db(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            x1 INTEGER,
            y1 INTEGER,
            x2 INTEGER,
            y2 INTEGER,
            thumbnail_path TEXT
        )
    """)
    conn.commit()


def generate_thumbnail(image_path: str, bbox, face_id: int, thumb_dir: str) -> str:
    x1, y1, x2, y2 = bbox
    try:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            crop = img
        else:
            crop = img.crop((x1, y1, x2, y2))

        crop.thumbnail((256, 256))
        os.makedirs(thumb_dir, exist_ok=True)
        thumb_path = os.path.join(thumb_dir, f"face_{face_id}.jpg")
        crop.save(thumb_path, "JPEG", quality=85)
        return thumb_path
    except Exception as e:
        print(f"Failed to create thumbnail for {image_path}: {e}")
        return ""


def get_index_dir(index_id: str) -> str:
    return os.path.join(INDEXES_DIR, index_id)


def get_index_thumb_dir(index_id: str) -> str:
    return os.path.join(get_index_dir(index_id), "thumbnails")


def build_index(photo_root_str: str, index_id: str, progress_cb=None):
    """
    Build an index for all images under `photo_root_str`.
    Uses multiprocessing when run as a normal script,
    and single-process when running as a frozen app.
    """
    photo_root = Path(photo_root_str)
    if not photo_root.exists():
        raise RuntimeError(f"PHOTO_DIR does not exist: {photo_root_str}")

    all_images = [p for p in photo_root.rglob("*") if is_image_file(p)]
    total_files = len(all_images)
    if not all_images:
        raise RuntimeError("No JPG/JPEG/PNG/HEIC images found in selected folder.")

    print(f"Indexing {total_files} image files from {photo_root} ...")

    index_dir = get_index_dir(index_id)
    thumb_dir = get_index_thumb_dir(index_id)
    os.makedirs(index_dir, exist_ok=True)
    os.makedirs(thumb_dir, exist_ok=True)

    db_path = os.path.join(index_dir, "faces.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    create_db(conn)
    cur = conn.cursor()

    for f in os.listdir(thumb_dir):
        try:
            os.remove(os.path.join(thumb_dir, f))
        except Exception:
            pass

    meta_json_path = os.path.join(index_dir, "metadata.json")
    if os.path.exists(meta_json_path):
        os.remove(meta_json_path)
    faiss_path = os.path.join(index_dir, "index.faiss")
    if os.path.exists(faiss_path):
        os.remove(faiss_path)

    embeddings = []
    metadata_local = []
    face_id_counter = 0
    processed_files = 0

    is_frozen = getattr(sys, "frozen", False)

    def handle_face_results(face_results):
        nonlocal face_id_counter
        if not face_results:
            return
        for face in face_results:
            image_path = face["image_path"]
            x1, y1, x2, y2 = face["bbox"]
            emb = np.array(face["embedding"], dtype="float32")

            cur.execute(
                """
                INSERT INTO faces (image_path, x1, y1, x2, y2, thumbnail_path)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (image_path, x1, y1, x2, y2, ""),
            )
            conn.commit()
            face_id = cur.lastrowid

            thumb_path = generate_thumbnail(
                image_path, (x1, y1, x2, y2), face_id, thumb_dir
            )
            cur.execute(
                "UPDATE faces SET thumbnail_path=? WHERE id=?",
                (thumb_path, face_id),
            )
            conn.commit()

            embeddings.append(emb)
            metadata_local.append(
                {
                    "face_id": face_id,
                    "image_path": image_path,
                    "thumbnail_path": thumb_path,
                }
            )
            face_id_counter += 1

    if not is_frozen:
        max_workers = min(4, (os.cpu_count() or 4))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_image, str(p)): p for p in all_images
            }
            for future in as_completed(futures):
                processed_files += 1
                if progress_cb is not None:
                    progress_cb(processed_files, total_files)
                face_results = future.result()
                handle_face_results(face_results)
    else:
        print("Running in frozen mode — single-process indexing.")
        for p in all_images:
            processed_files += 1
            if progress_cb is not None:
                progress_cb(processed_files, total_files)
            face_results = process_image(str(p))
            handle_face_results(face_results)

    print(f"Total faces indexed: {face_id_counter}")
    if not embeddings:
        conn.close()
        raise RuntimeError("No faces found in any images in this folder.")

    emb_array = np.vstack(embeddings).astype("float32")
    print(f"Embedding matrix shape: {emb_array.shape}")

    with open(meta_json_path, "w") as f:
        json.dump(metadata_local, f)

    dim = emb_array.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb_array)
    faiss.write_index(index, faiss_path)

    conn.close()
    print(f"Index saved to {index_dir}")


# ================== INDEX REGISTRY (MULTIPLE FOLDERS) ==================

def load_indexes_meta():
    meta_path = os.path.join(BASE_DATA_DIR, "indexes.json")
    if not os.path.exists(meta_path):
        return {"current_index_id": None, "indexes": []}
    with open(meta_path, "r") as f:
        return json.load(f)


def save_indexes_meta(meta):
    meta_path = os.path.join(BASE_DATA_DIR, "indexes.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def load_index(index_id: str):
    global FAISS_INDEX, METADATA, CURRENT_INDEX_ID, CURRENT_INDEX_DIR, CURRENT_THUMB_DIR

    index_dir = get_index_dir(index_id)
    faiss_path = os.path.join(index_dir, "index.faiss")
    meta_json_path = os.path.join(index_dir, "metadata.json")

    if not (os.path.exists(faiss_path) and os.path.exists(meta_json_path)):
        raise RuntimeError("Index files not found for this ID. Please re-index this folder.")

    FAISS_INDEX = faiss.read_index(faiss_path)
    with open(meta_json_path, "r") as f:
        METADATA = json.load(f)

    CURRENT_INDEX_ID = index_id
    CURRENT_INDEX_DIR = index_dir
    CURRENT_THUMB_DIR = get_index_thumb_dir(index_id)
    print(f"Loaded index {index_id} with {FAISS_INDEX.ntotal} vectors.")


# ================== Qt DESKTOP APP ==================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("go-PAL ETHOS Innovation Inc — Face Search")

        self.matches = []
        self.index_meta = load_indexes_meta()
        self.index_id_for_row = []

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Header
        header_layout = QHBoxLayout()
        app_dir = os.path.dirname(os.path.realpath(__file__))
        logo_path = os.path.join(app_dir, "logo.png")

        self.logo_label = QLabel()
        if os.path.exists(logo_path):
            pix = QPixmap(logo_path)
            if not pix.isNull():
                pix = pix.scaled(48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.logo_label.setPixmap(pix)
        else:
            self.logo_label.setText("🖼️")

        self.title_label = QLabel("go-PAL ETHOS Innovation Inc — Offline Face Search")
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold;")

        header_layout.addWidget(self.logo_label)
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()

        main_layout.addLayout(header_layout)

        # Top bar
        top_bar = QHBoxLayout()

        self.btn_index_folder = QPushButton("Index Folder…")
        self.btn_search_face = QPushButton("Pick Photo to Search…")
        self.btn_search_two = QPushButton("Find Photos with N People…")
        self.btn_export = QPushButton("Export Matches to Folder…")

        self.index_combo = QComboBox()
        self.index_combo.setMinimumWidth(260)

        self.topk_label = QLabel("Top K:")
        self.topk_spin = QSpinBox()
        self.topk_spin.setRange(1, 500)
        self.topk_spin.setValue(50)

        top_bar.addWidget(self.btn_index_folder)
        top_bar.addWidget(self.btn_search_face)
        top_bar.addWidget(self.btn_search_two)
        top_bar.addWidget(self.btn_export)
        top_bar.addWidget(QLabel("Index:"))
        top_bar.addWidget(self.index_combo)
        top_bar.addStretch()
        top_bar.addWidget(self.topk_label)
        top_bar.addWidget(self.topk_spin)

        main_layout.addLayout(top_bar)

        # Gallery
        self.gallery = QListWidget()
        self.gallery.setViewMode(QListWidget.IconMode)
        self.gallery.setIconSize(QSize(128, 128))
        self.gallery.setResizeMode(QListWidget.Adjust)
        self.gallery.setSpacing(10)
        main_layout.addWidget(self.gallery)

        # Bottom bar
        bottom_layout = QVBoxLayout()
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Ready.")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_bar)

        copyright_layout = QHBoxLayout()
        self.copyright_label = QLabel("© 2025 go-PAL ETHOS Innovation Inc")
        copyright_layout.addWidget(self.copyright_label)
        copyright_layout.addStretch()

        bottom_layout.addLayout(status_layout)
        bottom_layout.addLayout(copyright_layout)

        main_layout.addLayout(bottom_layout)

        # Connections
        self.btn_index_folder.clicked.connect(self.index_folder)
        self.btn_search_face.clicked.connect(self.search_by_face)
        self.btn_search_two.clicked.connect(self.search_two_people)
        self.btn_export.clicked.connect(self.export_matches)
        self.index_combo.currentIndexChanged.connect(self.on_index_selected)

        self.populate_index_combo()

    # ---------- Index dropdown helpers ----------

    def populate_index_combo(self):
        self.index_combo.blockSignals(True)
        self.index_combo.clear()
        self.index_id_for_row = []

        indexes = self.index_meta.get("indexes", [])
        for entry in indexes:
            label = f"{Path(entry['folder']).name}  ({entry['folder']})"
            self.index_combo.addItem(label)
            self.index_id_for_row.append(entry["id"])

        current_id = self.index_meta.get("current_index_id")
        if current_id and current_id in self.index_id_for_row:
            row = self.index_id_for_row.index(current_id)
            self.index_combo.setCurrentIndex(row)
            try:
                load_index(current_id)
                self.status_label.setText(
                    f"Loaded index for: {self.current_index_folder_label()}"
                )
            except Exception as e:
                self.status_label.setText(f"Failed to load index: {e}")
        else:
            self.status_label.setText("No index selected.")
        self.index_combo.blockSignals(False)

    def current_index_folder_label(self):
        idx = self.index_combo.currentIndex()
        if idx < 0 or idx >= len(self.index_id_for_row):
            return "None"
        entry = self.index_meta["indexes"][idx]
        return entry["folder"]

    def on_index_selected(self, row: int):
        if row < 0 or row >= len(self.index_id_for_row):
            return
        index_id = self.index_id_for_row[row]
        try:
            load_index(index_id)
            self.index_meta["current_index_id"] = index_id
            save_indexes_meta(self.index_meta)
            self.status_label.setText(
                f"Selected index for: {self.current_index_folder_label()}"
            )
        except Exception as e:
            QMessageBox.warning(self, "Load index", f"Failed to load index:\n{e}")

    # ---------- Status / progress helpers ----------

    def set_busy(self, busy: bool):
        self.btn_index_folder.setEnabled(not busy)
        self.btn_search_face.setEnabled(not busy)
        self.btn_search_two.setEnabled(not busy)
        self.btn_export.setEnabled(not busy)
        self.index_combo.setEnabled(not busy)

    def update_progress(self, done, total):
        percent = int(done * 100 / max(total, 1))
        self.progress_bar.setValue(percent)
        QApplication.processEvents()

    # ---------- Indexing action ----------

    def index_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder to Index (Recursively)",
            ""
        )
        if not folder:
            return

        if not HEIC_SUPPORTED:
            reply = QMessageBox.question(
                self,
                "HEIC support",
                "HEIC (iPhone) support is not installed.\n"
                "You can install later with:\n\n"
                "    pip install pillow-heif\n\n"
                "Continue indexing anyway?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        self.set_busy(True)
        self.status_label.setText(f"Indexing: {folder}")
        self.progress_bar.setValue(0)
        QApplication.processEvents()

        new_id = str(int(time.time()))

        try:
            build_index(folder, new_id, progress_cb=self.update_progress)
            load_index(new_id)

            entry = {
                "id": new_id,
                "folder": folder,
                "created": datetime.datetime.now().isoformat()
            }
            self.index_meta.setdefault("indexes", []).append(entry)
            self.index_meta["current_index_id"] = new_id
            save_indexes_meta(self.index_meta)

            self.populate_index_combo()
            self.status_label.setText(f"Index complete for: {folder}")
            self.progress_bar.setValue(100)
            QMessageBox.information(
                self,
                "Indexing complete",
                f"Index built successfully for:\n{folder}"
            )
        except Exception as e:
            self.status_label.setText("Indexing error.")
            QMessageBox.critical(
                self,
                "Indexing error",
                f"An error occurred while indexing:\n{e}"
            )
            self.progress_bar.setValue(0)
        finally:
            self.set_busy(False)
            QApplication.processEvents()

    # ---------- Query photo helper (pick only, no camera) ----------

    def _choose_query_photo(self, label="person"):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select Photo Containing the {label}",
            "",
            "Images (*.jpg *.jpeg *.png *.heic)"
        )
        if not file_path:
            return None
        try:
            return load_image_to_bgr(file_path)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not read image:\n{e}")
            return None

    # ---------- Single-person search ----------

    def search_by_face(self):
        global FAISS_INDEX, METADATA

        if FAISS_INDEX is None or not METADATA:
            try:
                meta = load_indexes_meta()
                current_id = meta.get("current_index_id")
                if current_id:
                    load_index(current_id)
                    self.index_meta = meta
                    self.populate_index_combo()
                    self.status_label.setText(
                        f"Loaded existing index for: {self.current_index_folder_label()}"
                    )
                else:
                    raise RuntimeError("No index defined.")
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "No index",
                    f"No index is loaded and loading failed:\n{e}\n\n"
                    "Please index a folder first."
                )
                return

        img_bgr = self._choose_query_photo("person")
        if img_bgr is None:
            return

        model = get_model()
        try:
            faces = model.get(img_bgr)
        except Exception as e:
            QMessageBox.warning(
                self,
                "Face detection error",
                f"Failed to process this image for face search.\n\nError:\n{e}"
            )
            return

        if not faces:
            QMessageBox.information(
                self,
                "No face",
                "No face detected in that photo."
            )
            return

        face = faces[0]
        emb = face.normed_embedding.astype("float32")
        emb = np.expand_dims(emb, axis=0)

        top_k = self.topk_spin.value()
        scores, idxs = FAISS_INDEX.search(emb, top_k)

        self.matches = []
        self.gallery.clear()

        for score, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            info = dict(METADATA[i])
            info["score"] = float(score)
            self.matches.append(info)

            thumb_path = info.get("thumbnail_path") or info["image_path"]
            item = QListWidgetItem()
            item.setText(f"{os.path.basename(info['image_path'])}\n{score:.3f}")

            if os.path.exists(thumb_path):
                pix = QPixmap(thumb_path)
            else:
                pix = QPixmap(info["image_path"])

            if not pix.isNull():
                icon = QIcon(pix)
                item.setIcon(icon)

            self.gallery.addItem(item)

        if self.matches:
            self.status_label.setText(
                f"Found {len(self.matches)} matches for that person."
            )
        else:
            self.status_label.setText("No matches found.")
            QMessageBox.information(self, "No matches", "No matches found in index.")

    # ---------- N-person co-occurrence search ----------

    def search_two_people(self):
        global FAISS_INDEX, METADATA

        if FAISS_INDEX is None or not METADATA:
            try:
                meta = load_indexes_meta()
                current_id = meta.get("current_index_id")
                if current_id:
                    load_index(current_id)
                    self.index_meta = meta
                    self.populate_index_combo()
                    self.status_label.setText(
                        f"Loaded existing index for: {self.current_index_folder_label()}"
                    )
                else:
                    raise RuntimeError("No index defined.")
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "No index",
                    f"No index is loaded and loading failed:\n{e}\n\n"
                    "Please index a folder first."
                )
                return

        n, ok = QInputDialog.getInt(
            self,
            "How many people?",
            "Number of distinct people to search for (1–5):",
            2, 1, 5, 1
        )
        if not ok:
            return

        if n == 1:
            self.search_by_face()
            return

        model = get_model()
        embeddings = []

        for i in range(n):
            label = f"person {i+1}"
            img_bgr = self._choose_query_photo(label)
            if img_bgr is None:
                return

            try:
                faces = model.get(img_bgr)
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Face detection error",
                    f"Failed to process the image for {label}.\n\nError:\n{e}"
                )
                return

            if not faces:
                QMessageBox.information(
                    self,
                    "No face",
                    f"No face detected in the photo for {label}."
                )
                return

            emb = faces[0].normed_embedding.astype("float32")
            emb = np.expand_dims(emb, axis=0)
            embeddings.append(emb)

        base_topk = self.topk_spin.value()
        top_k = max(base_topk, 100)

        hits_list = []
        for emb in embeddings:
            scores, idxs = FAISS_INDEX.search(emb, top_k)
            hits = {}
            for score, i in zip(scores[0], idxs[0]):
                if i < 0:
                    continue
                info = METADATA[i]
                path = info["image_path"]
                s = float(score)
                if path not in hits or s > hits[path]:
                    hits[path] = s
            hits_list.append(hits)

        common_paths = set(hits_list[0].keys())
        for h in hits_list[1:]:
            common_paths &= set(h.keys())

        self.matches = []
        self.gallery.clear()

        if not common_paths:
            self.status_label.setText(
                "No photos found where all selected people appear together."
            )
            QMessageBox.information(
                self,
                "No co-occurrence",
                "No photos found in this index that contain all selected people together."
            )
            return

        results = []
        for path in common_paths:
            per_scores = [h[path] for h in hits_list]
            combined = min(per_scores)

            thumb_path = None
            for m in METADATA:
                if m["image_path"] == path and m.get("thumbnail_path"):
                    thumb_path = m["thumbnail_path"]
                    break

            results.append({
                "image_path": path,
                "thumbnail_path": thumb_path,
                "per_scores": per_scores,
                "score": combined,
            })

        results.sort(key=lambda r: r["score"], reverse=True)
        self.matches = results

        for r in results:
            path = r["image_path"]
            thumb = r.get("thumbnail_path") or path
            combined = r["score"]

            item = QListWidgetItem()
            item.setText(f"{os.path.basename(path)}\nmin score: {combined:.3f}")

            if os.path.exists(thumb):
                pix = QPixmap(thumb)
            else:
                pix = QPixmap(path)

            if not pix.isNull():
                icon = QIcon(pix)
                item.setIcon(icon)

            self.gallery.addItem(item)

        self.status_label.setText(
            f"Found {len(results)} photos where all {n} people appear together."
        )

    # ---------- Export action ----------

    def export_matches(self):
        if not self.matches:
            QMessageBox.information(self, "No matches", "Run a search first.")
            return

        dest_dir = QFileDialog.getExistingDirectory(
            self,
            "Select destination folder for export"
        )
        if not dest_dir:
            return

        copied = 0
        seen = set()
        for info in self.matches:
            src = info["image_path"]
            if src in seen or not os.path.exists(src):
                continue
            seen.add(src)
            dst = os.path.join(dest_dir, os.path.basename(src))
            try:
                shutil.copy2(src, dst)
                copied += 1
            except Exception as e:
                print(f"Failed to copy {src}: {e}")

        self.status_label.setText(f"Exported {copied} images to: {dest_dir}")
        QMessageBox.information(
            self,
            "Export complete",
            f"Exported {copied} unique images to:\n{dest_dir}"
        )


def main():
    app_qt = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app_qt.exec())


if __name__ == "__main__":
    main()
