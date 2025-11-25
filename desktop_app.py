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

# Optional HEIC support (recommended).
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
    QInputDialog, QDialog
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
FACEID_TO_INDEX = {}  # face_id -> index in METADATA / FAISS
CURRENT_INDEX_ID = None
CURRENT_INDEX_DIR = None
CURRENT_THUMB_DIR = None

# InsightFace models (global + worker)
_model = None
_worker_model = None

# Clustering threshold (cosine-like similarity on normalized embeddings)
CLUSTER_SIM_THRESHOLD = 0.6


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


# ================== INDEXING + DB HELPERS ==================

def create_db(conn):
    cur = conn.cursor()

    # Faces table: one row per detected face in an image
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

    # Persons table: one logical person ("Mom", "Vivek", "Bart", etc.)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        )
    """)

    # Face-to-person mapping: which face belongs to which person
    cur.execute("""
        CREATE TABLE IF NOT EXISTS face_person (
            face_id INTEGER NOT NULL,
            person_id INTEGER NOT NULL,
            PRIMARY KEY (face_id, person_id),
            FOREIGN KEY (face_id) REFERENCES faces(id) ON DELETE CASCADE,
            FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
        )
    """)

    # Clusters: auto-grouped "unique people" from embeddings
    cur.execute("""
        CREATE TABLE IF NOT EXISTS clusters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT,
            prototype_face_id INTEGER,
            size INTEGER DEFAULT 0
        )
    """)

    # Mapping from clusters to faces
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cluster_face (
            cluster_id INTEGER NOT NULL,
            face_id INTEGER NOT NULL,
            PRIMARY KEY (cluster_id, face_id),
            FOREIGN KEY (cluster_id) REFERENCES clusters(id) ON DELETE CASCADE,
            FOREIGN KEY (face_id) REFERENCES faces(id) ON DELETE CASCADE
        )
    """)

    conn.commit()


def ensure_people_tables(conn):
    """
    Make sure persons/face_person tables exist in an existing DB.
    Safe to call multiple times.
    """
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS face_person (
            face_id INTEGER NOT NULL,
            person_id INTEGER NOT NULL,
            PRIMARY KEY (face_id, person_id),
            FOREIGN KEY (face_id) REFERENCES faces(id) ON DELETE CASCADE,
            FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
        )
    """)
    conn.commit()


def ensure_cluster_tables(conn):
    """Ensure clusters and cluster_face tables exist."""
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS clusters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT,
            prototype_face_id INTEGER,
            size INTEGER DEFAULT 0
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cluster_face (
            cluster_id INTEGER NOT NULL,
            face_id INTEGER NOT NULL,
            PRIMARY KEY (cluster_id, face_id),
            FOREIGN KEY (cluster_id) REFERENCES clusters(id) ON DELETE CASCADE,
            FOREIGN KEY (face_id) REFERENCES faces(id) ON DELETE CASCADE
        )
    """)
    conn.commit()


def get_current_db_path() -> str:
    """
    Returns the path to the current index's faces.db.
    Raises if no index is currently loaded.
    """
    if CURRENT_INDEX_DIR is None:
        raise RuntimeError("No index is currently loaded.")
    return os.path.join(CURRENT_INDEX_DIR, "faces.db")


def get_current_db_conn():
    """
    Open a SQLite connection for the current index DB and ensure
    people/clusters-related tables exist. Caller must close.
    """
    db_path = get_current_db_path()
    conn = sqlite3.connect(db_path)
    ensure_people_tables(conn)
    ensure_cluster_tables(conn)
    return conn


def create_person(conn, name: str) -> int:
    """
    Create a new person with the given name.
    Returns person_id. If name already exists, returns existing id.
    """
    cur = conn.cursor()
    cur.execute("SELECT id FROM persons WHERE name = ?", (name,))
    row = cur.fetchone()
    if row:
        return row[0]

    cur.execute("INSERT INTO persons (name) VALUES (?)", (name,))
    conn.commit()
    return cur.lastrowid


def get_clusters(conn):
    """
    Return list of clusters: {id, label, size, prototype_face_id}.
    """
    ensure_cluster_tables(conn)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, label, size, prototype_face_id
        FROM clusters
        ORDER BY size DESC, id ASC
    """)
    rows = cur.fetchall()
    clusters = []
    for cid, label, size, proto_face_id in rows:
        clusters.append({
            "id": cid,
            "label": label,
            "size": size,
            "prototype_face_id": proto_face_id,
        })
    return clusters


def get_cluster_faces(conn, cluster_id: int):
    """
    Get faces belonging to a given cluster.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT f.id, f.image_path, f.thumbnail_path
        FROM faces f
        JOIN cluster_face cf ON f.id = cf.face_id
        WHERE cf.cluster_id = ?
        ORDER BY f.id
    """, (cluster_id,))
    faces = []
    for row in cur.fetchall():
        faces.append({
            "face_id": row[0],
            "image_path": row[1],
            "thumbnail_path": row[2],
        })
    return faces


def get_cluster_representative_thumb(conn, cluster_id: int) -> str:
    """
    Return a thumbnail path for the cluster (prototype if available,
    else the first face in the cluster). Returns empty string if none.
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT prototype_face_id FROM clusters WHERE id = ?",
        (cluster_id,)
    )
    row = cur.fetchone()
    if row and row[0]:
        proto_face_id = row[0]
        cur.execute(
            "SELECT thumbnail_path, image_path FROM faces WHERE id = ?",
            (proto_face_id,)
        )
        face_row = cur.fetchone()
        if face_row:
            thumb, img_path = face_row
            return thumb or img_path

    # fallback: first face in cluster_face
    cur.execute("""
        SELECT f.thumbnail_path, f.image_path
        FROM faces f
        JOIN cluster_face cf ON f.id = cf.face_id
        WHERE cf.cluster_id = ?
        ORDER BY f.id
        LIMIT 1
    """, (cluster_id,))
    row = cur.fetchone()
    if not row:
        return ""
    thumb, img_path = row
    return thumb or img_path


def rename_cluster_to_person(conn, cluster_id: int, new_name: str):
    """
    Rename cluster label and map all faces in the cluster
    to a person with the given name (creating the person if needed).
    """
    ensure_people_tables(conn)
    ensure_cluster_tables(conn)
    cur = conn.cursor()

    new_name = new_name.strip()
    if not new_name:
        return

    # Create or find person
    person_id = create_person(conn, new_name)

    # Update cluster label
    cur.execute(
        "UPDATE clusters SET label = ? WHERE id = ?",
        (new_name, cluster_id)
    )

    # Get all face_ids in this cluster
    cur.execute(
        "SELECT face_id FROM cluster_face WHERE cluster_id = ?",
        (cluster_id,)
    )
    rows = cur.fetchall()
    face_ids = [r[0] for r in rows]

    # Link each face to this person
    for fid in face_ids:
        cur.execute("""
            INSERT OR IGNORE INTO face_person (face_id, person_id)
            VALUES (?, ?)
        """, (fid, person_id))

    conn.commit()


def run_clustering(conn, face_ids, emb_array, sim_threshold=CLUSTER_SIM_THRESHOLD):
    """
    Simple greedy clustering:
    - embeddings are assumed L2-normalized or close
    - similarity = dot product
    - if max similarity >= threshold -> assign to existing cluster
      else create new cluster
    Stores results in clusters + cluster_face tables.
    """
    ensure_cluster_tables(conn)
    cur = conn.cursor()

    if emb_array.size == 0 or not face_ids:
        return

    clusters = []  # list of (cluster_id, prototype_vector, count)

    for idx, (face_id, emb) in enumerate(zip(face_ids, emb_array)):
        emb = emb.astype("float32")
        norm = np.linalg.norm(emb) + 1e-6
        emb = emb / norm

        if not clusters:
            # First cluster
            label = f"Unknown Person 1"
            cur.execute(
                "INSERT INTO clusters (label, prototype_face_id, size) VALUES (?, ?, ?)",
                (label, face_id, 1),
            )
            cluster_id = cur.lastrowid
            clusters.append((cluster_id, emb.copy(), 1))
            cur.execute(
                "INSERT OR IGNORE INTO cluster_face (cluster_id, face_id) VALUES (?, ?)",
                (cluster_id, face_id),
            )
            continue

        # Compare to existing prototypes
        protos = np.stack([c[1] for c in clusters])  # (K, D)
        sims = protos @ emb  # (K,)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= sim_threshold:
            # Assign to existing cluster
            cluster_id, proto_vec, count = clusters[best_idx]
            new_count = count + 1
            new_proto = (proto_vec * count + emb) / (new_count + 1e-6)
            new_proto = new_proto / (np.linalg.norm(new_proto) + 1e-6)
            clusters[best_idx] = (cluster_id, new_proto, new_count)

            cur.execute(
                "UPDATE clusters SET size=? WHERE id=?",
                (new_count, cluster_id),
            )
            cur.execute(
                "INSERT OR IGNORE INTO cluster_face (cluster_id, face_id) VALUES (?, ?)",
                (cluster_id, face_id),
            )
        else:
            # Create new cluster
            label = f"Unknown Person {len(clusters) + 1}"
            cur.execute(
                "INSERT INTO clusters (label, prototype_face_id, size) VALUES (?, ?, ?)",
                (label, face_id, 1),
            )
            cluster_id = cur.lastrowid
            clusters.append((cluster_id, emb.copy(), 1))
            cur.execute(
                "INSERT OR IGNORE INTO cluster_face (cluster_id, face_id) VALUES (?, ?)",
                (cluster_id, face_id),
            )

    conn.commit()
    print(f"Clustering complete. Created {len(clusters)} clusters.")


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

    # Clean thumbnails
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

    # Save metadata JSON
    with open(meta_json_path, "w") as f:
        json.dump(metadata_local, f)

    # Build FAISS index
    dim = emb_array.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb_array)
    faiss.write_index(index, faiss_path)

    # Run clustering and store results
    face_ids = [m["face_id"] for m in metadata_local]
    try:
        run_clustering(conn, face_ids, emb_array)
    except Exception as e:
        print(f"Clustering failed: {e}")

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
    global FAISS_INDEX, METADATA, FACEID_TO_INDEX
    global CURRENT_INDEX_ID, CURRENT_INDEX_DIR, CURRENT_THUMB_DIR

    index_dir = get_index_dir(index_id)
    faiss_path = os.path.join(index_dir, "index.faiss")
    meta_json_path = os.path.join(index_dir, "metadata.json")

    if not (os.path.exists(faiss_path) and os.path.exists(meta_json_path)):
        raise RuntimeError("Index files not found for this ID. Please re-index this folder.")

    FAISS_INDEX = faiss.read_index(faiss_path)
    with open(meta_json_path, "r") as f:
        METADATA = json.load(f)

    # Build face_id -> index mapping
    FACEID_TO_INDEX = {}
    for idx, m in enumerate(METADATA):
        fid = m.get("face_id")
        if fid is not None:
            FACEID_TO_INDEX[fid] = idx

    CURRENT_INDEX_ID = index_id
    CURRENT_INDEX_DIR = index_dir
    CURRENT_THUMB_DIR = get_index_thumb_dir(index_id)
    print(f"Loaded index {index_id} with {FAISS_INDEX.ntotal} vectors.")


# ================== CLUSTER EMBEDDING HELPER ==================

def get_cluster_embedding(cluster_id: int):
    """
    Get a representative embedding for a cluster:
      - Prefer prototype_face_id if present.
      - Else first face from cluster_face.
    Returns a (1, D) float32 numpy array, L2-normalized, or None on failure.
    """
    if FAISS_INDEX is None or not METADATA:
        return None

    try:
        conn = get_current_db_conn()
    except Exception as e:
        print(f"DB error in get_cluster_embedding: {e}")
        return None

    try:
        cur = conn.cursor()
        cur.execute("SELECT prototype_face_id FROM clusters WHERE id = ?", (cluster_id,))
        row = cur.fetchone()
        face_id = None
        if row and row[0]:
            face_id = row[0]
        else:
            cur.execute("""
                SELECT face_id FROM cluster_face
                WHERE cluster_id = ?
                ORDER BY face_id
                LIMIT 1
            """, (cluster_id,))
            row2 = cur.fetchone()
            if row2:
                face_id = row2[0]

        if not face_id:
            return None

        # Map face_id -> vector index in FAISS
        idx = FACEID_TO_INDEX.get(face_id)
        if idx is None:
            return None

        emb = FAISS_INDEX.reconstruct(idx).astype("float32")
        norm = np.linalg.norm(emb) + 1e-6
        emb = emb / norm
        return np.expand_dims(emb, axis=0)
    except Exception as e:
        print(f"Error getting cluster embedding: {e}")
        return None
    finally:
        conn.close()


# ================== PEOPLE DIALOG ==================

class PeopleDialog(QDialog):
    """
    Shows clusters (auto-people) for the current index and allows:
      - renaming clusters to real person names
      - showing all photos for that person/cluster
      - searching via external photos (1-person or N-people)
    All in one place.
    """
    def __init__(self, parent=None, show_cluster_callback=None,
                 search_photo_callback=None, search_n_people_callback=None):
        super().__init__(parent)
        self.setWindowTitle("People and Face Search")
        self.resize(650, 540)

        self.show_cluster_callback = show_cluster_callback
        self.search_photo_callback = search_photo_callback
        # expects: search_n_people_callback(cluster_ids: list[int], n: int)
        self.search_n_people_callback = search_n_people_callback

        layout = QVBoxLayout(self)

        self.list = QListWidget()
        self.list.setIconSize(QSize(96, 96))
        self.list.setResizeMode(QListWidget.Adjust)
        self.list.setViewMode(QListWidget.IconMode)
        self.list.setSpacing(10)
        self.list.setSelectionMode(QListWidget.ExtendedSelection)
        layout.addWidget(self.list)

        btn_row1 = QHBoxLayout()
        self.btn_show_photos = QPushButton("Show Photos for Selected")
        self.btn_rename = QPushButton("Rename Selected Person…")
        btn_row1.addWidget(self.btn_show_photos)
        btn_row1.addWidget(self.btn_rename)
        btn_row1.addStretch()
        layout.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        self.btn_search_photo = QPushButton("Search by Local Photo…")
        self.btn_search_n = QPushButton("Find Photos with N People…")
        self.btn_close = QPushButton("Close")
        btn_row2.addWidget(self.btn_search_photo)
        btn_row2.addWidget(self.btn_search_n)
        btn_row2.addStretch()
        btn_row2.addWidget(self.btn_close)
        layout.addLayout(btn_row2)

        self.btn_rename.clicked.connect(self.rename_selected)
        self.btn_close.clicked.connect(self.accept)
        self.btn_show_photos.clicked.connect(self.show_selected_photos)
        self.btn_search_photo.clicked.connect(self.search_by_photo)
        self.btn_search_n.clicked.connect(self.search_n_people)
        self.list.itemDoubleClicked.connect(self.show_item_photos)

        self.refresh()

    def refresh(self):
        """Reload clusters from the current index DB and populate the list."""
        self.list.clear()

        try:
            conn = get_current_db_conn()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open index DB:\n{e}")
            return

        try:
            clusters = get_clusters(conn)
            if not clusters:
                item = QListWidgetItem("No people detected in this index yet.")
                self.list.addItem(item)
                return

            for c in clusters:
                cid = c["id"]
                label = c["label"] or f"Unknown Person {cid}"
                size = c["size"] or 0
                thumb_path = get_cluster_representative_thumb(conn, cid)

                text = f"{label}\n{size} face(s)"
                item = QListWidgetItem(text)

                if thumb_path and os.path.exists(thumb_path):
                    pix = QPixmap(thumb_path)
                    if not pix.isNull():
                        icon = QIcon(pix)
                        item.setIcon(icon)

                # Store cluster_id and label on the item
                item.setData(Qt.UserRole, (cid, label))
                self.list.addItem(item)
        finally:
            conn.close()

    def _rename_cluster(self, cid: int, new_name: str):
        """Apply rename logic in DB."""
        try:
            conn = get_current_db_conn()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open index DB:\n{e}")
            return
        try:
            rename_cluster_to_person(conn, cid, new_name)
        except Exception as e:
            QMessageBox.warning(self, "Rename error", f"Failed to rename:\n{e}")
        finally:
            conn.close()

    def rename_selected(self):
        item = self.list.currentItem()
        if not item:
            QMessageBox.information(self, "No selection", "Select a person/cluster first.")
            return
        self._rename_item_impl(item)

    def _rename_item_impl(self, item):
        data = item.data(Qt.UserRole)
        if not data:
            return
        cid, old_label = data
        old_text = old_label

        new_name, ok = QInputDialog.getText(
            self,
            "Rename Person",
            "Enter a name for this person (e.g., Mom, Dad, Vivek):",
            text=old_text
        )
        if not ok or not new_name.strip():
            return

        self._rename_cluster(cid, new_name.strip())
        self.refresh()

    # ----- Show photos via callback into main window -----

    def show_selected_photos(self):
        items = self.list.selectedItems()
        if not items:
            QMessageBox.information(self, "No selection", "Select a person/cluster first.")
            return
        # Only show photos for the first selected cluster (simpler UX)
        self._show_item_photos_impl(items[0])

    def show_item_photos(self, item):
        self._show_item_photos_impl(item)

    def _show_item_photos_impl(self, item):
        if self.show_cluster_callback is None:
            QMessageBox.warning(
                self,
                "No handler",
                "Internal error: no callback for showing photos."
            )
            return
        data = item.data(Qt.UserRole)
        if not data:
            return
        cid, label = data
        self.show_cluster_callback(cid, label)

    # ----- Unified search actions -----

    def search_by_photo(self):
        """
        Trigger single-person FAISS search using a local photo.
        """
        if self.search_photo_callback is None:
            QMessageBox.warning(
                self,
                "No handler",
                "Internal error: no callback for search by photo."
            )
            return
        self.search_photo_callback()

    def search_n_people(self):
        """
        Trigger N-person co-occurrence search.

        Workflow:
          1. User may multi-select some people tiles (clusters).
          2. Ask for N (default = max(2, #selected)).
          3. Use selected clusters first; if less than N, fill remaining
             slots with example photos from local disk (handled by main window).
        """
        if self.search_n_people_callback is None:
            QMessageBox.warning(
                self,
                "No handler",
                "Internal error: no callback for N-people search."
            )
            return

        selected_items = self.list.selectedItems()
        selected_cluster_ids = []
        for it in selected_items:
            data = it.data(Qt.UserRole)
            if data:
                cid, _ = data
                selected_cluster_ids.append(cid)

        default_n = max(2, len(selected_cluster_ids) or 2)
        n, ok = QInputDialog.getInt(
            self,
            "How many people?",
            "Number of distinct people to search for (1–5):",
            default_n, 1, 5, 1
        )
        if not ok:
            return

        # Call back into main window with cluster_ids and N
        self.search_n_people_callback(selected_cluster_ids, n)


# ================== Qt DESKTOP APP ==================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.matches = []
        self.index_meta = load_indexes_meta()
        self.index_id_for_row = []

        # Window title & icon
        self.setWindowTitle("go-PAL ETHOS Innovation Inc — Face Search")
        app_dir = os.path.dirname(os.path.realpath(__file__))
        icns_path = os.path.join(app_dir, "FaceSearch.icns")
        if os.path.exists(icns_path):
            self.setWindowIcon(QIcon(icns_path))

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Header
        header_layout = QHBoxLayout()
        logo_path = os.path.join(app_dir, "logo.png")

        self.logo_label = QLabel()
        if os.path.exists(logo_path):
            pix = QPixmap(logo_path)
            if not pix.isNull():
                pix = pix.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.logo_label.setPixmap(pix)
        else:
            self.logo_label.setText("🧠")

        self.title_label = QLabel("go-PAL ETHOS Innovation Inc — Offline Face Search")
        self.title_label.setStyleSheet("font-size: 18px; font-weight: 700;")

        header_layout.addWidget(self.logo_label)
        header_layout.addSpacing(8)
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()

        main_layout.addLayout(header_layout)

        # Top bar
        top_bar = QHBoxLayout()

        self.btn_index_folder = QPushButton("Index / Open Folder…")
        self.btn_people = QPushButton("Faces / People…")
        self.btn_export = QPushButton("Export Matches…")

        self.index_combo = QComboBox()
        self.index_combo.setMinimumWidth(260)

        self.topk_label = QLabel("Top K:")
        self.topk_spin = QSpinBox()
        self.topk_spin.setRange(1, 500)
        self.topk_spin.setValue(50)

        top_bar.addWidget(self.btn_index_folder)
        top_bar.addWidget(self.btn_people)
        top_bar.addWidget(self.btn_export)
        top_bar.addSpacing(12)
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
        self.btn_people.clicked.connect(self.show_people)
        self.btn_export.clicked.connect(self.export_matches)
        self.index_combo.currentIndexChanged.connect(self.on_index_selected)

        # Apply a modern dark theme via stylesheets
        self.apply_styles()

        self.populate_index_combo()

    # ---------- Styling ----------

    def apply_styles(self):
        self.setStyleSheet("""
        QMainWindow {
            background-color: #020617;
        }
        QWidget {
            color: #e5e7eb;
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Arial;
            font-size: 13px;
        }
        QLabel {
            color: #e5e7eb;
        }
        QPushButton {
            background-color: #1d4ed8;
            color: #f9fafb;
            border-radius: 6px;
            padding: 6px 12px;
            border: 1px solid #1d4ed8;
        }
        QPushButton:hover {
            background-color: #2563eb;
            border-color: #2563eb;
        }
        QPushButton:pressed {
            background-color: #1e40af;
            border-color: #1e40af;
        }
        QPushButton:disabled {
            background-color: #4b5563;
            border-color: #4b5563;
            color: #9ca3af;
        }
        QComboBox {
            background-color: #020617;
            color: #e5e7eb;
            border-radius: 4px;
            padding: 4px 8px;
            border: 1px solid #374151;
        }
        QComboBox QAbstractItemView {
            background-color: #020617;
            color: #e5e7eb;
            selection-background-color: #1d4ed8;
        }
        QSpinBox {
            background-color: #020617;
            color: #e5e7eb;
            border-radius: 4px;
            padding: 2px 6px;
            border: 1px solid #374151;
        }
        QListWidget {
            background-color: #020617;
            border: 1px solid #111827;
        }
        QListWidget::item {
            border-radius: 6px;
            padding: 4px;
            margin: 4px;
        }
        QListWidget::item:selected {
            background-color: #1d4ed8;
            color: #f9fafb;
        }
        QProgressBar {
            background-color: #020617;
            border: 1px solid #111827;
            border-radius: 4px;
            text-align: center;
            color: #e5e7eb;
        }
        QProgressBar::chunk {
            background-color: #22c55e;
            border-radius: 4px;
        }
        QDialog {
            background-color: #020617;
        }
        """)

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
        enabled = not busy
        self.btn_index_folder.setEnabled(enabled)
        self.btn_people.setEnabled(enabled)
        self.btn_export.setEnabled(enabled)
        self.index_combo.setEnabled(enabled)

    def update_progress(self, done, total):
        percent = int(done * 100 / max(total, 1))
        self.progress_bar.setValue(percent)
        QApplication.processEvents()

    # ---------- Indexing / open-folder action ----------

    def index_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder to Index / Open (Recursively)",
            ""
        )
        if not folder:
            return

        # If this folder is already indexed, offer to reuse or rebuild.
        existing = None
        for entry in self.index_meta.get("indexes", []):
            if entry["folder"] == folder:
                existing = entry
                break

        if existing is not None:
            reply = QMessageBox.question(
                self,
                "Folder already indexed",
                "This folder already has an index.\n\n"
                "Yes: Reuse the existing index (keeps people labels).\n"
                "No:  Rebuild the index (use if photos have changed).",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                try:
                    load_index(existing["id"])
                    self.index_meta["current_index_id"] = existing["id"]
                    save_indexes_meta(self.index_meta)
                    self.populate_index_combo()
                    self.status_label.setText(
                        f"Reused existing index for: {folder}"
                    )
                except Exception as e:
                    QMessageBox.warning(
                        self,
                        "Load index error",
                        f"Failed to load existing index:\n{e}\n\n"
                        "You may try re-indexing."
                    )
                return
            # If No -> fall through to full re-index below.

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

            # If we previously had an index entry for this folder (and user chose rebuild),
            # replace that entry instead of duplicating.
            replaced = False
            for i, e in enumerate(self.index_meta.get("indexes", [])):
                if e["folder"] == folder:
                    self.index_meta["indexes"][i] = entry
                    replaced = True
                    break
            if not replaced:
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

    # ---------- People dialog ----------

    def show_people(self):
        if CURRENT_INDEX_DIR is None:
            QMessageBox.information(
                self,
                "No index",
                "Please index a folder or select an existing index first."
            )
            return
        dlg = PeopleDialog(
            self,
            show_cluster_callback=self.show_cluster_photos,
            search_photo_callback=self.search_by_face,
            search_n_people_callback=self.search_two_people,
        )
        dlg.exec()

    def show_cluster_photos(self, cluster_id: int, label: str):
        """
        Called by PeopleDialog when user chooses a person/cluster.
        Shows all photos containing that cluster's faces in the main gallery.
        """
        try:
            conn = get_current_db_conn()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open index DB:\n{e}")
            return

        try:
            faces = get_cluster_faces(conn, cluster_id)
        finally:
            conn.close()

        if not faces:
            QMessageBox.information(
                self,
                "No faces for cluster",
                f"No faces found for '{label}' in this index."
            )
            return

        # Dedupe by image path; user wants photos, not per-face entries.
        images = {}
        for f in faces:
            path = f["image_path"]
            thumb = f["thumbnail_path"] or path
            if path not in images:
                images[path] = thumb

        self.matches = []
        self.gallery.clear()

        for path, thumb in images.items():
            info = {
                "image_path": path,
                "thumbnail_path": thumb,
                "label": label,
            }
            self.matches.append(info)

            item = QListWidgetItem()
            item.setText(f"{os.path.basename(path)}\n{label}")

            if os.path.exists(thumb):
                pix = QPixmap(thumb)
            else:
                pix = QPixmap(path)

            if not pix.isNull():
                icon = QIcon(pix)
                item.setIcon(icon)

            self.gallery.addItem(item)

        self.status_label.setText(
            f"Found {len(self.matches)} photos containing '{label}'."
        )

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

    # ---------- Single-person search (by example photo) ----------

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

    # ---------- N-person co-occurrence search (clusters + local photos) ----------

    def search_two_people(self, cluster_ids=None, n=None):
        """
        Co-occurrence search for N people.

        cluster_ids: list of cluster IDs selected in PeopleDialog (may be empty).
        n: total number of distinct people to search for (>= 1).

        Strategy:
          1. Turn each cluster_id into a representative embedding.
          2. If len(cluster_ids) < n, ask the user for (n - len(cluster_ids))
             example photos from disk and extract embeddings.
          3. Run multi-person co-occurrence search.
        """
        global FAISS_INDEX, METADATA

        if cluster_ids is None:
            cluster_ids = []

        if n is None:
            n = max(2, len(cluster_ids) or 2)

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

        # 1) Embeddings from clusters
        embeddings = []
        for cid in cluster_ids:
            emb = get_cluster_embedding(cid)
            if emb is not None:
                embeddings.append(emb)

        # 2) If we still need more people, fill with local photos
        model = get_model()
        while len(embeddings) < n:
            label = f"person {len(embeddings)+1}"
            img_bgr = self._choose_query_photo(label)
            if img_bgr is None:
                # User cancelled selection; if we have at least 2, search with those.
                if len(embeddings) < 2:
                    QMessageBox.information(
                        self,
                        "Not enough people",
                        "Need at least 2 people to run a co-occurrence search."
                    )
                    return
                break

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

        # If we ended up with 1 embedding, just do a normal search.
        if len(embeddings) == 1:
            self.search_by_face()
            return

        # 3) Multi-person co-occurrence search
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
            f"Found {len(results)} photos where all {len(embeddings)} people appear together."
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
    app_qt.setStyle("Fusion")
    win = MainWindow()
    win.resize(1000, 700)
    win.show()
    sys.exit(app_qt.exec())


if __name__ == "__main__":
    main()
