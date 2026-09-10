"""
face_store.py — shared face-encoding database used by both the Register
and Recognize pages of main.py.

Design notes (why this exists / why matching doesn't get slower as the
database grows):

  * Everything is dlib 128-d encodings from `face_recognition`, one file
    (encodings.pkl) instead of the old two-file Eigenface DB. Eigenfaces
    needs a full PCA re-train on every launch and gets less reliable as
    pose/lighting vary; encoding distance does not, and new people can be
    added without retraining anything.

  * All encodings are kept as ONE stacked numpy matrix (`_matrix`) rather
    than compared person-by-person in a Python loop. `match()` is a single
    vectorized `np.linalg.norm` call over the whole matrix, which is what
    actually keeps per-scan cost low as the roster grows — the old
    per-person Python loop in main.py re-did this work with an interpreted
    loop on every frame.

  * The matrix is rebuilt in memory on every add/remove (cheap — it's a
    handful of numpy concatenations even with hundreds of people) and the
    file is written atomically. Recognition never re-reads the pickle from
    disk, so registering someone doesn't stall anyone mid-scan.
"""

from __future__ import annotations

import os
import pickle
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "encodings.pkl")


@dataclass
class Person:
    id: str
    name: str
    encodings: list  # list[np.ndarray shape (128,)]
    created_at: float = field(default_factory=time.time)

    @property
    def sample_count(self) -> int:
        return len(self.encodings)


class FaceStore:
    """Thread-safe face-encoding database with an O(1)-append match cache."""

    def __init__(self, path: str = DEFAULT_DB_PATH):
        self.path = path
        self._lock = threading.RLock()
        self.people: dict[str, Person] = {}
        self._matrix: Optional[np.ndarray] = None   # (total_samples, 128)
        self._owners: list[str] = []                # owner id of each matrix row
        self._load()

    # ---------------------------------------------------------- persistence
    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "rb") as f:
                    raw = pickle.load(f)
                for pid, rec in raw.items():
                    self.people[pid] = Person(
                        id=pid,
                        name=rec.get("name", pid),
                        encodings=[np.asarray(e, dtype=np.float64) for e in rec["encodings"]],
                        created_at=rec.get("created_at", time.time()),
                    )
            except (pickle.PickleError, EOFError, KeyError, OSError):
                # Corrupt DB file: start empty rather than crash the whole app.
                self.people = {}
        self._rebuild_matrix()

    def _save(self) -> None:
        raw = {
            p.id: {"name": p.name, "encodings": p.encodings, "created_at": p.created_at}
            for p in self.people.values()
        }
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(raw, f)
        os.replace(tmp_path, self.path)  # atomic on POSIX and Windows

    def _rebuild_matrix(self) -> None:
        rows, owners = [], []
        for p in self.people.values():
            for enc in p.encodings:
                rows.append(enc)
                owners.append(p.id)
        self._matrix = np.stack(rows) if rows else None
        self._owners = owners

    # ------------------------------------------------------------ mutation
    def add_person(self, pid: str, name: str, encodings: list) -> None:
        """Add samples to an existing person, or create a new one."""
        with self._lock:
            if pid in self.people:
                self.people[pid].encodings.extend(encodings)
                self.people[pid].name = name or self.people[pid].name
            else:
                self.people[pid] = Person(id=pid, name=name or pid, encodings=list(encodings))
            self._rebuild_matrix()
            self._save()

    def replace_person(self, pid: str, name: str, encodings: list) -> None:
        """Overwrite all samples for a person (used for re-registration)."""
        with self._lock:
            self.people[pid] = Person(id=pid, name=name or pid, encodings=list(encodings))
            self._rebuild_matrix()
            self._save()

    def remove_person(self, pid: str) -> bool:
        with self._lock:
            if pid in self.people:
                del self.people[pid]
                self._rebuild_matrix()
                self._save()
                return True
            return False

    # ------------------------------------------------------------- queries
    def list_people(self) -> list[Person]:
        with self._lock:
            return sorted(self.people.values(), key=lambda p: p.name.lower())

    def get(self, pid: str) -> Optional[Person]:
        with self._lock:
            return self.people.get(pid)

    def __len__(self) -> int:
        with self._lock:
            return len(self.people)

    def sample_count(self) -> int:
        with self._lock:
            return sum(p.sample_count for p in self.people.values())

    def match(self, encoding: np.ndarray, tolerance: float):
        """
        Best match for one encoding against the whole DB in a single
        vectorized pass. Returns (person_id_or_None, best_distance_or_None).
        """
        with self._lock:
            matrix, owners = self._matrix, self._owners
        if matrix is None:
            return None, None
        distances = np.linalg.norm(matrix - encoding, axis=1)
        idx = int(np.argmin(distances))
        best = float(distances[idx])
        if best <= tolerance:
            return owners[idx], best
        return None, best
