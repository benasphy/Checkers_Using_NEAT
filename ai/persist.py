"""Crash-safe persistence (power-outage-proof file writes).

Every write goes to a temp file in the same directory, is fsync'd, and is
then atomically renamed over the target (``os.replace`` is atomic on POSIX
and Windows). A power cut can therefore never leave a half-written
checkpoint, hall of fame, genome, or calibration file on disk: you keep
either the old file or the new one, never a corrupt one.
"""

from __future__ import annotations

import gzip
import json
import os
import pickle
import shutil
import tempfile


def _atomic_write(path: str, write_fn) -> None:
    dir_ = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as f:
            write_fn(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_pickle_dump(obj, path: str) -> None:
    _atomic_write(path, lambda f: pickle.dump(obj, f))


def atomic_pickle_dump_gzip(obj, path: str, compresslevel: int = 5) -> None:
    def write(f):
        # Closing the GzipFile writes the gzip trailer (end-of-stream
        # marker); without this the file is unreadable. fileobj=f stays
        # open so _atomic_write can flush+fsync it.
        with gzip.GzipFile(fileobj=f, mode="wb",
                           compresslevel=compresslevel) as gz:
            pickle.dump(obj, gz, protocol=pickle.HIGHEST_PROTOCOL)
    _atomic_write(path, write)


def atomic_json_dump(obj, path: str) -> None:
    _atomic_write(path, lambda f: f.write(json.dumps(obj, indent=2).encode()))


def atomic_text_dump(text: str, path: str) -> None:
    _atomic_write(path, lambda f: f.write(text.encode("utf-8")))


def atomic_directory_dump(files: dict[str, str], path: str) -> None:
    """Commit a group of UTF-8 files as one immutable directory."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    if os.path.exists(path):
        raise FileExistsError(f"Artifact already exists: {path}")
    tmp = tempfile.mkdtemp(dir=parent, prefix=".tmp-bundle-")
    try:
        for name, text in files.items():
            if os.path.isabs(name) or ".." in name.split(os.sep):
                raise ValueError(f"Unsafe bundle path: {name}")
            target = os.path.join(tmp, name)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w", encoding="utf-8", newline="") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def newest_checkpoint(out_dir: str, prefix: str = "neat-checkpoint-"):
    """Path of the highest-numbered checkpoint in out_dir, or None."""
    best, best_n = None, -1
    if not os.path.isdir(out_dir):
        return None
    for name in os.listdir(out_dir):
        if name.startswith(prefix):
            try:
                n = int(name[len(prefix):])
            except ValueError:
                continue
            if n > best_n:
                best, best_n = os.path.join(out_dir, name), n
    return best
