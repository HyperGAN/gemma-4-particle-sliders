"""Restore the original bytes of the frozen selection evidence."""

import argparse
from hashlib import sha256
import os
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile
from urllib.error import URLError
from urllib.request import urlopen

ARCHIVE_REPOSITORY = "gemma-4-particle-sliders"
ARCHIVE_COMMIT = "f97a3e9a0711a8b81f87e396a88ff125090f0c26"
ARCHIVED_FILES = {
    'research/dev-selection-v1.json': (10759, "e20a744e77d3dbeb2bb22ac857d49821bc2c6bd1edbaffa6a7bf69556fc1ebf5"),
    'research/dev-selection-empf-v1.json': (9261, "b1f63deebb50acb29e15560fecd56b6509edbc3031d8204f37b4737eeb9b2911"),
}


def _verified(data, relative):
    size, expected = ARCHIVED_FILES[relative]
    if len(data) != size or sha256(data).hexdigest() != expected:
        raise ValueError(f"Archived evidence differs from its recorded SHA-256: {relative}")
    return data


def _read(path, size):
    with path.open("rb") as stream:
        return stream.read(size + 1)


def _git_blob(root, relative):
    try:
        result = subprocess.run(
            ["git", "--no-replace-objects", "cat-file", "blob", f"{ARCHIVE_COMMIT}:{relative}"],
            cwd=root,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def _original_bytes(root, relative, size, archive_dir, allow_network):
    if archive_dir is not None:
        return _read(Path(archive_dir) / relative, size)
    data = _git_blob(root, relative)
    if data is not None:
        return data
    if not allow_network:
        raise FileNotFoundError(f"Archived evidence is absent from local Git history: {relative}")
    url = f"https://raw.githubusercontent.com/HyperGAN/{ARCHIVE_REPOSITORY}/{ARCHIVE_COMMIT}/{relative}"
    try:
        with urlopen(url, timeout=20) as response:
            return response.read(size + 1)
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not fetch {relative}. Supply the original checkout with --archive-dir.") from exc


def restore_file(root, relative, *, archive_dir=None, allow_network=True):
    """Materialize one pinned file without replacing any existing evidence."""
    size, _ = ARCHIVED_FILES[relative]
    root = Path(root).resolve()
    destination = root / relative
    if destination.exists():
        _verified(_read(destination, size), relative)
        return destination

    data = _original_bytes(root, relative, size, archive_dir, allow_network)
    _verified(data, relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.close()
            try:
                os.link(temporary, destination)
            except FileExistsError:
                _verified(_read(destination, size), relative)
        finally:
            temporary.unlink()
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="destination root")
    parser.add_argument("--archive-dir", type=Path, help="offline copy of the original source checkout")
    parser.add_argument("--offline", action="store_true", help="allow only local sources")
    options = parser.parse_args(argv)
    for relative in ARCHIVED_FILES:
        path = restore_file(
            options.root, relative, archive_dir=options.archive_dir, allow_network=not options.offline
        )
        print(path)


if __name__ == "__main__":
    main()
