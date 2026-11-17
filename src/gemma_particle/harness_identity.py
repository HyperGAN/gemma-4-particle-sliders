"""Hash the actual harness sources and sandbox inputs used by a cohort."""
import hashlib
import importlib.metadata
from pathlib import Path


def sha256_file(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def package_identity():
    result = {}
    for name in ('swegemma', 'adk-submission', 'adk-eval-core', 'google-adk', 'litellm'):
        dist = importlib.metadata.distribution(name)
        result[name] = {'version': dist.version, 'sources': {
            str(p): sha256_file(dist.locate_file(p)) for p in sorted(dist.files or [])
            if str(p).endswith('.py') and dist.locate_file(p).is_file()}}
    return result


def wheel_identity(directory):
    return {p.name: sha256_file(p) for p in sorted(Path(directory).glob('*.whl'))}
