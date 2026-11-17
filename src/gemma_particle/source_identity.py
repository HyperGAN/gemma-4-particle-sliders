"""Reject an environment that does not implement the checked-out source lock."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path


def require_core_source():
    from particle_sliders import pg0100 as core
    root=Path(__file__).resolve().parents[2]
    lock=json.loads((root/'sources.lock.json').read_text())
    pg=lock['particlegan'];bridge=lock['bridge_provenance']
    if (version('particlegan')!=pg['version'] or core.PG_VERSION!=pg['version']
            or core.PG_SOURCE_COMMIT!=pg['commit']):
        raise RuntimeError('environment does not match sources.lock.json; bootstrap the current source cohort in its own environment')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if (sha(core.__file__)!=bridge['engine_sha256']
            or sha(Path(core.__file__).with_name('pg0100_sources.json'))!=bridge['manifest_sha256']):
        raise RuntimeError('shared conditional engine differs from the locked bridge')
    return {'pg_version':pg['version'],'pg_source_commit':pg['commit'],
        'pg_source_profile':core.PG_SOURCE_PROFILE,'pg_sources_sha256':core._released_sources(),
        'engine_sha256':bridge['engine_sha256'],'manifest_sha256':bridge['manifest_sha256']}
