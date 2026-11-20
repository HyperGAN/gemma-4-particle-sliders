"""Verify the installed source closure before a native model is opened."""
import hashlib
from importlib.metadata import distribution, version
import json
from pathlib import Path


def require_core_source():
    from particle_sliders import engine as core
    from . import response_engine as bridge_engine

    lock = json.loads(Path(__file__).with_name("engine.lock.json").read_text())
    pg, shared, bridge = (lock[key] for key in ("particlegan", "particle_sliders", "bridge_provenance"))
    installed = distribution("particle-sliders-core")
    origin = json.loads(installed.read_text("direct_url.json") or "{}")
    if (version("particlegan") != pg["version"] or core.PG_VERSION != pg["version"]
            or core.PG_SOURCE_COMMIT != pg["commit"] or installed.version != shared["version"]
            or origin.get("url") != shared["url"]
            or origin.get("vcs_info", {}).get("commit_id") != shared["commit"]):
        raise RuntimeError("environment does not match the locked source cohort")
    engine = hashlib.sha256(Path(bridge_engine.__file__).read_bytes()).hexdigest()
    core_root = Path(core.__file__).parent
    actual = {name: hashlib.sha256((core_root / name).read_bytes()).hexdigest()
              for name in shared["sources"]}
    if actual != shared["sources"]:
        raise RuntimeError("shared engine source bytes differ from the locked dependency")
    manifest = hashlib.sha256((core_root / "engine_release.json").read_bytes()).hexdigest()
    if engine != bridge["engine_sha256"] or manifest != bridge["manifest_sha256"]:
        raise RuntimeError("conditional engine differs from the locked bridge or upstream manifest")
    return {"pg_version": pg["version"], "pg_source_commit": pg["commit"],
            "pg_source_profile": core.PG_SOURCE_PROFILE, "pg_sources_sha256": core._released_sources(),
            "engine_sha256": engine, "manifest_sha256": manifest}
