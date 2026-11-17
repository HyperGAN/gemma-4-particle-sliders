"""Install the exact unreleased source cohort in an already-created environment."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LOCK = json.loads((ROOT / "sources.lock.json").read_text())


def run(*args):
    subprocess.run(list(map(str, args)), check=True, cwd=ROOT)


destinations={}
for key, default_directory in [("particlegan", "ParticleGAN"), ("particle_sliders", "particle-sliders")]:
    pin = LOCK[key]
    directory=pin.get('directory',default_directory)
    destination = ROOT / ".deps" / directory
    destinations[key]=destination
    if not destination.exists():
        run("git", "clone", "--filter=blob:none", "--no-checkout", pin["url"], destination)
        run("git", "-C", destination, "sparse-checkout", "set", "particlegan", "particlegan_forge", "packages/particle-sliders-core")
        run("git", "-C", destination, "checkout", "--detach", pin["commit"])
    actual = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
    if actual != pin["commit"]:
        raise ValueError(f"source pin mismatch: {directory}")
    required=destination/('particlegan/__init__.py' if key=='particlegan' else 'packages/particle-sliders-core/pyproject.toml')
    if not required.exists():
        run('git','-C',destination,'sparse-checkout','set','particlegan','particlegan_forge','packages/particle-sliders-core')
        run('git','-C',destination,'checkout','--detach',pin['commit'])
    if "patch" in pin:
        patch = ROOT / pin["patch"]
        if hashlib.sha256(patch.read_bytes()).hexdigest() != pin["patch_sha256"]:
            raise ValueError("source patch hash mismatch")
        applied = subprocess.run(["git", "-C", str(destination), "apply", "--reverse", "--check", str(patch)], capture_output=True)
        if applied.returncode:
            run("git", "-C", destination, "apply", "--check", patch)
            run("git", "-C", destination, "apply", patch)
run(sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", destinations['particlegan'], "-e", destinations['particle_sliders']/'packages/particle-sliders-core', "-e", ROOT)
