"""Package a prompt baseline or a measured PEFT conversion, never native weights."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

import yaml
from gemma_particle.curation import TARGET_INSTRUCTIONS

MODELS = [f"gemma-4-{size}-it-qat-w4a16-ct" for size in ("31b", "12b", "e4b", "e2b")]
ROOT = Path(__file__).resolve().parents[1]


def build(out, model, adapter=None, instruction=None):
    out = Path(out)
    if out.exists():
        raise FileExistsError("build directory already exists; use a new artifact path")
    if model not in MODELS:
        raise ValueError("unknown competition model")
    agent = out / "agent"
    shutil.copytree(ROOT / "submission", agent)
    text = (agent / "agent.yaml").read_text().replace(MODELS[0], model)
    if adapter:
        adapter = Path(adapter)
        config = json.loads((adapter / "adapter_config.json").read_text())
        if config.get("peft_type") != "LORA" or not 1 <= config.get("r", 0) <= 128:
            raise ValueError("a rank <=128 PEFT LoRA export is required")
        expected_base = "google/" + model.removesuffix("-qat-w4a16-ct").replace("-e2b-", "-E2B-").replace("-e4b-", "-E4B-").replace("-31b-", "-31B-").replace("-12b-", "-12B-")
        if config.get("base_model_name_or_path", "").lower() not in (expected_base.lower(), ("google/" + model).lower()):
            raise ValueError("adapter base does not match the selected competition model")
        destination = agent / "adapters" / "particle_lora"
        destination.mkdir(parents=True)
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            source = adapter / name
            if source.is_symlink() or not source.is_file():
                raise ValueError("adapter must contain real PEFT files")
            shutil.copyfile(source, destination / name)
        text += "adapter: particle_lora\n"
    (agent / "agent.yaml").write_text(text)
    if instruction:
        prompt = agent / "prompts/system.md"
        prompt.write_text(prompt.read_text().rstrip() + "\n\n" + TARGET_INSTRUCTIONS[instruction] + "\n")
    files = sorted(p for p in agent.rglob("*") if p.is_file())
    if sum(p.stat().st_size for p in files) >= 3 * 1024**3:
        raise ValueError("unpacked submission must be smaller than 3 GiB")
    archive = out / "submission.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
        for path in files:
            if path.is_symlink():
                raise ValueError("symlinks forbidden")
            info = ZipInfo(str(path.relative_to(agent)), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, path.read_bytes())
    receipt = {"model": model, "arm": "distilled-particle" if adapter else "prompt-"+instruction if instruction else "off",
               "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
               "files": {str(p.relative_to(agent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
               "official_validation": "pending", "scorer_vllm_parity": "unmeasured", "kaggle_score": None}
    (out / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, default=MODELS[0])
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--target-instruction", choices=TARGET_INSTRUCTIONS)
    args = parser.parse_args()
    print(json.dumps(build(args.out, args.model, args.adapter, args.target_instruction), indent=2))
