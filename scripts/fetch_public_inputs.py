"""Inspect/fetch small members from a pinned public mirror of Kaggle inputs.

This mirror enables CPU preparation. It does not establish competition entry,
current harness parity, current model eligibility, or a leaderboard score.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from gemma_particle.remote_zip import RangeFile

REVISION = "12aaee3902c7bb573d516113a695150d90d50250"
FILES = {
    "data": ("gemma-4-developer-agent.zip", 21923985083),
    "wheels": ("gemma-4-developer-agent-wheelhouse-v30.zip", 880849882),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", choices=FILES, default="data")
    parser.add_argument("--member", action="append", default=[])
    parser.add_argument("--out", type=Path, default=Path("data/public-mirror"))
    args = parser.parse_args()
    filename, size = FILES[args.archive]
    url = f"https://huggingface.co/datasets/HanG/gemma-4-developer-agent/resolve/{REVISION}/{filename}"
    with RangeFile(url, size) as remote, ZipFile(remote) as archive:
        if not args.member:
            for member in archive.infolist():
                print(json.dumps({"member": member.filename, "bytes": member.file_size}))
        else:
            receipts = []
            for name in args.member:
                member = archive.getinfo(name)
                if member.file_size > 64 * 1024**2:
                    raise ValueError("only small metadata/source members may be fetched")
                relative = PurePosixPath(name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("unsafe archive path")
                destination = args.out / relative
                payload = archive.read(member)
                if destination.exists() and destination.read_bytes() != payload:
                    raise FileExistsError(f"refusing to overwrite different input: {destination}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)
                receipts.append({"member": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
            args.out.mkdir(parents=True, exist_ok=True)
            receipt_path = args.out / f"{args.archive}-receipt.json"
            previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
            if previous and previous["source"] != url:
                raise ValueError("input directory belongs to a different archive source")
            by_name = {r["member"]: r for r in previous["members"]} if previous else {}
            for row in receipts:
                if row["member"] in by_name and by_name[row["member"]] != row:
                    raise ValueError("input receipt changed")
                by_name[row["member"]] = row
            receipt = {"source": url, "mirror_revision": REVISION,
                       "authority": "public mirror; requires current Kaggle verification", "members": list(by_name.values())}
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
            print(json.dumps(receipt, indent=2))
        print(json.dumps({"transferred_bytes": remote.transferred}))


if __name__ == "__main__":
    main()
