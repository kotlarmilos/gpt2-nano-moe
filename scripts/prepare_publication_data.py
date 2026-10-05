from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import urllib.request


SOURCE = (
    "https://huggingface.co/kotlarmilos/gpt2-nano/resolve/"
    "ee547b4112bce2d2557dd0975038035bab6c86eb/"
    "bpe-shards/train_000000.bin"
)
SHA256 = "f08410ea036cc56b78ffc3f6edbf94f5a1d469aeb7c708bb56e4ce544623a69e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected: str = SHA256) -> bool:
    return path.is_file() and sha256(path) == expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/train.bin"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if verify(args.output):
        print(f"verified {args.output}")
        return
    if args.verify_only:
        raise ValueError(f"{args.output} is missing or failed SHA-256 verification")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    try:
        with urllib.request.urlopen(SOURCE, timeout=60) as response:
            with temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    if not verify(args.output):
        args.output.unlink(missing_ok=True)
        raise ValueError("downloaded corpus failed SHA-256 verification")
    print(f"downloaded and verified {args.output}")


if __name__ == "__main__":
    main()
