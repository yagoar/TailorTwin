"""Download the BodyM dataset from its public S3 bucket.

BodyM (Ruiz et al., arXiv:2210.05667) is hosted in the AWS Open Data
bucket ``s3://amazon-bodym`` (us-west-2, CC BY-NC 4.0) — see
https://github.com/awslabs/open-data-registry/blob/main/datasets/bodym.yaml

The bucket is public, so no AWS account or credentials are needed: the
S3 client is created with an unsigned config. The download is
**resumable** — a file already on disk with the right size is skipped —
and parallel over a thread pool.

Requires ``boto3`` (``pip install boto3``).

  python -m tailor_twin.bmnet.download_bodym --dest data/bodym
  python -m tailor_twin.bmnet.download_bodym --splits train testA
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUCKET = "amazon-bodym"
REGION = "us-west-2"


def _client():
    """Unsigned S3 client — the bucket is public, credentials not used."""
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.client import Config
    except ImportError as e:  # noqa: BLE001
        raise SystemExit("boto3 is required: pip install boto3") from e
    return boto3.client("s3", region_name=REGION,
                        config=Config(signature_version=UNSIGNED))


def _list_keys(s3, prefix: str) -> list[tuple[str, int]]:
    """Every (key, size) under ``prefix`` — paginated."""
    out: list[tuple[str, int]] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            out.append((obj["Key"], obj["Size"]))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dest", type=Path, default=Path("data/bodym"),
                   help="local directory the bucket is mirrored into")
    p.add_argument("--splits", nargs="+",
                   default=["train", "testA", "testB"],
                   help="BodyM splits to fetch")
    p.add_argument("--workers", type=int, default=16,
                   help="parallel download threads")
    args = p.parse_args(argv)

    s3 = _client()
    args.dest.mkdir(parents=True, exist_ok=True)

    keys: list[tuple[str, int]] = []
    for split in args.splits:
        split_keys = _list_keys(s3, f"{split}/")
        print(f"{split}: {len(split_keys)} objects")
        keys += split_keys
    print(f"total: {len(keys)} objects")

    # Resume: skip keys whose local copy already has the right size.
    todo = []
    for key, size in keys:
        dst = args.dest / key
        if dst.exists() and dst.stat().st_size == size:
            continue
        todo.append((key, dst))
    print(f"to download: {len(todo)}  (skipped {len(keys) - len(todo)} "
          f"already present)")
    if not todo:
        print("dataset already complete")
        return 0

    done = 0

    def fetch(item: tuple[str, Path]) -> None:
        nonlocal done
        key, dst = item
        dst.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(BUCKET, key, str(dst))
        done += 1
        if done % 250 == 0 or done == len(todo):
            print(f"  {done}/{len(todo)}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(fetch, todo))

    print(f"done -> {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
