"""Download the Date Fruit Dataset (Zenodo record 4639543) into data/raw/.

Resumable (complete files are skipped). Zenodo rate-limits parallel downloads,
so each file is size-checked, failures are retried with backoff, and the script
exits non-zero unless every file is on disk at the end.
"""
import concurrent.futures as cf
import json
import pathlib
import socket
import sys
import time
import urllib.request

RECORD = "https://zenodo.org/api/records/4639543/files"
OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"


def complete(entry):
    dest = OUT / entry["key"]
    return dest.exists() and dest.stat().st_size == entry["size"]


def fetch(entry):
    if complete(entry):
        return entry["key"], "skip"
    dest = OUT / entry["key"]
    tmp = dest.with_name(dest.name + ".part")
    err = None
    for attempt in range(8):
        try:
            urllib.request.urlretrieve(entry["links"]["content"], tmp)
            if tmp.stat().st_size != entry["size"]:
                raise OSError(f"got {tmp.stat().st_size} bytes, expected {entry['size']}")
            tmp.rename(dest)
            return entry["key"], "ok"
        except Exception as e:  # noqa: BLE001 - network errors of all kinds are retried
            err = e
            headers = getattr(e, "headers", None)
            wait = headers.get("Retry-After") if headers else None
            time.sleep(float(wait) if wait else min(60, 3 * 2**attempt))
    tmp.unlink(missing_ok=True)
    return entry["key"], f"FAILED: {err}"


def main():
    socket.setdefaulttimeout(60)
    OUT.mkdir(parents=True, exist_ok=True)
    entries = json.load(urllib.request.urlopen(RECORD))["entries"]
    with cf.ThreadPoolExecutor(3) as pool:
        for key, status in pool.map(fetch, entries):
            if status.startswith("FAILED"):
                print(key, status, flush=True)
    missing = [e["key"] for e in entries if not complete(e)]
    print(f"{len(entries) - len(missing)}/{len(entries)} files complete")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
