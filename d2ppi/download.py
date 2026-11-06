"""Download the D2PPI benchmarks from Zenodo (record 23874759) and unpack them.

    python -m d2ppi.download                      # everything: benchmark.zip (6.6 GB), md5-checked, unzipped
    python -m d2ppi.download --only bernett_human --splits test     # just these files (~160 MB), no full download

Files land in data/benchmark/<benchmark>/<split>.{tsv,npz} (+ summary.json). The full download resumes if it is
interrupted. The selective mode reads only the requested members out of the remote zip (HTTP range requests) and
checks each one's CRC-32.
"""

import argparse
import hashlib
import io
import os
import shutil
import sys
import urllib.request
import zipfile

RECORD = "https://zenodo.org/records/23874759"
URL = f"{RECORD}/files/benchmark.zip?download=1"
MD5 = "33237993b9c9458fe93e7b5471fbe4bf"
SIZE = 6604010441
BENCHMARKS = ("bernett_human", "dscript_xspecies", "d2ppi_xspecies")
SPLITS = ("train", "val", "test")
UA = {"User-Agent": "d2ppi-download"}


def _progress(done, total, t="download"):
    if total:
        sys.stderr.write(f"\r[{t}] {done / 1e9:6.2f} / {total / 1e9:.2f} GB ({100 * done / total:5.1f} %)")
        sys.stderr.flush()


def md5sum(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def fetch(url, dest):
    """download url -> dest, resuming a partial file."""
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    req = urllib.request.Request(url, headers={**UA, **({"Range": f"bytes={have}-"} if have else {})})
    with urllib.request.urlopen(req) as r:
        if have and r.status != 206:  # server ignored the range: start over
            have = 0
        total = have + int(r.headers.get("Content-Length", 0))
        with open(part, "ab" if have else "wb") as f:
            done = have
            while b := r.read(1 << 22):
                f.write(b)
                done += len(b)
                _progress(done, total)
    sys.stderr.write("\n")
    os.replace(part, dest)


class RemoteFile(io.RawIOBase):
    """a read-only, seekable view of a remote file through HTTP range requests."""

    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n is None or n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        req = urllib.request.Request(self.url, headers={**UA, "Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        with urllib.request.urlopen(req) as r:
            b = r.read()
        self.pos += len(b)
        return b

    def readinto(self, buf):
        b = self.read(len(buf))
        buf[: len(b)] = b
        return len(b)


def members(names, only, splits):
    keep = []
    for n in names:
        parts = n.split("/")
        if len(parts) != 2 or not parts[1] or parts[0] not in only:
            continue
        stem = parts[1].split(".")[0]
        if stem in splits or parts[1] == "summary.json":
            keep.append(n)
    return keep


def extract(zf, names, out):
    """extract the members `names` of zf into out (zipfile checks every member's CRC-32 as it reads)."""
    for n in names:
        dest = os.path.join(out, *n.split("/"))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        info = zf.getinfo(n)
        with zf.open(info) as src, open(dest + ".part", "wb") as f:
            done = 0
            while b := src.read(1 << 22):
                f.write(b)
                done += len(b)
                _progress(done, info.file_size, n)
        sys.stderr.write("\n")
        os.replace(dest + ".part", dest)


def main():
    ap = argparse.ArgumentParser(description="Download the D2PPI benchmarks from Zenodo")
    ap.add_argument("--out", default="data/benchmark", help="where the benchmark folders go (default data/benchmark)")
    ap.add_argument("--only", nargs="+", choices=BENCHMARKS, help="selective mode: only these benchmarks")
    ap.add_argument("--splits", nargs="+", choices=SPLITS, help="selective mode: only these splits")
    ap.add_argument("--keep-zip", action="store_true", help="full mode: keep benchmark.zip after unpacking")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.only or a.splits:  # selective: read the wanted members straight from the remote zip
        only, splits = a.only or BENCHMARKS, a.splits or SPLITS
        rf = io.BufferedReader(RemoteFile(URL, SIZE), buffer_size=1 << 25)  # 32 MB per range request
        with zipfile.ZipFile(rf) as zf:
            names = members(zf.namelist(), only, splits)
            print(f"[download] {len(names)} files from {RECORD}", file=sys.stderr)
            extract(zf, names, a.out)
    else:
        zpath = os.path.join(a.out, "benchmark.zip")
        if not (os.path.exists(zpath) and os.path.getsize(zpath) == SIZE):
            print(f"[download] {URL} -> {zpath} ({SIZE / 1e9:.1f} GB)", file=sys.stderr)
            fetch(URL, zpath)
        print("[download] checking md5 ...", file=sys.stderr)
        got = md5sum(zpath)
        if got != MD5:
            raise SystemExit(f"md5 mismatch for {zpath}: {got} != {MD5}; delete it and download again")
        with zipfile.ZipFile(zpath) as zf:
            extract(zf, members(zf.namelist(), BENCHMARKS, SPLITS), a.out)
        if not a.keep_zip:
            os.remove(zpath)
    free = shutil.disk_usage(a.out).free / 1e9
    print(f"[download] done: {os.path.abspath(a.out)} ({free:.0f} GB free)", file=sys.stderr)


if __name__ == "__main__":
    main()
