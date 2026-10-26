"""Download the Inside Airbnb snapshots listed in a config file and check them.

    python src/download.py --config configs/cape_town.yaml [--force]

Files go to data/raw/<city>/<snapshot>/ (git-ignored).
A summary goes to reports/download_manifest.csv.
Data: Inside Airbnb (https://insideairbnb.com), CC BY 4.0.
"""
import argparse
import csv
import gzip
import hashlib
import http.client
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml


""""
6 sections:
  config: load the YAML config file
  where is what: compute URLs and local paths
  check a file: read a downloaded file and check its columns
  download one: download a single file, with retries
  the manifest: read/write the CSV manifest of all files
  main: command-line interface
"""



ROOT = Path(__file__).resolve().parents[1]  # repo root: paths don't depend on where you run from
BASE_URL = "https://data.insideairbnb.com"
RAW_DIR, REPORTS_DIR = "data/raw", "reports"
RETRIES, TIMEOUT, RETRY_WAIT = 3, 60, 2

# name -> (path on the server, columns later steps (qa / kpi / viz) need).
# A file missing any of these columns gets status "error". Edit the lists here.
FILES = {
    "listings": ("data/listings.csv.gz",
                 ["id", "price", "neighbourhood_cleansed", "room_type", "latitude", "longitude",
                  "number_of_reviews", "minimum_nights", "last_scraped"]),
    "calendar": ("data/calendar.csv.gz", ["listing_id", "date", "available", "price", "minimum_nights"]),
    "reviews": ("data/reviews.csv.gz", ["listing_id", "id", "date", "comments"]),
    "neighbourhoods": ("visualisations/neighbourhoods.geojson", []),
}


class BadFile(Exception):
    """The file is corrupt or incomplete: throw it away and download again."""


class Fatal(Exception):
    """Retrying cannot help (HTTP 4xx, for example 404)."""


# IncompleteRead is raised by http.client when a transfer is cut short; it is not an OSError.
DOWNLOAD_ERRORS = (Fatal, BadFile, OSError, http.client.IncompleteRead)


# ---------------------------------------------------------------- config
def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["snapshots"] = [str(s) for s in cfg["snapshots"]]  # YAML turns 2025-09-28 into a date object
    return cfg


# ------------------------------------------------------------ where is what
def url_for(cfg, snap, name):
    return f"{BASE_URL}/{cfg['data_root']}/{snap}/{FILES[name][0]}"


def raw_file(cfg, snap, name):
    """Local path of a downloaded file. Other modules (qa.py, ...) call this."""
    return ROOT / RAW_DIR / cfg["city"] / snap / Path(FILES[name][0]).name


# -------------------------------------------------------------- check a file
def inspect_csv_gz(path):
    """Read the WHOLE gzip (this exposes truncation). Return (column names, line count)."""
    try:
        with gzip.open(path, "rb") as f:
            header = f.readline()
            lines = 0
            while chunk := f.read(1 << 20):
                lines += chunk.count(b"\n")
        return next(csv.reader([header.decode("utf-8")])), lines
    except (OSError, EOFError, UnicodeDecodeError) as e:  # BadGzipFile is an OSError
        raise BadFile(f"not a complete gzip CSV ({e})")


def inspect_geojson(path):
    """Return (property names, number of features)."""
    try:
        features = json.loads(path.read_text(encoding="utf-8")).get("features")
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise BadFile(f"not valid JSON ({e})")
    if not features:
        raise BadFile("GeoJSON has no features")
    return list(features[0].get("properties") or {}), len(features)


def inspect_file(path, name):
    """`name` is the key in FILES; the path may be a .part file, so do not guess from it."""
    return inspect_geojson(path) if name == "neighbourhoods" else inspect_csv_gz(path)


def judge(columns, name):
    """Compare columns with what later steps need -> (status, message)."""
    missing = [c for c in FILES[name][1] if c not in columns]
    return ("error", f"missing columns {missing}") if missing else ("ok", "")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------- download one
def fetch(url, dest):
    """Copy the URL into dest. 4xx -> Fatal; 5xx and network errors -> OSError (will be retried)."""
    request = urllib.request.Request(url, headers={"User-Agent": "uet-dse2049-student-project"})
    try:
        response = urllib.request.urlopen(request, timeout=TIMEOUT)
    except urllib.error.HTTPError as e:
        raise (Fatal if 400 <= e.code < 500 else OSError)(f"HTTP {e.code} for {url}")
    with response, open(dest, "wb") as out:
        shutil.copyfileobj(response, out, 1 << 20)


def download(url, dest, name):
    """Download to <name>.part, check it, and only then give it its real name."""
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, RETRIES + 1):
        try:
            fetch(url, part)
            info = inspect_file(part, name)
            part.replace(dest)
            return info
        except DOWNLOAD_ERRORS as e:
            part.unlink(missing_ok=True)
            if isinstance(e, Fatal) or attempt == RETRIES:
                raise
            print(f"  attempt {attempt}/{RETRIES} failed ({e}); retrying")
            time.sleep(RETRY_WAIT * attempt)


def get_file(cfg, snap, name, force):
    """Make sure one valid file is on disk. Return (action, columns, lines).

    With --force the old file stays until the new one has passed the check.
    """
    dest = raw_file(cfg, snap, name)
    if dest.exists() and not force:
        try:
            return ("skipped", *inspect_file(dest, name))
        except BadFile as e:
            print(f"  existing file is damaged ({e}); downloading again")
    return ("downloaded", *download(url_for(cfg, snap, name), dest, name))


# ------------------------------------------------------------- the manifest
FIELDS = ["snapshot", "file", "status", "action", "size_bytes", "lines", "sha256",
          "downloaded_at", "message", "url"]


def read_manifest(path):
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["snapshot"], r["file"]): r for r in csv.DictReader(f)}


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def process(cfg, snap, name, force, old):
    """Handle one file and return its manifest row (never raises for download problems)."""
    row = dict.fromkeys(FIELDS, "")
    row.update(snapshot=snap, file=name, url=url_for(cfg, snap, name))
    try:
        action, columns, lines = get_file(cfg, snap, name, force)
    except DOWNLOAD_ERRORS as e:
        return {**row, "status": "error", "action": "failed", "message": str(e)}
    path = raw_file(cfg, snap, name)
    status, message = judge(columns, name)
    # keep the ORIGINAL download time when we only re-checked an existing file
    when = (old or {}).get("downloaded_at", "")
    if action == "downloaded":
        when = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {**row, "status": status, "action": action, "size_bytes": path.stat().st_size,
            "lines": lines, "sha256": sha256(path), "downloaded_at": when, "message": message}


# --------------------------------------------------------------------- main
def run(cfg, force=False):
    """Returns 0 if every file is usable, else 1. (run_pipeline.py can call this.)"""
    manifest_path = ROOT / REPORTS_DIR / "download_manifest.csv"
    manifest = read_manifest(manifest_path)
    for snap in cfg["snapshots"]:
        for name in FILES:
            print(f"{snap}/{name}")
            manifest[(snap, name)] = process(cfg, snap, name, force, manifest.get((snap, name)))
            write_manifest(manifest_path, manifest.values())  # saved after every file
    rows = [manifest[(s, n)] for s in cfg["snapshots"] for n in FILES]
    print(f"\n{'snapshot':<12}{'file':<16}{'action':<12}{'MB':>8}  status")
    for r in rows:
        mb = f"{int(r['size_bytes']) / 1e6:.1f}" if r["size_bytes"] != "" else "-"
        print(f"{r['snapshot']:<12}{r['file']:<16}{r['action']:<12}{mb:>8}  {r['status']}  {r['message']}")
    print(f"manifest: {manifest_path}")
    return 1 if any(r["status"] == "error" for r in rows) else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and check Inside Airbnb snapshots.")
    parser.add_argument("--config", required=True, help="e.g. configs/cape_town.yaml")
    parser.add_argument("--force", action="store_true", help="download again even if files exist")
    args = parser.parse_args()
    sys.exit(run(load_config(args.config), args.force))