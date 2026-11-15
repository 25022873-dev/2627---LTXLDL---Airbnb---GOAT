"""Tải dữ liệu Inside Airbnb: python src/download.py configs/cape_town.yaml"""
import csv
import gzip
import sys
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://data.insideairbnb.com"
KEY_COLS = {  # tệp -> cột khoá bắt buộc
    "listings.csv.gz": {"id"},
    "calendar.csv.gz": {"listing_id", "date"},
    "reviews.csv.gz": {"listing_id", "id", "date"},
}
RETRIES = 3


def build_url(data_root, snapshot, file):
    """<URL gốc><mốc>/data/<tệp>, đúng mẫu trong tệp Excel phân công."""
    return f"{BASE_URL}/{data_root}/{snapshot}/data/{file}"


def build_path(data_root, snapshot, file):
    return ROOT / "data" / "raw" / data_root / snapshot / file


def is_valid(path):
    """Tệp tồn tại, đọc hết được gzip (không cắt cụt/sai CRC) và đủ cột khoá."""
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
            if not KEY_COLS[path.name] <= set(next(csv.reader(f), [])):
                return False
            while f.read(1 << 20):
                pass
        return True
    except Exception:
        return False


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:  # byte thô, không để requests giải nén
            for chunk in r.raw.stream(1 << 20, decode_content=False):
                f.write(chunk)
    tmp.replace(path)


def fetch(url, path):
    """Bỏ qua nếu tệp đã hợp lệ; lỗi gì cũng thử lại tối đa RETRIES lần."""
    if is_valid(path):
        print(f"[skip] {path}")
        return True
    for attempt in range(RETRIES + 1):
        try:
            download(url, path)
            if not is_valid(path):
                raise ValueError("tệp hỏng hoặc thiếu cột khoá")
            print(f"[ok]   {path}")
            return True
        except Exception as e:
            print(f"[fail] {url} (lần {attempt + 1}/{RETRIES + 1}): {e}")
            path.unlink(missing_ok=True)
            time.sleep(2 ** attempt if attempt < RETRIES else 0)
    return False


def main(config_path):
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = cfg["data_root"].strip("/ ")
    failed = [build_url(root, snap, file)
              for snap in map(str, cfg["snapshots"]) for file in KEY_COLS
              if not fetch(build_url(root, snap, file), build_path(root, snap, file))]
    if failed:
        sys.exit("Tải thất bại:\n" + "\n".join(failed))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/cape_town.yaml")