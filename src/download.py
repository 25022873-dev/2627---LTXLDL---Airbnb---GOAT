"""Tải dữ liệu Inside Airbnb cho một thành phố: python src/download.py configs/cape_town.yaml"""
import gzip
import sys
import time
from pathlib import Path

import requests
import yaml
from urllib3.exceptions import HTTPError as RawError  # r.raw bỏ qua lớp bọc lỗi của requests

ROOT = Path(__file__).resolve().parents[1]  # gốc repo: không phụ thuộc thư mục đang đứng
BASE_URL = "https://data.insideairbnb.com"
FILES = ["listings.csv.gz", "calendar.csv.gz", "reviews.csv.gz"]
RETRIES = 3

def build_url(data_root, snapshot, file):
    return f"{BASE_URL}/{data_root}/{snapshot}/data/{file}"


def build_path(data_root, snapshot, file):
    return ROOT / "data/raw" / data_root / snapshot / file


def is_valid(path):
    """Tệp tồn tại và đọc hết được luồng gzip (bắt lỗi cắt cụt / sai CRC)."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with gzip.open(path, "rb") as f:
            while f.read(1 << 20):
                pass
        return True
    except (OSError, EOFError):
        return False


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        expected = int(r.headers.get("Content-Length", 0))
        with open(tmp, "wb") as f:  # byte thô, không để requests giải nén gzip
            for chunk in r.raw.stream(1 << 20, decode_content=False):
                f.write(chunk)
    if expected and tmp.stat().st_size != expected:
        raise IOError(f"size mismatch: {tmp.stat().st_size} != {expected}")
    tmp.replace(path)


def fetch(url, path):
    """Tải (bỏ qua nếu tệp đã hợp lệ), thử lại tối đa RETRIES lần. Trả về True nếu có tệp hợp lệ."""
    if is_valid(path):
        print(f"[skip] {path}")
        return True
    for attempt in range(RETRIES + 1):
        try:
            download(url, path)
            if is_valid(path):
                print(f"[ok]   {path}")
                return True
            raise IOError("gzip integrity check failed")
        except (OSError, RawError) as e:  # OSError gồm cả requests.RequestException
            print(f"[fail] {url} (attempt {attempt + 1}/{RETRIES + 1}): {e}")
            path.unlink(missing_ok=True)
            time.sleep(2 ** attempt)
    return False


def main(config_path):
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    root = cfg["data_root"].strip("/ ")
    failed = [build_url(root, snap, file)
              for snap in map(str, cfg["snapshots"]) for file in FILES
              if not fetch(build_url(root, snap, file), build_path(root, snap, file))]
    if failed:
        sys.exit("Tải thất bại:\n" + "\n".join(failed))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/cape_town.yaml")