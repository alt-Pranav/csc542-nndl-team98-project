import re
import urllib.request
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from loguru import logger

from unet_mech.config import DEFAULT_CFG

MONTGOMERY_URL = (
    "https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip"
)
MONTGOMERY_BASE_URL = (
    "https://data.lhncbc.nlm.nih.gov/public/"
    "Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet"
)


def _request_url(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(_request_url(url), timeout=120) as response:
        dest.write_bytes(response.read())


def _remote_png_names(subdir: str) -> list[str]:
    with urllib.request.urlopen(_request_url(f"{MONTGOMERY_BASE_URL}/{subdir}/"), timeout=60) as response:
        html = response.read().decode("utf-8", "replace")
    return sorted(set(re.findall(r'href="([^"]+\.png)"', html)))


def _paired_count(root: Path) -> int:
    img = {p.name for p in (root / "CXR_png").glob("*.png")}
    left = {p.name for p in (root / "ManualMask" / "leftMask").glob("*.png")}
    right = {p.name for p in (root / "ManualMask" / "rightMask").glob("*.png")}
    return len(img & left & right)


def _has_complete_dataset(root: Path) -> bool:
    return _paired_count(root) >= 138


def _try_extract_archive(zip_path: Path, root: Path) -> bool:
    try:
        with ZipFile(zip_path) as zf:
            zf.extractall(root.parent)
    except BadZipFile:
        logger.warning(f"[data] Ignoring invalid archive at {zip_path}")
        return False
    return _has_complete_dataset(root)


def _download_directory_dataset(root: Path) -> None:
    names = set(_remote_png_names("CXR_png"))
    names.update(_remote_png_names("ManualMask/leftMask"))
    names.update(_remote_png_names("ManualMask/rightMask"))
    if not names:
        raise RuntimeError("Could not list Montgomery PNG files from NLM data directory")

    downloads: list[tuple[str, Path]] = []
    for name in sorted(names):
        for rel in ("CXR_png", "ManualMask/leftMask", "ManualMask/rightMask"):
            dest = root / rel / name
            if not dest.exists() or dest.stat().st_size == 0:
                downloads.append((f"{MONTGOMERY_BASE_URL}/{rel}/{name}", dest))

    if downloads:
        logger.info(f"[data] Downloading {len(downloads)} Montgomery files from NLM")
    for i, (url, dest) in enumerate(downloads, start=1):
        if i == 1 or i % 25 == 0 or i == len(downloads):
            logger.info(f"[data]   {i}/{len(downloads)} {dest.relative_to(root)}")
        _download(url, dest)


def download_montgomery(data_dir: str | None = None) -> Path:
    """
    Download the Montgomery County CXR dataset from the NLM OpenI server.

    Directory layout after extraction:
      <data_dir>/CXR_png/           — 138 greyscale chest PNGs
      <data_dir>/ManualMask/leftMask/
      <data_dir>/ManualMask/rightMask/
    """
    if data_dir is None:
        data_dir = DEFAULT_CFG["data_dir"]
    root = Path(data_dir)

    if _has_complete_dataset(root):
        logger.info(f"[data] Dataset already present at {root}")
        return root

    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "montgomery.zip"

    if zip_path.exists() and _try_extract_archive(zip_path, root):
        logger.info(f"[data] Extracted dataset from {zip_path}")
        return root

    if not zip_path.exists():
        logger.info("[data] Downloading Montgomery dataset …")
        _download(MONTGOMERY_URL, zip_path)
        logger.info(f"[data] Download complete → {zip_path}")

    if _try_extract_archive(zip_path, root):
        logger.info(f"[data] Extracted dataset from {zip_path}")
        return root

    _download_directory_dataset(root)
    if not _has_complete_dataset(root):
        raise RuntimeError(
            f"Montgomery dataset incomplete under {root}: "
            f"{_paired_count(root)} paired samples found"
        )

    logger.info(f"[data] Dataset ready at {root}")
    return root
