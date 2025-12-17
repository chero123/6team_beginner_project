# ai 허브 전체에 대해 각 약별로 이미지를 crop해서 저장

import json
import hashlib
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from collections import defaultdict

import cv2

# =========================================================
# 경로
# =========================================================
PROJECT_ROOT = Path(r"C:\Users\sangj\workspace\6team_beginner_project")
AIHUB_ROOT = PROJECT_ROOT / "aihub_data_json"
MAP_PATH = PROJECT_ROOT / "category_id_mapping.json"

OUT_ROOT = PROJECT_ROOT / "cropped_all_full_raw_v1"
OUT_IMG_DIR = OUT_ROOT / "images" / "train"
OUT_LBL_DIR = OUT_ROOT / "labels" / "train"

# =========================================================
# 옵션
# =========================================================
PAD_RATIO = 0.20
MIN_CROP_WH = 32
JPEG_QUALITY = 90
DRY_RUN = False  # True면 저장 없이 카운트만

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP"}

LOG_EVERY = 2000

# ✅ 안전 종료(디스크 부족)
DISK_FREE_GB_MIN = 10.0
DISK_CHECK_EVERY = 500

# ✅ 재실행 resume 동작
# - True: out_img/out_lbl 둘 다 있으면 스킵 (중복 방지)
# - False: 무조건 다시 씀(비추천)
RESUME_SKIP_EXISTING = True

# =========================================================
# 유틸
# =========================================================
def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def crop_with_padding(img, bbox_xywh: Tuple[float, float, float, float], pad_ratio: float):
    H, W = img.shape[:2]
    x, y, bw, bh = bbox_xywh

    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + bw))
    y2 = int(round(y + bh))

    pad_x = int(round(bw * pad_ratio))
    pad_y = int(round(bh * pad_ratio))

    cx1 = clamp(x1 - pad_x, 0, W - 1)
    cy1 = clamp(y1 - pad_y, 0, H - 1)
    cx2 = clamp(x2 + pad_x, 0, W)
    cy2 = clamp(y2 + pad_y, 0, H)

    cw = cx2 - cx1
    ch = cy2 - cy1
    if cw <= 1 or ch <= 1:
        return None

    cropped = img[cy1:cy2, cx1:cx2].copy()

    nx1 = clamp(x1 - cx1, 0, cw - 1)
    ny1 = clamp(y1 - cy1, 0, ch - 1)
    nx2 = clamp(x2 - cx1, 1, cw)
    ny2 = clamp(y2 - cy1, 1, ch)

    nbw = max(1, nx2 - nx1)
    nbh = max(1, ny2 - ny1)
    return cropped, (nx1, ny1, nbw, nbh)

def yolo_line(cls_id: int, bbox_xywh: Tuple[int, int, int, int], img_w: int, img_h: int) -> str:
    x, y, w, h = bbox_xywh
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    return f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"

def is_valid_coco_like(d: Dict[str, Any]) -> bool:
    return isinstance(d, dict) and isinstance(d.get("images"), list) and isinstance(d.get("annotations"), list)

def parse_cls_from_mapping_value(s: str) -> Optional[int]:
    if not isinstance(s, str):
        return None
    key = "(cls "
    i = s.rfind(key)
    if i == -1:
        return None
    j = s.find(")", i)
    if j == -1:
        return None
    num = s[i + len(key):j].strip()
    try:
        return int(num)
    except:
        return None

def build_image_index(root: Path) -> Dict[str, Path]:
    t0 = time.time()
    idx: Dict[str, Path] = {}
    scanned = 0
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in IMG_EXTS:
            scanned += 1
            name = p.name
            if name not in idx:
                idx[name] = p
    dt = time.time() - t0
    print(f"[INFO] image index built: {len(idx)} unique names (scanned={scanned}) in {dt:.1f}s")
    return idx

def get_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(str(path))
    return usage.free / (1024 ** 3)

# =========================================================
# 메인
# =========================================================
def main():
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_LBL_DIR.mkdir(parents=True, exist_ok=True)

    free0 = get_free_gb(OUT_ROOT)
    print(f"[INFO] free disk at OUTPUT drive: {free0:.2f} GB (min={DISK_FREE_GB_MIN} GB)")
    if free0 < DISK_FREE_GB_MIN:
        print("[ABORT] Not enough free disk space to start.")
        return

    # 0) 이미지 인덱스(핵심 최적화)
    img_index = build_image_index(AIHUB_ROOT)

    # 1) dl_idx -> cls 매핑
    raw = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    dlidx_to_cls: Dict[int, int] = {}
    skipped_no_cls = 0
    for k, v in raw.items():
        try:
            dlidx = int(str(k).strip())
        except:
            continue
        cls_id = parse_cls_from_mapping_value(v)
        if cls_id is None:
            skipped_no_cls += 1
            continue
        dlidx_to_cls[dlidx] = cls_id

    mapped_classes = sorted(set(dlidx_to_cls.values()))
    print(f"[INFO] mapped dl_idx count = {len(dlidx_to_cls)} (skipped_no_cls={skipped_no_cls})")
    print(f"[INFO] mapped cls count    = {len(mapped_classes)}")
    print(f"[INFO] PAD_RATIO={PAD_RATIO} | JPEG_QUALITY={JPEG_QUALITY} | MIN_CROP_WH={MIN_CROP_WH}")
    print(f"[INFO] DRY_RUN={DRY_RUN} | RESUME_SKIP_EXISTING={RESUME_SKIP_EXISTING}")
    print(f"[INFO] OUTPUT = {OUT_ROOT}")

    # 통계
    saved_total = 0
    saved_per_cls = defaultdict(int)

    warn_parse = 0
    skipped_not_mapped = 0
    skipped_no_img = 0
    skipped_small = 0
    skipped_invalid_bbox = 0
    skipped_write_fail = 0
    skipped_already_done = 0

    t_start = time.time()
    last_disk_check_at = 0

    for jp_i, jp in enumerate(AIHUB_ROOT.rglob("*.json"), start=1):
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            try:
                data = json.loads(jp.read_text(encoding="utf-8", errors="ignore"))
            except Exception as e:
                warn_parse += 1
                if warn_parse <= 10:
                    print(f"[WARN] JSON parse fail: {jp} -> {e}")
                continue

        if not is_valid_coco_like(data):
            continue

        # image_id -> (file_name_only, dl_idx)
        id_to_info: Dict[int, Tuple[str, int]] = {}
        for im in data["images"]:
            try:
                iid = int(im.get("id"))
                fn = im.get("file_name") or im.get("imgfile")
                dl = int(str(im.get("dl_idx")).strip())
                if fn:
                    id_to_info[iid] = (Path(fn).name, dl)
            except:
                continue

        if not id_to_info:
            continue

        jp_hash = hashlib.md5(str(jp).encode("utf-8")).hexdigest()[:8]
        img_cache: Dict[str, Optional[Any]] = {}

        for ann_idx, ann in enumerate(data["annotations"]):
            if "image_id" not in ann or "bbox" not in ann:
                continue

            try:
                img_id = int(ann["image_id"])
            except:
                continue

            if img_id not in id_to_info:
                continue

            file_name_only, dl_idx = id_to_info[img_id]

            if dl_idx not in dlidx_to_cls:
                skipped_not_mapped += 1
                continue

            cls_id = dlidx_to_cls[dl_idx]

            bbox = ann["bbox"]
            try:
                bbox_xywh = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
                if bbox_xywh[2] <= 1 or bbox_xywh[3] <= 1:
                    skipped_invalid_bbox += 1
                    continue
            except:
                skipped_invalid_bbox += 1
                continue

            img_path = img_index.get(file_name_only)
            if img_path is None:
                skipped_no_img += 1
                continue

            # 파일명 고유화(덮어쓰기 방지)
            out_stem = f"{img_path.stem}__dl{dl_idx}__ann{ann_idx}__j{jp_hash}"
            out_img_path = OUT_IMG_DIR / f"{out_stem}.jpg"
            out_lbl_path = OUT_LBL_DIR / f"{out_stem}.txt"

            # ✅ RESUME: 둘 다 이미 존재하면 스킵
            if RESUME_SKIP_EXISTING and out_img_path.exists() and out_lbl_path.exists():
                skipped_already_done += 1
                continue

            # 이미지 로드 캐시
            if file_name_only in img_cache:
                img = img_cache[file_name_only]
            else:
                img = cv2.imread(str(img_path))
                img_cache[file_name_only] = img

            if img is None:
                skipped_no_img += 1
                continue

            cropped_pack = crop_with_padding(img, bbox_xywh, PAD_RATIO)
            if cropped_pack is None:
                continue

            cropped, nbbox_xywh = cropped_pack
            ch, cw = cropped.shape[:2]
            if cw < MIN_CROP_WH or ch < MIN_CROP_WH:
                skipped_small += 1
                continue

            yline = yolo_line(cls_id, tuple(map(int, nbbox_xywh)), cw, ch)

            if DRY_RUN:
                saved_total += 1
                saved_per_cls[cls_id] += 1
                continue

            ok = cv2.imwrite(str(out_img_path), cropped, [cv2.IMWRITE_JPEG_QUALITY, int(JPEG_QUALITY)])
            if not ok:
                skipped_write_fail += 1
                continue

            out_lbl_path.write_text(yline + "\n", encoding="utf-8")

            saved_total += 1
            saved_per_cls[cls_id] += 1

            # ✅ 디스크 안전 체크
            if saved_total - last_disk_check_at >= DISK_CHECK_EVERY:
                last_disk_check_at = saved_total
                free_gb = get_free_gb(OUT_ROOT)
                if free_gb < DISK_FREE_GB_MIN:
                    elapsed = time.time() - t_start
                    print("\n[STOP] Low disk space -> 안전 종료")
                    print(f"[STOP] free_gb={free_gb:.2f} GB < {DISK_FREE_GB_MIN} GB")
                    print(f"[STOP] saved_total(new)={saved_total} | skipped_already_done={skipped_already_done}")
                    print(f"[STOP] elapsed={elapsed/60:.1f} min | json_scanned~{jp_i}")
                    print(f"[STOP] output={OUT_ROOT}")
                    return

            if saved_total % LOG_EVERY == 0:
                elapsed = time.time() - t_start
                rate = saved_total / max(1e-9, elapsed)
                free_gb = get_free_gb(OUT_ROOT)
                print(f"[INFO] new_saved={saved_total} | skipped_done={skipped_already_done} | {rate:.2f} crops/s | elapsed={elapsed/60:.1f} min | free={free_gb:.1f} GB | json~{jp_i}")

    elapsed = time.time() - t_start
    print("\n[DONE]")
    print(f"new_saved_total: {saved_total} | elapsed: {elapsed/60:.1f} min")
    print(f"skipped_already_done: {skipped_already_done}")
    print(f"warn_parse: {warn_parse}")
    print(f"skipped_not_mapped: {skipped_not_mapped}")
    print(f"skipped_no_img: {skipped_no_img}")
    print(f"skipped_small: {skipped_small}")
    print(f"skipped_invalid_bbox: {skipped_invalid_bbox}")
    print(f"skipped_write_fail: {skipped_write_fail}")
    print(f"output: {OUT_ROOT}")

if __name__ == "__main__":
    main()
