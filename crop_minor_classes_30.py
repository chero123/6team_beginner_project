#데이터가 20개 미만이었던 클래스들에 대해 클래스별로 30장씩 크롭

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import cv2

# =========================================================
# 경로
# =========================================================
PROJECT_ROOT = Path(r"C:\Users\sangj\workspace\6team_beginner_project")
AIHUB_ROOT = PROJECT_ROOT / "aihub_data_json"

# ✅ 너가 준 매핑 파일 (key=dl_idx, value에 "(cls N)" 포함)
MAP_PATH = PROJECT_ROOT / "category_id_mapping.json"

# ✅ 출력 폴더(새로)
OUT_ROOT = PROJECT_ROOT / "cropped_add_minor30_v2"
OUT_IMG_DIR = OUT_ROOT / "images" / "train"
OUT_LBL_DIR = OUT_ROOT / "labels" / "train"

# =========================================================
# ✅ 부족하다고 한 클래스들 (cls 기준)
# =========================================================
TARGET_CLASSES = {
    6, 8, 11, 12, 17, 18, 19, 20, 21, 24, 29, 31, 45, 47, 50, 54, 55, 61
}

MAX_PER_CLASS = 30     # ✅ 클래스당 최대 몇 장 저장할지
PAD_RATIO = 0.20
MIN_CROP_WH = 32
DRY_RUN = False        # True면 저장 안 하고 카운트/로그만

IMG_EXTS = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]


# =========================================================
# 유틸
# =========================================================
def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def crop_with_padding(img, bbox_xywh: Tuple[float, float, float, float], pad_ratio: float):
    h, w = img.shape[:2]
    x, y, bw, bh = bbox_xywh

    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + bw))
    y2 = int(round(y + bh))

    pad_x = int(round(bw * pad_ratio))
    pad_y = int(round(bh * pad_ratio))

    cx1 = clamp(x1 - pad_x, 0, w - 1)
    cy1 = clamp(y1 - pad_y, 0, h - 1)
    cx2 = clamp(x2 + pad_x, 0, w)
    cy2 = clamp(y2 + pad_y, 0, h)

    cw = cx2 - cx1
    ch = cy2 - cy1
    if cw <= 1 or ch <= 1:
        return None

    cropped = img[cy1:cy2, cx1:cx2].copy()

    # 크롭 좌표계 bbox로 변환
    nx1 = clamp(x1 - cx1, 0, cw - 1)
    ny1 = clamp(y1 - cy1, 0, ch - 1)
    nx2 = clamp(x2 - cx1, 0, cw)
    ny2 = clamp(y2 - cy1, 0, ch)

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

def find_image_path(file_name: str) -> Optional[Path]:
    # 1) 상대경로 그대로 시도
    p = AIHUB_ROOT / file_name
    if p.exists():
        return p
    # 2) 파일명만으로 재귀 검색
    base = Path(file_name).name
    hits = list(AIHUB_ROOT.rglob(base))
    return hits[0] if hits else None

def is_valid_coco_like(d: Dict[str, Any]) -> bool:
    return isinstance(d, dict) and isinstance(d.get("images"), list) and isinstance(d.get("annotations"), list)

def parse_cls_from_mapping_value(s: str) -> Optional[int]:
    # value 예: "뉴로메드정(옥시라세탐) (cls 6)"
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


# =========================================================
# 메인
# =========================================================
def main():
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_LBL_DIR.mkdir(parents=True, exist_ok=True)

    # 1) dl_idx -> cls 만들기 (너가 준 JSON 활용)
    raw = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    dlidx_to_cls: Dict[int, int] = {}

    for k, v in raw.items():
        try:
            dlidx = int(str(k).strip())
        except:
            continue
        cls_id = parse_cls_from_mapping_value(v)
        if cls_id is None:
            continue
        if cls_id in TARGET_CLASSES:
            dlidx_to_cls[dlidx] = cls_id

    print(f"[INFO] AIHUB_ROOT = {AIHUB_ROOT}")
    print(f"[INFO] MAP_PATH = {MAP_PATH}")
    print(f"[INFO] TARGET_CLASSES = {sorted(TARGET_CLASSES)}")
    print(f"[INFO] TARGET dl_idx count = {len(dlidx_to_cls)}")
    print(f"[INFO] MAX_PER_CLASS = {MAX_PER_CLASS} | PAD_RATIO={PAD_RATIO} | DRY_RUN={DRY_RUN}")
    print(f"[INFO] OUTPUT = {OUT_ROOT}")

    # 클래스별 저장 카운트
    saved_per_cls = {c: 0 for c in TARGET_CLASSES}

    json_paths = list(AIHUB_ROOT.rglob("*.json"))
    print(f"[INFO] JSON files found: {len(json_paths)}")

    warn_parse = 0
    saved_total = 0
    skipped_write_fail = 0
    skipped_small = 0
    skipped_no_img = 0

    # 조기종료 체크용
    def all_done():
        return all(saved_per_cls[c] >= MAX_PER_CLASS for c in TARGET_CLASSES)

    for jp in json_paths:
        if all_done():
            break

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

        # image_id -> (file_name, dl_idx)
        id_to_info: Dict[int, Tuple[str, int]] = {}
        for im in data["images"]:
            try:
                iid = int(im.get("id"))
                fn = im.get("file_name") or im.get("imgfile")
                dl = int(str(im.get("dl_idx")).strip())
                if fn:
                    id_to_info[iid] = (fn, dl)
            except:
                continue

        if not id_to_info:
            continue

        # json 경로 고유 해시(덮어쓰기 방지용)
        jp_hash = hashlib.md5(str(jp).encode("utf-8")).hexdigest()[:8]

        for ann_idx, ann in enumerate(data["annotations"]):
            if all_done():
                break

            if "image_id" not in ann or "bbox" not in ann:
                continue

            try:
                img_id = int(ann["image_id"])
            except:
                continue

            if img_id not in id_to_info:
                continue

            file_name, dl_idx = id_to_info[img_id]

            # dl_idx가 타겟이면 cls 얻기
            if dl_idx not in dlidx_to_cls:
                continue

            cls_id = dlidx_to_cls[dl_idx]

            # 클래스별 cap
            if saved_per_cls[cls_id] >= MAX_PER_CLASS:
                continue

            bbox = ann["bbox"]
            try:
                bbox_xywh = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            except:
                continue

            img_path = find_image_path(file_name)
            if img_path is None:
                skipped_no_img += 1
                continue

            img = cv2.imread(str(img_path))
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

            stem = img_path.stem
            # ✅ jp_hash를 섞어서 덮어쓰기 방지
            out_stem = f"{stem}__dl{dl_idx}__ann{ann_idx}__j{jp_hash}"

            out_img_path = OUT_IMG_DIR / f"{out_stem}.jpg"
            out_lbl_path = OUT_LBL_DIR / f"{out_stem}.txt"

            yline = yolo_line(cls_id, tuple(map(int, nbbox_xywh)), cw, ch)

            if DRY_RUN:
                saved_per_cls[cls_id] += 1
                saved_total += 1
                continue

            ok = cv2.imwrite(str(out_img_path), cropped)
            if not ok:
                skipped_write_fail += 1
                continue

            out_lbl_path.write_text(yline + "\n", encoding="utf-8")

            saved_per_cls[cls_id] += 1
            saved_total += 1

            if saved_total % 200 == 0:
                print(f"[INFO] saved_total: {saved_total} | per_cls: " +
                      ", ".join([f"{c}:{saved_per_cls[c]}" for c in sorted(TARGET_CLASSES)]))

    print("\n[DONE]")
    print(f"saved_total: {saved_total}")
    print(f"warn_parse: {warn_parse}")
    print(f"skipped_no_img: {skipped_no_img}")
    print(f"skipped_small: {skipped_small}")
    print(f"skipped_write_fail: {skipped_write_fail}")
    print("saved_per_cls:")
    for c in sorted(TARGET_CLASSES):
        print(f"  cls {c}: {saved_per_cls[c]}")
    print(f"output: {OUT_ROOT}")

if __name__ == "__main__":
    main()
