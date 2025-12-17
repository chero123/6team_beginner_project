# 데이터가 하나도 존재하지 않았던 62, 62, 65, 66, 67 클래스에 대해 각각의 약을 crop해서 저장

import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import cv2

# =========================================================
# 경로
# =========================================================
PROJECT_ROOT = Path(r"C:\Users\sangj\workspace\6team_beginner_project")
AIHUB_ROOT = PROJECT_ROOT / "aihub_data_json"

# ✅ 저장: 프로젝트 루트에 cropped_add 생성
OUT_ROOT = PROJECT_ROOT / "cropped_add"
OUT_IMG_DIR = OUT_ROOT / "images" / "train"
OUT_LBL_DIR = OUT_ROOT / "labels" / "train"
OUT_META_DIR = OUT_ROOT / "meta" / "train"

# =========================================================
# ✅ 타겟: images[].dl_idx 값 -> YOLO cls
# =========================================================
TARGET_DLIDX_TO_CLS = {
    5093: 62,
    5885: 63,
    27652: 65,
    22626: 66,
    23222: 67,
    6191: 68,
}

PAD_RATIO = 0.20
MIN_CROP_WH = 32
DRY_RUN = False  # True면 저장 안 하고 카운트/로그만

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

# =========================================================
# 메인
# =========================================================
def main():
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_LBL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_META_DIR.mkdir(parents=True, exist_ok=True)

    json_paths = list(AIHUB_ROOT.rglob("*.json"))
    print(f"[INFO] AIHUB_ROOT = {AIHUB_ROOT}")
    print(f"[INFO] JSON files found: {len(json_paths)}")
    print(f"[INFO] TARGET_DLIDX_TO_CLS: {TARGET_DLIDX_TO_CLS}")  # ✅ 이 로그가 떠야 정상
    print(f"[INFO] DRY_RUN={DRY_RUN} | PAD_RATIO={PAD_RATIO}")
    print(f"[INFO] OUTPUT = {OUT_ROOT}")

    saved = 0
    warn_parse = 0
    skipped_non_coco = 0
    skipped_no_img = 0
    skipped_small = 0

    for jp in json_paths:
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            try:
                data = json.loads(jp.read_text(encoding="utf-8", errors="ignore"))
            except Exception as e:
                warn_parse += 1
                # 너무 많이 찍히면 지저분하니까 20개까지만 출력
                if warn_parse <= 20:
                    print(f"[WARN] JSON parse fail: {jp} -> {e}")
                continue

        if not is_valid_coco_like(data):
            skipped_non_coco += 1
            continue

        # ✅ image_id -> (file_name, dl_idx) 맵
        id_to_info: Dict[int, Tuple[str, int]] = {}
        for im in data["images"]:
            try:
                iid = int(im.get("id"))
                fn = im.get("file_name") or im.get("imgfile")
                dl = int(str(im.get("dl_idx")).strip())
                if fn:
                    id_to_info[iid] = (fn, dl)
            except Exception:
                continue

        if not id_to_info:
            continue

        # annotations 돌면서 bbox 크롭
        for ann_idx, ann in enumerate(data["annotations"]):
            if "image_id" not in ann or "bbox" not in ann:
                continue

            try:
                img_id = int(ann["image_id"])
            except Exception:
                continue

            if img_id not in id_to_info:
                continue

            file_name, dl_idx = id_to_info[img_id]

            # ✅ 타겟 dl_idx만
            if dl_idx not in TARGET_DLIDX_TO_CLS:
                continue

            cls_id = TARGET_DLIDX_TO_CLS[dl_idx]

            bbox = ann["bbox"]
            try:
                bbox_xywh = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            except Exception:
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
            out_stem = f"{stem}__dl{dl_idx}__ann{ann_idx}"

            out_img_path = OUT_IMG_DIR / f"{out_stem}.jpg"
            out_lbl_path = OUT_LBL_DIR / f"{out_stem}.txt"
            out_meta_path = OUT_META_DIR / f"{out_stem}.json"

            yline = yolo_line(cls_id, tuple(map(int, nbbox_xywh)), cw, ch)

            if DRY_RUN:
                saved += 1
                if saved <= 30:
                    print(f"[DRY] {out_stem} -> cls {cls_id} | src={img_path.name}")
                continue

            cv2.imwrite(str(out_img_path), cropped)
            out_lbl_path.write_text(yline + "\n", encoding="utf-8")

            meta = {
                "source_json": str(jp),
                "source_image": str(img_path),
                "source_file_name": file_name,
                "dl_idx": dl_idx,
                "cls": cls_id,
                "source_bbox_xywh": bbox_xywh,
                "crop_pad_ratio": PAD_RATIO,
                "crop_image_size": [cw, ch],
                "crop_bbox_xywh": [int(nbbox_xywh[0]), int(nbbox_xywh[1]), int(nbbox_xywh[2]), int(nbbox_xywh[3])],
            }
            out_meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

            saved += 1
            if saved % 200 == 0:
                print(f"[INFO] saved: {saved}")

    print("\n[DONE]")
    print(f"saved: {saved}")
    print(f"warn_parse: {warn_parse}")
    print(f"skipped_non_coco: {skipped_non_coco}")
    print(f"skipped_no_img: {skipped_no_img}")
    print(f"skipped_small: {skipped_small}")
    print(f"output: {OUT_ROOT}")

if __name__ == "__main__":
    main()
