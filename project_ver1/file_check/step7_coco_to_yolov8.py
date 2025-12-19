import os
import re
import json
import shutil
from tqdm import tqdm

# =========================
# [INPUTS]
# =========================
IMG_ROOT = "/home/ohs3201/work/step2_unified_coco/images"  # symlink로 모아둔 이미지 루트 (png들)
TRAIN_JSON = "/home/ohs3201/work/step4_runs_remap/train_remap.json"
VAL_JSON   = "/home/ohs3201/work/step4_runs_remap/val_remap.json"
REMAPPED_ID_JSON = "/home/ohs3201/work/step4_runs_remap/category_id_remap.json"  # orig_to_train_id + K
CATEGORY_ID_MAPPING_JSON = "/mnt/c/Users/ohs32/Desktop/codeit/01.데이터/박상진/category_id_mapping.json"  # dl_idx -> "(cls N)" 문자열

# =========================
# [OUTPUT]
# =========================
OUT_ROOT = "/home/ohs3201/work/step7_yolov8"
IMG_OUT_TRAIN = os.path.join(OUT_ROOT, "images", "train")
IMG_OUT_VAL   = os.path.join(OUT_ROOT, "images", "val")
LBL_OUT_TRAIN = os.path.join(OUT_ROOT, "labels", "train")
LBL_OUT_VAL   = os.path.join(OUT_ROOT, "labels", "val")

# -------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def safe_symlink(src, dst):
    # dst가 이미 있으면 제거 후 재생성
    if os.path.islink(dst) or os.path.exists(dst):
        try:
            os.remove(dst)
        except IsADirectoryError:
            shutil.rmtree(dst)
    os.symlink(src, dst)

def coco_bbox_to_yolo(x, y, bw, bh, W, H):
    # (x,y,w,h) -> (xc,yc,w,h) normalized
    xc = (x + bw / 2.0) / W
    yc = (y + bh / 2.0) / H
    nw = bw / W
    nh = bh / H
    return xc, yc, nw, nh

def parse_dlidx_to_cls_map(mapping_json_path):
    """
    category_id_mapping.json 예:
      "16550": "동아가바펜틴정 800mg (cls 16)"
    => {16550: 16}
    """
    j = json.load(open(mapping_json_path, "r", encoding="utf-8"))
    dl_to_cls = {}
    pat = re.compile(r"\(cls\s*(\d+)\)")
    for k, v in j.items():
        try:
            dl = int(k)
        except:
            continue
        m = pat.search(str(v))
        if not m:
            continue
        cls = int(m.group(1))
        dl_to_cls[dl] = cls
    return dl_to_cls

def build_trainid_to_dlidx(remap_json_path):
    """
    category_id_remap.json:
      {"orig_to_train_id": {"1899":2, ...}, "K":63}
    => invert => {2:1899, ...}
    """
    r = json.load(open(remap_json_path, "r", encoding="utf-8"))
    orig_to_train = r["orig_to_train_id"]
    train_to_orig = {}
    for orig_str, train_id in orig_to_train.items():
        train_to_orig[int(train_id)] = int(orig_str)
    K = int(r["K"])
    return train_to_orig, K

def convert_split(coco_json_path, img_out_dir, lbl_out_dir, split_name, train_to_dlidx, dlidx_to_cls, K):
    ensure_dir(img_out_dir)
    ensure_dir(lbl_out_dir)

    coco = json.load(open(coco_json_path, "r", encoding="utf-8"))

    images = coco.get("images", [])
    anns   = coco.get("annotations", [])

    # image_id -> image meta
    img_by_id = {im["id"]: im for im in images}

    # image_id -> list of anns
    ann_by_img = {}
    for a in anns:
        ann_by_img.setdefault(a["image_id"], []).append(a)

    saved_images = 0
    skipped_zero_wh = 0
    skipped_bad_bbox = 0
    skipped_bad_class = 0
    wrote_labels = 0

    pbar = tqdm(images, desc=f"[{split_name}]", ncols=140)
    for im in pbar:
        img_id = im["id"]
        fn = im.get("file_name", None)
        W = int(im.get("width", 0))
        H = int(im.get("height", 0))

        if not fn:
            continue

        # 이미지 w/h가 0이면 변환 불가
        if W <= 0 or H <= 0:
            skipped_zero_wh += 1
            continue

        src_img = os.path.join(IMG_ROOT, fn)
        if not os.path.exists(src_img):
            # 이미지가 없으면 스킵(여기서 터지면 step2 이미지 symlink 문제가 다시 의심됨)
            continue

        # 이미지 symlink 생성
        dst_img = os.path.join(img_out_dir, fn)
        try:
            safe_symlink(src_img, dst_img)
        except OSError:
            # symlink 실패하면 복사로 fallback
            if not os.path.exists(dst_img):
                shutil.copy2(src_img, dst_img)

        saved_images += 1

        # 라벨 파일 생성
        lbl_path = os.path.join(lbl_out_dir, os.path.splitext(fn)[0] + ".txt")
        lines = []

        for a in ann_by_img.get(img_id, []):
            bbox = a.get("bbox", [])
            if not isinstance(bbox, list) or len(bbox) != 4:
                skipped_bad_bbox += 1
                continue
            x, y, bw, bh = bbox

            # bbox 유효성
            try:
                x = float(x); y = float(y); bw = float(bw); bh = float(bh)
            except:
                skipped_bad_bbox += 1
                continue

            if bw <= 0 or bh <= 0:
                skipped_bad_bbox += 1
                continue

            # COCO category_id는 train_id (1..K) 여야 함
            train_id = a.get("category_id", None)
            if train_id is None:
                skipped_bad_class += 1
                continue
            try:
                train_id = int(train_id)
            except:
                skipped_bad_class += 1
                continue

            # train_id -> dl_idx
            dl_idx = train_to_dlidx.get(train_id, None)
            if dl_idx is None:
                skipped_bad_class += 1
                continue

            # dl_idx -> cls (0..)
            cls = dlidx_to_cls.get(dl_idx, None)
            if cls is None:
                skipped_bad_class += 1
                continue

            # cls 범위 체크 (0..K-1 권장. 데이터가 K와 다를 수 있으면 여기서 바로 잡힘)
            if not (0 <= cls <= 10**9):
                skipped_bad_class += 1
                continue

            xc, yc, nw, nh = coco_bbox_to_yolo(x, y, bw, bh, W, H)

            # clamp (안전)
            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 <= nw <= 1 and 0 <= nh <= 1):
                # 이미지 밖 bbox가 있더라도 학습은 가능하지만, 극단값은 스킵
                # (원하면 clamp로 바꿀 수 있음)
                pass

            lines.append(f"{cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        # 라벨이 없으면 빈 파일로 두지 말고 "파일 자체를 만들지 않음" (Ultralytics 규칙에 안전)
        if lines:
            with open(lbl_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            wrote_labels += 1

    print(f"\n[{split_name} DONE]")
    print(f"Saved images         : {saved_images}")
    print(f"Label files written  : {wrote_labels}")
    print(f"Skipped zero w/h     : {skipped_zero_wh}")
    print(f"Skipped bad bbox     : {skipped_bad_bbox}")
    print(f"Skipped bad class/map: {skipped_bad_class}")

def main():
    ensure_dir(OUT_ROOT)
    ensure_dir(IMG_OUT_TRAIN); ensure_dir(IMG_OUT_VAL)
    ensure_dir(LBL_OUT_TRAIN); ensure_dir(LBL_OUT_VAL)

    # 매핑 로드
    train_to_dlidx, K = build_trainid_to_dlidx(REMAPPED_ID_JSON)
    dlidx_to_cls = parse_dlidx_to_cls_map(CATEGORY_ID_MAPPING_JSON)

    print("[INFO] K(from remap.json):", K)
    print("[INFO] dl_idx->cls entries:", len(dlidx_to_cls))

    # sanity: dl_idx->cls 중 cls 중복/범위 확인
    cls_set = set(dlidx_to_cls.values())
    print("[INFO] unique cls:", len(cls_set), "min:", min(cls_set) if cls_set else None, "max:", max(cls_set) if cls_set else None)

    convert_split(TRAIN_JSON, IMG_OUT_TRAIN, LBL_OUT_TRAIN, "TRAIN", train_to_dlidx, dlidx_to_cls, K)
    convert_split(VAL_JSON,   IMG_OUT_VAL,   LBL_OUT_VAL,   "VAL",   train_to_dlidx, dlidx_to_cls, K)

    print("\n✅ STEP 7 COMPLETE:", OUT_ROOT)
    print(" - images/train, images/val")
    print(" - labels/train, labels/val")
    print("\n👉 다음: STEP 7-5로 라벨 분포/범위 검증 + 문제 있으면 즉시 탐지")

if __name__ == "__main__":
    main()