import os
import json
import shutil
from pycocotools.coco import COCO
from tqdm import tqdm

BASE = "/home/ohs3201/6team_beginner_project"

COCO_DIR = f"{BASE}/yolo_dataset"          # 원본 COCO json
TRAIN_JSON = f"{COCO_DIR}/coco/train.json"
VAL_JSON   = f"{COCO_DIR}/coco/val.json"

OUT_DIR = f"{BASE}/yolo_dataset_v3"        # YOLO용 출력 폴더
IMG_TRAIN_DIR = f"{OUT_DIR}/images/train"
IMG_VAL_DIR   = f"{OUT_DIR}/images/val"
LBL_TRAIN_DIR = f"{OUT_DIR}/labels/train"
LBL_VAL_DIR   = f"{OUT_DIR}/labels/val"

TRAIN_IMG_SRC = f"{BASE}/data/train_images"   # 원본 이미지 위치 (train/val 모두 여기서 가져온다고 가정)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def build_category_mapping(train_json, val_json):
    """
    COCO의 실제 annotation에 등장하는 category_id만 모아서
    정렬 후 0 ~ N-1로 매핑을 만든다. (이 결과가 28개가 되어야 함)
    """
    coco_train = COCO(train_json)
    coco_val   = COCO(val_json)

    train_cats = {ann["category_id"] for ann in coco_train.dataset["annotations"]}
    val_cats   = {ann["category_id"] for ann in coco_val.dataset["annotations"]}

    used_categories = sorted(list(train_cats | val_cats))  # 예: [1, 6, 8, ..., 55]

    cat2yolo = {cid: idx for idx, cid in enumerate(used_categories)}   # orig → yolo idx(0~27)
    yolo2cat = {idx: cid for idx, cid in enumerate(used_categories)}   # yolo idx → orig

    print(f"📌 실제 사용되는 클래스 수: {len(used_categories)} (기대: 28)")
    print("➡ category_id 목록:", used_categories)

    return cat2yolo, yolo2cat


def convert_coco_to_yolo(json_path, split, cat2yolo):
    """
    COCO 어노테이션을 YOLO 형식(.txt)으로 변환
    - 이미지: yolo_dataset_v3/images/{train,val}/
    - 라벨:   yolo_dataset_v3/labels/{train,val}/
    """
    coco = COCO(json_path)

    if split == "train":
        out_img_dir = IMG_TRAIN_DIR
        out_lbl_dir = LBL_TRAIN_DIR
    else:
        out_img_dir = IMG_VAL_DIR
        out_lbl_dir = LBL_VAL_DIR

    ensure_dir(out_img_dir)
    ensure_dir(out_lbl_dir)

    img_ids = coco.getImgIds()

    for img_id in tqdm(img_ids, desc=f"COCO→YOLO ({split})"):
        img_info = coco.loadImgs([img_id])[0]
        file_name = img_info["file_name"]     # 예: K-xxxx.png 또는 .jpg

        # 1. 이미지 복사
        src_path = os.path.join(TRAIN_IMG_SRC, file_name)
        dst_path = os.path.join(out_img_dir, file_name)

        if not os.path.exists(src_path):
            print(f"⚠️ WARNING: 이미지 없음: {src_path}")
        else:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)

        # 2. 라벨 작성 (YOLO txt)
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)

        h, w = img_info["height"], img_info["width"]

        stem, _ = os.path.splitext(file_name)
        label_path = os.path.join(out_lbl_dir, f"{stem}.txt")

        with open(label_path, "w") as f:
            for ann in anns:
                orig_cid = ann["category_id"]
                if orig_cid not in cat2yolo:
                    continue

                yolo_cid = cat2yolo[orig_cid]

                x, y, bw, bh = ann["bbox"]  # COCO: x,y,w,h (좌상단, 폭/높이)
                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                nw = bw / w
                nh = bh / h

                # YOLO 형식: class cx cy w h
                f.write(f"{yolo_cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")


def main():
    print("\n========== STEP01: YOLO Dataset v3 생성 (28 클래스 매핑) ==========")
    ensure_dir(OUT_DIR)

    # 1) category_mapping 생성
    cat2yolo, yolo2cat = build_category_mapping(TRAIN_JSON, VAL_JSON)

    mapping_path = os.path.join(BASE, "category_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(
            {
                "cat2yolo": {str(k): int(v) for k, v in cat2yolo.items()},
                "yolo2cat": {str(k): int(v) for k, v in yolo2cat.items()},
            },
            f,
            indent=4,
        )

    print(f"📁 category_mapping.json 저장 완료 → {mapping_path}")

    # 2) COCO → YOLO 변환
    convert_coco_to_yolo(TRAIN_JSON, "train", cat2yolo)
    convert_coco_to_yolo(VAL_JSON, "val", cat2yolo)

    print("🎉 STEP01 완료! YOLO 학습용 데이터셋(yolo_dataset_v3) 준비 완료")


if __name__ == "__main__":
    main()