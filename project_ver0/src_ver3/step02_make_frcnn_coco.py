import os
import json
from pycocotools.coco import COCO
from tqdm import tqdm

BASE = "/home/ohs3201/6team_beginner_project"

COCO_ORI = f"{BASE}/yolo_dataset"
TRAIN_JSON = f"{COCO_ORI}/coco/train.json"
VAL_JSON   = f"{COCO_ORI}/coco/val.json"

OUT_DIR = f"{BASE}/frcnn_dataset_v3"
os.makedirs(OUT_DIR, exist_ok=True)

# category_mapping 로드 (STEP01에서 생성된 것)
with open(os.path.join(BASE, "category_mapping.json"), "r") as f:
    mp = json.load(f)

cat2yolo = {int(k): int(v) for k, v in mp["cat2yolo"].items()}   # orig cid → 0~27
yolo2cat = {int(k): int(v) for k, v in mp["yolo2cat"].items()}   # 0~27 → orig cid
NUM_CLASSES = len(yolo2cat)   # 28


def convert_to_frcnn_json(json_path, split):
    """
    원본 COCO JSON을 읽어서,
    category_id를 cat2yolo 기준으로 재매핑해서 FRCNN용 JSON 생성
    (category id: 0 ~ 27)
    """
    coco = COCO(json_path)

    output = {
        "images": [],
        "annotations": [],
        "categories": [],
    }

    # 카테고리: YOLO 인덱스 순서대로 (0~27)
    for yolo_idx in range(NUM_CLASSES):
        output["categories"].append(
            {
                "id": yolo_idx,           # 0 ~ 27
                "name": f"cls_{yolo_idx}",
                "supercategory": "object",
            }
        )

    ann_id_new = 1

    for img_id in tqdm(coco.getImgIds(), desc=f"FRCNN COCO ({split})"):
        img_info = coco.loadImgs([img_id])[0]

        # 이미지 정보 그대로 사용
        output["images"].append(
            {
                "id": img_info["id"],
                "file_name": img_info["file_name"],
                "width": img_info["width"],
                "height": img_info["height"],
            }
        )

        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            orig_cid = ann["category_id"]

            if orig_cid not in cat2yolo:
                continue

            new_cid = cat2yolo[orig_cid]  # 0~27

            output["annotations"].append(
                {
                    "id": ann_id_new,
                    "image_id": img_info["id"],
                    "category_id": new_cid,
                    "bbox": ann["bbox"],
                    "area": float(
                        ann.get("area", ann["bbox"][2] * ann["bbox"][3])
                    ),
                    "iscrowd": int(ann.get("iscrowd", 0)),
                }
            )
            ann_id_new += 1

    save_path = os.path.join(OUT_DIR, f"{split}.json")
    with open(save_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"📁 FasterRCNN {split}.json 저장 완료 → {save_path}")


def main():
    print("\n========== STEP02: FasterRCNN용 COCO JSON (28클래스) 생성 ==========")
    os.makedirs(OUT_DIR, exist_ok=True)

    convert_to_frcnn_json(TRAIN_JSON, "train")
    convert_to_frcnn_json(VAL_JSON, "val")

    print("🎉 STEP02 완료! frcnn_dataset_v3/train.json & val.json 생성")


if __name__ == "__main__":
    main()