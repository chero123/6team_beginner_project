import os
import json
from tqdm import tqdm
from PIL import Image

# PATH 설정
BASE = "/home/ohs3201/6team_beginner_project"
YOLO_DIR = f"{BASE}/yolo_dataset"
IMG_DIR = f"{YOLO_DIR}/images"
LABEL_DIR = f"{YOLO_DIR}/labels"
OUT_DIR = f"{YOLO_DIR}/coco"
os.makedirs(OUT_DIR, exist_ok=True)

# category mapping 로드
with open(f"{BASE}/category_mapping.json", "r") as f:
    mp = json.load(f)

# YOLO index → 원본 category_id
yolo2cat = {int(k): int(v) for k, v in mp["yolo2cat"].items()}
num_classes = len(yolo2cat)

print(f"📌 총 클래스 수: {num_classes}")
print(f"📌 yolo2cat 샘플: {list(yolo2cat.items())[:5]}")


# COCO 구조 초기화
def coco_init():
    return {
        "images": [],
        "annotations": [],
        "categories": []
    }


# COCO categories 생성
def build_categories():
    cats = []
    for yidx, cat_id in yolo2cat.items():
        cats.append({
            "id": cat_id,   # 원본 category_id 유지
            "name": f"cls_{yidx}",
            "supercategory": "object"
        })
    return cats


# YOLO txt → COCO annotation 변환
def convert_split(split):
    """
    split: 'train' or 'val'
    """
    print(f"\n📌 Split 변환 시작: {split}")

    image_list_path = f"{YOLO_DIR}/{split}.txt"
    if not os.path.exists(image_list_path):
        raise FileNotFoundError(f"❌ {image_list_path} 없음")

    with open(image_list_path, "r") as f:
        image_files = [x.strip() for x in f.readlines()]

    coco = coco_init()
    coco["categories"] = build_categories()

    ann_id = 1

    for img_id, img_name in enumerate(tqdm(image_files)):
        img_path = os.path.join(IMG_DIR, img_name)
        label_path = os.path.join(LABEL_DIR, img_name.replace(".png", ".txt").replace(".jpg", ".txt"))

        if not os.path.exists(img_path):
            print(f"⚠ 이미지 없음: {img_path}")
            continue

        W, H = Image.open(img_path).size

        # COCO images 정보 저장
        coco["images"].append({
            "id": img_id + 1,
            "file_name": img_name,
            "width": W,
            "height": H
        })

        # annotation이 없으면 스킵
        if not os.path.exists(label_path):
            continue

        with open(label_path, "r") as lf:
            lines = lf.readlines()

        for line in lines:
            c, x, y, w, h = line.strip().split()
            c = int(c)  # YOLO class index

            if c not in yolo2cat:
                print(f"⚠ yolo2cat에 없는 클래스 등장: {c}")
                continue

            cat_id = yolo2cat[c]   # ⭐ 원본 category_id로 변환

            # YOLO bbox → COCO bbox 변환
            x, y, w, h = float(x), float(y), float(w), float(h)
            cx, cy = x * W, y * H
            bw, bh = w * W, h * H
            x_min = cx - bw / 2
            y_min = cy - bh / 2

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id + 1,
                "category_id": cat_id,  # ⭐ 반드시 원본 category_id
                "bbox": [x_min, y_min, bw, bh],
                "area": bw * bh,
                "iscrowd": 0
            })
            ann_id += 1

    # COCO JSON 저장
    out_path = f"{OUT_DIR}/{split}.json"
    with open(out_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"✅ COCO {split}.json 생성 완료 → {out_path}")


# 실행
if __name__ == "__main__":
    print("\n🔥 Step01-2: YOLO → COCO 변환 시작")

    convert_split("train")
    convert_split("val")

    print("\n🎉 모든 변환 완료!")