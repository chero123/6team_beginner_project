import os
from collections import Counter
from tqdm import tqdm

YOLO_ROOT = "/home/ohs3201/6team_beginner_project/project_ver3/work/yolo"
IMG_DIRS = {
    "train": os.path.join(YOLO_ROOT, "images/train"),
    "val":   os.path.join(YOLO_ROOT, "images/val"),
}
LBL_DIRS = {
    "train": os.path.join(YOLO_ROOT, "labels/train"),
    "val":   os.path.join(YOLO_ROOT, "labels/val"),
}

def check_split(split):
    img_dir = IMG_DIRS[split]
    lbl_dir = LBL_DIRS[split]

    imgs = {os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith(".png")}
    lbls = {os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith(".txt")}

    missing_lbl = imgs - lbls
    missing_img = lbls - imgs

    cls_counter = Counter()
    bad_bbox = 0
    bad_cls = 0

    for f in tqdm(lbls, desc=f"Checking {split} labels"):
        path = os.path.join(lbl_dir, f + ".txt")
        with open(path) as fp:
            for line in fp:
                parts = line.strip().split()
                if len(parts) != 5:
                    bad_bbox += 1
                    continue
                cls, x, y, w, h = parts
                try:
                    cls = int(cls)
                    x, y, w, h = map(float, (x, y, w, h))
                except:
                    bad_bbox += 1
                    continue

                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    bad_bbox += 1
                    continue

                if cls < 0:
                    bad_cls += 1
                    continue

                cls_counter[cls] += 1

    print(f"\n[Sanity {split}]")
    print(f" images: {len(imgs)}")
    print(f" labels: {len(lbls)}")
    print(f" missing labels: {len(missing_lbl)}")
    print(f" missing images: {len(missing_img)}")
    print(f" bad bbox lines: {bad_bbox}")
    print(f" bad cls lines: {bad_cls}")
    print(f" unique classes: {len(cls_counter)}")

    return cls_counter

train_stats = check_split("train")
val_stats   = check_split("val")

print("\n[Top classes in train]")
for cls, cnt in train_stats.most_common(10):
    print(f" cls {cls}: {cnt}")