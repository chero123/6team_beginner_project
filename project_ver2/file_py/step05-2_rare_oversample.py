import os
import shutil
from collections import Counter

# =========================
# PATH
# =========================
YOLO_ROOT = "/home/ohs3201/work/step4_yolov8"
IMG_IN = os.path.join(YOLO_ROOT, "images/train")
LBL_IN = os.path.join(YOLO_ROOT, "labels/train")

IMG_OUT = os.path.join(YOLO_ROOT, "images/train_rare")
LBL_OUT = os.path.join(YOLO_ROOT, "labels/train_rare")

os.makedirs(IMG_OUT, exist_ok=True)
os.makedirs(LBL_OUT, exist_ok=True)

# =========================
# CONFIG
# =========================
RARE_THRES = 10      # annotation < 10
OS_FACTOR = 3        # 3배 반복

# =========================
# 1. class stats
# =========================
cls_cnt = Counter()
label_classes = {}

for lf in os.listdir(LBL_IN):
    with open(os.path.join(LBL_IN, lf)) as f:
        clss = {int(l.split()[0]) for l in f if l.strip()}
    label_classes[lf] = clss
    cls_cnt.update(clss)

rare_classes = {c for c, n in cls_cnt.items() if n < RARE_THRES}
print("[INFO] Rare classes:", sorted(rare_classes))

# =========================
# 2. copy originals
# =========================
for img in os.listdir(IMG_IN):
    shutil.copy(os.path.join(IMG_IN, img), os.path.join(IMG_OUT, img))
    shutil.copy(
        os.path.join(LBL_IN, img.replace(".png", ".txt")),
        os.path.join(LBL_OUT, img.replace(".png", ".txt"))
    )

# =========================
# 3. oversample
# =========================
for lf, clss in label_classes.items():
    if not (clss & rare_classes):
        continue

    base = lf.replace(".txt", "")
    for i in range(OS_FACTOR - 1):
        new = f"{base}_os{i}"
        shutil.copy(
            os.path.join(IMG_IN, base + ".png"),
            os.path.join(IMG_OUT, new + ".png")
        )
        shutil.copy(
            os.path.join(LBL_IN, lf),
            os.path.join(LBL_OUT, new + ".txt")
        )

print("[DONE] STEP 5-2 rare oversampling done")