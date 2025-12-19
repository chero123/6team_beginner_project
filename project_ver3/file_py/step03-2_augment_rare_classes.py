import os
import shutil
from collections import Counter

# =========================
# PATH
# =========================
YOLO_ROOT = "/home/ohs3201/6team_beginner_project/project_ver3/work/yolo"

IMG_IN = os.path.join(YOLO_ROOT, "images/all")
LBL_IN = os.path.join(YOLO_ROOT, "labels/all")

IMG_OUT = os.path.join(YOLO_ROOT, "images/aug")
LBL_OUT = os.path.join(YOLO_ROOT, "labels/aug")

os.makedirs(IMG_OUT, exist_ok=True)
os.makedirs(LBL_OUT, exist_ok=True)

# =========================
# CONFIG
# =========================
MIN_IMAGES = 300     # 👈 이보다 적은 클래스만 증강
OS_FACTOR = 3        # 몇 배로 늘릴지

# =========================
# COUNT PER CLASS
# =========================
cls_imgs = Counter()
img_cls = {}

for f in os.listdir(LBL_IN):
    if not f.endswith(".txt"):
        continue
    with open(os.path.join(LBL_IN, f)) as fp:
        clss = {int(l.split()[0]) for l in fp if l.strip()}
    img_cls[f] = clss
    for c in clss:
        cls_imgs[c] += 1

rare_classes = {c for c, n in cls_imgs.items() if n < MIN_IMAGES}
print("[INFO] rare classes:", sorted(rare_classes))

# =========================
# OVERSAMPLE
# =========================
for lbl, clss in img_cls.items():
    if not (clss & rare_classes):
        continue

    base = lbl.replace(".txt", "")
    img = base + ".png"

    for i in range(OS_FACTOR):
        new_base = f"{base}_aug{i}"
        shutil.copy(
            os.path.join(IMG_IN, img),
            os.path.join(IMG_OUT, new_base + ".png")
        )
        shutil.copy(
            os.path.join(LBL_IN, lbl),
            os.path.join(LBL_OUT, new_base + ".txt")
        )

print("[DONE] STEP 03-2 augmentation completed")