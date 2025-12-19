# file_py/step03-4_make_train_val_split.py
import os, random, shutil

YOLO_ROOT = "/home/ohs3201/6team_beginner_project/project_ver3/work/yolo"

IMG_ALL = os.path.join(YOLO_ROOT, "images/all")
LBL_ALL = os.path.join(YOLO_ROOT, "labels/all")

IMG_TRAIN = os.path.join(YOLO_ROOT, "images/train")
IMG_VAL   = os.path.join(YOLO_ROOT, "images/val")
LBL_TRAIN = os.path.join(YOLO_ROOT, "labels/train")
LBL_VAL   = os.path.join(YOLO_ROOT, "labels/val")

for d in [IMG_TRAIN, IMG_VAL, LBL_TRAIN, LBL_VAL]:
    os.makedirs(d, exist_ok=True)

VAL_RATIO = 0.1
random.seed(42)

# -------------------------
# 1. 원본 / 증강 분리
# -------------------------
orig_imgs = []
aug_imgs = []

for f in os.listdir(IMG_ALL):
    if not f.endswith(".png"):
        continue
    if "_aug" in f:
        aug_imgs.append(f)
    else:
        orig_imgs.append(f)

print(f"original images: {len(orig_imgs)}")
print(f"aug images     : {len(aug_imgs)}")

# -------------------------
# 2. 원본만 train/val split
# -------------------------
random.shuffle(orig_imgs)
n_val = int(len(orig_imgs) * VAL_RATIO)

val_set = set(orig_imgs[:n_val])
train_set = set(orig_imgs[n_val:])

# aug는 전부 train
train_set.update(aug_imgs)

# -------------------------
# 3. copy
# -------------------------
def copy_pair(fname, img_dst, lbl_dst):
    shutil.copy(
        os.path.join(IMG_ALL, fname),
        os.path.join(img_dst, fname)
    )
    shutil.copy(
        os.path.join(LBL_ALL, fname.replace(".png", ".txt")),
        os.path.join(lbl_dst, fname.replace(".png", ".txt"))
    )

print("[COPY] train...")
for f in train_set:
    copy_pair(f, IMG_TRAIN, LBL_TRAIN)

print("[COPY] val...")
for f in val_set:
    copy_pair(f, IMG_VAL, LBL_VAL)

print("\n[DONE] STEP 03-4 split (aug excluded from val)")
print(f"train images: {len(train_set)}")
print(f"val images  : {len(val_set)}")