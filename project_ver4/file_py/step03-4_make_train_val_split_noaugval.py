import os
import random
import shutil

PROJECT_ROOT = "/home/ohs3201/6team_beginner_project/project_ver4"
YOLO_ROOT = os.path.join(PROJECT_ROOT, "work", "yolo")

IMG_POOL = os.path.join(YOLO_ROOT, "images", "pool")
LBL_POOL = os.path.join(YOLO_ROOT, "labels", "pool")

IMG_TRAIN = os.path.join(YOLO_ROOT, "images", "train")
LBL_TRAIN = os.path.join(YOLO_ROOT, "labels", "train")
IMG_VAL = os.path.join(YOLO_ROOT, "images", "val")
LBL_VAL = os.path.join(YOLO_ROOT, "labels", "val")

for d in [IMG_TRAIN, LBL_TRAIN, IMG_VAL, LBL_VAL]:
    os.makedirs(d, exist_ok=True)

VAL_RATIO = 0.10
SEED = 0
random.seed(SEED)

def is_aug(name: str) -> bool:
    stem = os.path.splitext(name)[0]
    return ("_aug" in stem)

def safe_copy(src, dst):
    if not os.path.exists(dst):
        shutil.copy2(src, dst)

# collect valid samples
imgs = [f for f in os.listdir(IMG_POOL) if f.lower().endswith(".png")]
pairs = []
for fn in imgs:
    lbl = os.path.join(LBL_POOL, os.path.splitext(fn)[0] + ".txt")
    if os.path.exists(lbl):
        pairs.append(fn)

# split only originals into val
orig = [fn for fn in pairs if not is_aug(fn)]
aug = [fn for fn in pairs if is_aug(fn)]

random.shuffle(orig)
val_n = max(1, int(len(orig) * VAL_RATIO))
val_set = set(orig[:val_n])
train_set = set(orig[val_n:]) | set(aug)

print(f"original images: {len(orig)}")
print(f"aug images     : {len(aug)}")
print(f"val images     : {len(val_set)} (orig only)")
print(f"train images   : {len(train_set)}")

# copy
print("[COPY] train...")
for fn in train_set:
    safe_copy(os.path.join(IMG_POOL, fn), os.path.join(IMG_TRAIN, fn))
    safe_copy(os.path.join(LBL_POOL, os.path.splitext(fn)[0] + ".txt"),
              os.path.join(LBL_TRAIN, os.path.splitext(fn)[0] + ".txt"))

print("[COPY] val...")
for fn in val_set:
    safe_copy(os.path.join(IMG_POOL, fn), os.path.join(IMG_VAL, fn))
    safe_copy(os.path.join(LBL_POOL, os.path.splitext(fn)[0] + ".txt"),
              os.path.join(LBL_VAL, os.path.splitext(fn)[0] + ".txt"))

print("\n[DONE] STEP 03-4 split (aug excluded from val)")
print(f"train images: {len(train_set)}")
print(f"val images  : {len(val_set)}")