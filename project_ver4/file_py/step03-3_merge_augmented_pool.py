import os
import shutil

PROJECT_ROOT = "/home/ohs3201/6team_beginner_project/project_ver4"
YOLO_ROOT = os.path.join(PROJECT_ROOT, "work", "yolo")

IMG_ALL = os.path.join(YOLO_ROOT, "images", "all")
LBL_ALL = os.path.join(YOLO_ROOT, "labels", "all")

IMG_AUG = os.path.join(YOLO_ROOT, "images", "aug")
LBL_AUG = os.path.join(YOLO_ROOT, "labels", "aug")

IMG_POOL = os.path.join(YOLO_ROOT, "images", "pool")
LBL_POOL = os.path.join(YOLO_ROOT, "labels", "pool")

os.makedirs(IMG_POOL, exist_ok=True)
os.makedirs(LBL_POOL, exist_ok=True)

def safe_copy(src, dst):
    if not os.path.exists(dst):
        shutil.copy2(src, dst)
        return 1
    return 0

added_img = 0
added_lbl = 0

# 1) copy all
for fn in os.listdir(IMG_ALL):
    if not fn.lower().endswith(".png"):
        continue
    src_img = os.path.join(IMG_ALL, fn)
    src_lbl = os.path.join(LBL_ALL, os.path.splitext(fn)[0] + ".txt")
    if not os.path.exists(src_lbl):
        continue
    added_img += safe_copy(src_img, os.path.join(IMG_POOL, fn))
    added_lbl += safe_copy(src_lbl, os.path.join(LBL_POOL, os.path.splitext(fn)[0] + ".txt"))

# 2) copy aug
if os.path.exists(IMG_AUG) and os.path.exists(LBL_AUG):
    for fn in os.listdir(IMG_AUG):
        if not fn.lower().endswith(".png"):
            continue
        src_img = os.path.join(IMG_AUG, fn)
        src_lbl = os.path.join(LBL_AUG, os.path.splitext(fn)[0] + ".txt")
        if not os.path.exists(src_lbl):
            continue
        added_img += safe_copy(src_img, os.path.join(IMG_POOL, fn))
        added_lbl += safe_copy(src_lbl, os.path.join(LBL_POOL, os.path.splitext(fn)[0] + ".txt"))

print("[DONE] STEP 03-3 (pool merge)")
print(f" - images added to pool: {added_img}")
print(f" - labels added to pool: {added_lbl}")
print(f" - pool dirs: {IMG_POOL}, {LBL_POOL}")