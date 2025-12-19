# STEP 03-3 (FIXED)
# all + aug → all (pool 확장)
import os
import shutil

YOLO_ROOT = "/home/ohs3201/6team_beginner_project/project_ver3/work/yolo"

IMG_ALL = os.path.join(YOLO_ROOT, "images/all")
LBL_ALL = os.path.join(YOLO_ROOT, "labels/all")

IMG_AUG = os.path.join(YOLO_ROOT, "images/aug")
LBL_AUG = os.path.join(YOLO_ROOT, "labels/aug")

img_added = 0
lbl_added = 0

for f in os.listdir(IMG_AUG):
    src_img = os.path.join(IMG_AUG, f)
    dst_img = os.path.join(IMG_ALL, f)

    src_lbl = os.path.join(LBL_AUG, f.replace(".png", ".txt"))
    dst_lbl = os.path.join(LBL_ALL, f.replace(".png", ".txt"))

    if not os.path.exists(dst_img):
        shutil.copy(src_img, dst_img)
        img_added += 1

    if os.path.exists(src_lbl) and not os.path.exists(dst_lbl):
        shutil.copy(src_lbl, dst_lbl)
        lbl_added += 1

print("[DONE] STEP 03-3 (pool merge)")
print(f" - images added to all: {img_added}")
print(f" - labels added to all: {lbl_added}")