import os

PROJECT_ROOT = "/home/ohs3201/6team_beginner_project/project_ver4"
YOLO_ROOT = os.path.join(PROJECT_ROOT, "work", "yolo")

def check_split(split):
    img_dir = os.path.join(YOLO_ROOT, "images", split)
    lbl_dir = os.path.join(YOLO_ROOT, "labels", split)

    imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(".png")]
    lbls = [f for f in os.listdir(lbl_dir) if f.lower().endswith(".txt")]

    img_set = set(os.path.splitext(f)[0] for f in imgs)
    lbl_set = set(os.path.splitext(f)[0] for f in lbls)

    missing_labels = sorted(list(img_set - lbl_set))
    missing_images = sorted(list(lbl_set - img_set))

    bad_bbox = 0
    bad_cls = 0
    uniq = set()

    for stem in lbl_set:
        lp = os.path.join(lbl_dir, stem + ".txt")
        with open(lp, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split()
                if len(parts) != 5:
                    bad_bbox += 1
                    continue
                try:
                    cls = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:])
                except:
                    bad_bbox += 1
                    continue
                if cls < 0:
                    bad_cls += 1
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    bad_bbox += 1
                uniq.add(cls)

    print(f"\n[Sanity {split}]")
    print(f" images: {len(imgs)}")
    print(f" labels: {len(lbls)}")
    print(f" missing labels: {len(missing_labels)}")
    print(f" missing images: {len(missing_images)}")
    print(f" bad bbox lines: {bad_bbox}")
    print(f" bad cls lines: {bad_cls}")
    print(f" unique classes: {len(uniq)}")
    return uniq

u1 = check_split("train")
u2 = check_split("val")