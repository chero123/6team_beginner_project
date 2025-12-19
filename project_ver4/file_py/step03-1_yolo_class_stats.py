import os
from collections import Counter, defaultdict
import csv

PROJECT_ROOT = "/home/ohs3201/6team_beginner_project/project_ver4"
YOLO_ROOT = os.path.join(PROJECT_ROOT, "work", "yolo")
LBL_ALL = os.path.join(YOLO_ROOT, "labels", "all")

OUT_DIR = os.path.join(PROJECT_ROOT, "step03_stats")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_CSV = os.path.join(OUT_DIR, "class_distribution_trainid.csv")
OUT_TXT = os.path.join(OUT_DIR, "summary.txt")

cnt = Counter()
img_cnt = Counter()
bad_lines = 0
total_lines = 0
files = [f for f in os.listdir(LBL_ALL) if f.endswith(".txt")]

for f in files:
    p = os.path.join(LBL_ALL, f)
    seen = set()
    with open(p, "r", encoding="utf-8") as fp:
        for ln in fp:
            ln = ln.strip()
            if not ln:
                continue
            total_lines += 1
            parts = ln.split()
            if len(parts) != 5:
                bad_lines += 1
                continue
            try:
                cls = int(parts[0])
                vals = list(map(float, parts[1:]))
            except:
                bad_lines += 1
                continue

            # YOLO norm bbox sanity
            cx, cy, w, h = vals
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                bad_lines += 1
                continue

            cnt[cls] += 1
            seen.add(cls)

    for c in seen:
        img_cnt[c] += 1

# save csv
rows = []
for cls in sorted(cnt.keys()):
    rows.append({
        "cls_trainid": cls,
        "instances": cnt[cls],
        "images_with_cls": img_cnt[cls],
    })

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["cls_trainid", "instances", "images_with_cls"])
    w.writeheader()
    w.writerows(rows)

# summary
unique_classes = len(cnt)
total_instances = sum(cnt.values())
min_cls = min(cnt.items(), key=lambda x: x[1])[0] if cnt else None
max_cls = max(cnt.items(), key=lambda x: x[1])[0] if cnt else None

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write(f"labels: {len(files)}\n")
    f.write(f"total_instances: {total_instances}\n")
    f.write(f"unique_classes: {unique_classes}\n")
    f.write(f"bad_lines: {bad_lines} / total_lines: {total_lines}\n")
    if cnt:
        f.write(f"min_cls: {min_cls} ({cnt[min_cls]})\n")
        f.write(f"max_cls: {max_cls} ({cnt[max_cls]})\n")

print("[DONE] STEP 03-1 class stats generated")
print(f" - saved to {OUT_DIR}")