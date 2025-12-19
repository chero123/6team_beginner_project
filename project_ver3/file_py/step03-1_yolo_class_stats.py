import os
import csv
from collections import Counter, defaultdict

# =========================
# PATH
# =========================
YOLO_ROOT = "/home/ohs3201/6team_beginner_project/project_ver3/work/yolo"
LBL_DIR = os.path.join(YOLO_ROOT, "labels/all")

OUT_DIR = "/home/ohs3201/6team_beginner_project/project_ver3/step03_stats"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# COLLECT STATS
# =========================
ann_counter = Counter()
img_counter = defaultdict(set)

for txt in os.listdir(LBL_DIR):
    if not txt.endswith(".txt"):
        continue

    path = os.path.join(LBL_DIR, txt)
    with open(path) as f:
        used = set()
        for line in f:
            cls = int(line.split()[0])
            ann_counter[cls] += 1
            used.add(cls)

        for cls in used:
            img_counter[cls].add(txt)

# =========================
# SAVE CSV
# =========================
csv_path = os.path.join(OUT_DIR, "class_distribution.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["train_id", "num_annotations", "num_images"])
    for cls in sorted(ann_counter):
        writer.writerow([
            cls,
            ann_counter[cls],
            len(img_counter[cls])
        ])

# =========================
# SAVE TXT SUMMARY
# =========================
txt_path = os.path.join(OUT_DIR, "summary.txt")
with open(txt_path, "w") as f:
    for cls in sorted(ann_counter):
        f.write(
            f"cls {cls:>3} | anns {ann_counter[cls]:>5} | imgs {len(img_counter[cls]):>4}\n"
        )

print("[DONE] STEP 03-1 class stats generated")
print(f" - saved to {OUT_DIR}")