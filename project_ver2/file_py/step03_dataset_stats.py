import os, json
import csv
from collections import Counter, defaultdict

IN_JSON = "/home/ohs3201/work/step2_clean_coco/clean.json"
OUT_DIR = "/home/ohs3201/work/step3_stats"
os.makedirs(OUT_DIR, exist_ok=True)

data = json.load(open(IN_JSON))

images = data["images"]
annotations = data["annotations"]

# =========================
# 통계 집계
# =========================
cls_counter = Counter()
img_counter = defaultdict(set)

for an in annotations:
    dl = int(an["category_id"])
    cls_counter[dl] += 1
    img_counter[dl].add(an["image_id"])

# =========================
# CSV 출력
# =========================
csv_path = os.path.join(OUT_DIR, "class_distribution.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "dl_idx",
        "num_annotations",
        "num_images"
    ])
    for dl in sorted(cls_counter.keys()):
        writer.writerow([
            dl,
            cls_counter[dl],
            len(img_counter[dl])
        ])

# =========================
# TXT 요약
# =========================
txt_path = os.path.join(OUT_DIR, "class_distribution.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    for dl in sorted(cls_counter.keys()):
        f.write(
            f"dl_idx {dl:>6} | anns {cls_counter[dl]:>5} | imgs {len(img_counter[dl]):>4}\n"
        )

# =========================
# 전체 요약
# =========================
summary_path = os.path.join(OUT_DIR, "summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(f"total_images: {len(images)}\n")
    f.write(f"total_annotations: {len(annotations)}\n")
    f.write(f"unique_classes(dl_idx): {len(cls_counter)}\n")

print("[DONE] STEP 3 stats generated")
print(f"classes: {len(cls_counter)}")
print(f"images: {len(images)}")
print(f"annotations: {len(annotations)}")