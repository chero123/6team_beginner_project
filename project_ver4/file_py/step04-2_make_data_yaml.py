import os
import json

PROJECT_ROOT = "/home/ohs3201/6team_beginner_project/project_ver4"
YOLO_ROOT = os.path.join(PROJECT_ROOT, "work", "yolo")
MAP_T2DL = os.path.join(PROJECT_ROOT, "mappings", "trainid_to_dlidx.json")

DATA_YAML = os.path.join(YOLO_ROOT, "data.yaml")

# class count = mapping 기준
with open(MAP_T2DL, "r", encoding="utf-8") as f:
    t2dl = json.load(f)
nc = len(t2dl)

# names는 train_id 인덱스 기준으로 “dl_{dlidx}” 형태
names = []
for i in range(nc):
    dl = t2dl[str(i)]
    names.append(f"dl_{dl}")

txt = []
txt.append(f"path: {YOLO_ROOT}")
txt.append("train: images/train")
txt.append("val: images/val")
txt.append(f"nc: {nc}")
txt.append("names:")
for i, n in enumerate(names):
    txt.append(f"  {i}: {n}")

with open(DATA_YAML, "w", encoding="utf-8") as f:
    f.write("\n".join(txt) + "\n")

print("[DONE] data.yaml created")
print(" -", DATA_YAML)
print(" - nc =", nc)