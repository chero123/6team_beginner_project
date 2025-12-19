import json
import os

YOLO_ROOT = "/home/ohs3201/6team_beginner_project/project_ver3/work/yolo"
MAP_JSON = "/home/ohs3201/6team_beginner_project/project_ver3/mappings/trainid_to_dlidx.json"

DATA_YAML = os.path.join(YOLO_ROOT, "data.yaml")

# =========================
# load mapping
# =========================
with open(MAP_JSON, "r", encoding="utf-8") as f:
    trainid_to_dlidx = json.load(f)

nc = len(trainid_to_dlidx)

# =========================
# write data.yaml
# =========================
lines = []
lines.append(f"path: {YOLO_ROOT}")
lines.append("train: images/train")
lines.append("val: images/val")
lines.append("")
lines.append(f"nc: {nc}")
lines.append("names:")

for i in range(nc):
    dl = trainid_to_dlidx[str(i)]
    lines.append(f"  - dl_{dl}")

with open(DATA_YAML, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("[DONE] data.yaml created")
print(f" - path: {DATA_YAML}")
print(f" - nc: {nc}")