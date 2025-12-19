import yaml
import os

BASE = "/home/ohs3201/work/step4_yolov8"
OUT_YAML = os.path.join(BASE, "data_oversample.yaml")

data = {
    "path": BASE,
    "train": "images/train_rare",   # 🔥 5-2에서 만든 oversample
    "val": "images/val",
    "nc": 62,
    "names": [f"class_{i}" for i in range(62)],
}

with open(OUT_YAML, "w") as f:
    yaml.dump(data, f, sort_keys=False)

print("[DONE] data_oversample.yaml created")
print("Saved to:", OUT_YAML)