import os
import csv
import torch
from tqdm import tqdm
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

IMG_ROOT = "/home/ohs3201/work/step2_unified_coco/images"
VAL_JSON = "/home/ohs3201/work/step4_runs_remap/val_remap.json"
CKPT_DIR = "/home/ohs3201/work/step5_sampler_runs"
OUT_CSV  = "/home/ohs3201/work/step6_eval_summary.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CocoDS(CocoDetection):
    def __getitem__(self, idx):
        img, _ = super().__getitem__(idx)
        return F.to_tensor(img), self.ids[idx]

ds = CocoDS(IMG_ROOT, VAL_JSON)
loader = torch.utils.data.DataLoader(ds, batch_size=1)

coco_gt = COCO(VAL_JSON)

rows = []

for ckpt in sorted(os.listdir(CKPT_DIR)):
    if not ckpt.endswith(".pth"):
        continue

    print("Evaluating", ckpt)

    model = fasterrcnn_resnet50_fpn(
        num_classes=len(coco_gt.cats) + 1
    )
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, ckpt)))
    model.to(DEVICE)
    model.eval()

    results = []

    with torch.no_grad():
        for imgs, img_ids in tqdm(loader):
            imgs = [i.to(DEVICE) for i in imgs]
            outs = model(imgs)

            for o, img_id in zip(outs, img_ids):
                for b, s, l in zip(
                    o["boxes"], o["scores"], o["labels"]
                ):
                    if s < 0.05:
                        continue
                    x1, y1, x2, y2 = b.tolist()
                    results.append({
                        "image_id": int(img_id),
                        "category_id": int(l),
                        "bbox": [x1, y1, x2-x1, y2-y1],
                        "score": float(s),
                    })

    coco_dt = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    rows.append([
        ckpt,
        ev.stats[0],  # mAP
        ev.stats[1],  # mAP50
        ev.stats[6],  # AR
    ])

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["checkpoint", "mAP", "mAP50", "AR"])
    writer.writerows(rows)

print("✅ STEP 6 DONE:", OUT_CSV)