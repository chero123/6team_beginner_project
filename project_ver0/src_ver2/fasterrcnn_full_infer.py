import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image
import pandas as pd
from tqdm import tqdm

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_frcnn(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    return model


def infer_fasterrcnn(weight_path, csv_path, test_dir, num_classes):

    print(f"🔄 FasterRCNN weight 로드: {weight_path}")

    # 1) build model with correct num_classes
    model = build_frcnn(num_classes)
    state = torch.load(weight_path, map_location=DEVICE)

    # classifier head는 checkpoint와 동일해야 함 → strict=True 가능
    model.load_state_dict(state, strict=True)

    model.to(DEVICE)
    model.eval()

    results = []

    img_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".png")])

    for img_name in tqdm(img_files, desc="FRCNN Inference"):

        img_id = int(img_name.replace(".png", ""))
        img_path = os.path.join(test_dir, img_name)

        img = Image.open(img_path).convert("RGB")
        tensor = F.to_tensor(img).to(DEVICE)

        out = model([tensor])[0]

        for box, label, score in zip(out["boxes"], out["labels"], out["scores"]):

            if score < 0.01:
                continue

            x1, y1, x2, y2 = box.tolist()
            w = x2 - x1
            h = y2 - y1

            results.append([
                img_id,
                int(label),  # 이미 original category id로 훈련된 모델
                int(x1), int(y1), int(w), int(h),
                float(score)
            ])

    df = pd.DataFrame(results, columns=[
        "image_id", "category_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"
    ])

    df.to_csv(csv_path, index=False)
    print(f"💾 FasterRCNN inference 저장 완료 → {csv_path}")