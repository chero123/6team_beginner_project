from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dataset import PillDataset  # 위에서 만든 dataset.py


# -------------------------------------------------------
# 0) 디바이스 선택 (M1 GPU(MPS) 우선 사용)
# -------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🔥 Using Apple GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("🔥 Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("⚠️ Using CPU")


# -------------------------------------------------------
# 1) 모델 생성 함수
# -------------------------------------------------------
def get_detection_model(num_classes: int):
    """
    num_classes: 전체 클래스 개수 (배경 포함)
    """

    # 가능하면 최신 모델 사용
    try:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights="DEFAULT"
        )
    except:
        # fallback
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights="DEFAULT"
        )

    # Head 교체 (클래스 수 맞추기)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# -------------------------------------------------------
# 2) 데이터로더
# -------------------------------------------------------
def collate_fn(batch):
    return tuple(zip(*batch))


def get_loaders(root, batch_size=2, num_workers=0):
    # M1 메모리 고려해서 batch_size=2 추천
    train_dataset = PillDataset(root=root, split="train")
    val_dataset   = PillDataset(root=root, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


# -------------------------------------------------------
# 3) 학습 메인 함수
# -------------------------------------------------------
def train():
    print("✨ 사용 디바이스:", device)

    # 네 데이터 경로
    root = "/Users/apple/Downloads/프로젝트1/ai06-level1-project"

    train_loader, val_loader = get_loaders(
        root=root,
        batch_size=2,   # M1이면 2 정도가 안전
        num_workers=0
    )

    # 🔥🔥 여기만 네 클래스 개수에 맞게 수정하면 됨!!
    NUM_CLASSES = 1 + 56   # 배경 1 + pill class 56개

    # 모델 불러오기
    model = get_detection_model(NUM_CLASSES)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005,
    )

    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    num_epochs = 10

    # -------------------------------------------------------
    # 🔥 학습 루프
    # -------------------------------------------------------
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for images, targets in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            ncols=100
        ):
            # 디바이스로 이동
            images = [img.to(device) for img in images]
            targets = [
                {k: v.to(device) for k, v in t.items()}
                for t in targets
            ]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()

        print(f"[{epoch+1}/{num_epochs}] 🔥 train loss: {epoch_loss:.4f}")

        # -------------------------------------------------------
        # 간단한 validation (1개 배치만)
        # -------------------------------------------------------
        model.eval()
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                outputs = model(images)

                print("📌 예측 boxes:", outputs[0]["boxes"].shape)
                print("📌 예측 labels 샘플:", outputs[0]["labels"][:5])
                break  # 한 배치만 확인
        model.train()

    # -------------------------------------------------------
    # 모델 저장
    # -------------------------------------------------------
    torch.save(model.state_dict(), "fasterrcnn_pill_m1.pth")
    print("✅ 모델 저장 완료 → fasterrcnn_pill_m1.pth")


# -------------------------------------------------------
# 4) 실행
# -------------------------------------------------------
if __name__ == "__main__":
    train()
