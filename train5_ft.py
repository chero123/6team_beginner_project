from ultralytics import YOLO
import torch

def main():
    # M1인지 확인 (윈도우면 cuda)
    if torch.cuda.is_available():
        device_str = "cuda"
        print("🔥 Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device_str = "mps"
        print("🔥 Using Apple M1 GPU (MPS)")
    else:
        device_str = "cpu"
        print("⚠️ GPU 없음, CPU 사용")

    # 기존 best.pt 로드해서 파인튜닝 시작
    model = YOLO("runs/detect/train5/weights/best.pt")

    model.train(
        data="data.yml",
        epochs=30,        # 파인튜닝은 20~30 정도면 충분
        batch=16,
        imgsz=640,
        lr0=0.0005,       # 🔥 파인튜닝 핵심: lr 줄이기
        patience=8,
        device=device_str,
        project="runs/detect",
        name="train5_ft",
        workers=0,        # 🔥 윈도우에서는 0으로 두면 안전함
    )

if __name__ == "__main__":   # 🔥 이게 중요!
    main()
