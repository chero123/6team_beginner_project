from ultralytics import YOLO

model = YOLO("runs/detect/train3/weights/best.pt")

# 테스트할 이미지 경로
img_path = "test_images/1.png"   # ← 수정해줘

results = model.predict(
    source=img_path,
    save=True,         # 예측 결과를 runs/predict 폴더에 저장
    imgsz=640,
    conf=0.25
)

print("완료! 결과는 runs/detect3/predict 안에 저장됨.")
