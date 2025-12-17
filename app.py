# unlabeled_images 를 불러와서 모델이 예측하고 예측 못했다면 박스를 직접 그려 클래스 매핑해주는 코드

import os
from pathlib import Path

import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from ultralytics import YOLO

# ---------------------------------------------
# 1) 경로 설정
# ---------------------------------------------
BASE_DIR = Path(r"C:\Users\sangj\workspace\6team_beginner_project")

MODEL_PATH = BASE_DIR / r"runs\detect\train12\weights\best.pt"
UNLABELED_IMG_DIR = BASE_DIR / "unlabeled_images"   # 라벨 없는 이미지 폴더
SAVE_LABEL_DIR = BASE_DIR / "self_labels"           # 라벨 저장 폴더 (YOLO txt)
os.makedirs(SAVE_LABEL_DIR, exist_ok=True)

# ---------------------------------------------
# 2) YOLO 모델 로드 → 클래스 개수 자동 설정
# ---------------------------------------------
model = YOLO(str(MODEL_PATH))
CLASS_NAMES = model.names              # 예: {0: 'pill_0', 1: 'pill_1', ...}
NUM_CLASSES = len(CLASS_NAMES)

st.title("💊 YOLO 예측 + 수동 드로잉 라벨링 툴 (YOLO txt 저장)")
st.write(f"모델 클래스 개수: {NUM_CLASSES}개")
st.write(f"CLASS_NAMES: {CLASS_NAMES}")

# ---------------------------------------------
# 3) Unlabeled 이미지 목록 로드
# ---------------------------------------------
all_images = set()
for ext in ["*.jpg", "*.jpeg", "*.png"]:
    all_images.update(UNLABELED_IMG_DIR.glob(ext))
all_images = sorted(all_images)

if len(all_images) == 0:
    st.error("❌ unlabeled_images 폴더에 이미지가 없습니다.")
    st.stop()

st.write(f"📁 이미지 개수: {len(all_images)}장")

# ---------------------------------------------
# 4) 현재 인덱스 상태 관리
# ---------------------------------------------
if "idx" not in st.session_state:
    st.session_state.idx = 0

idx = st.session_state.idx
idx = max(0, min(idx, len(all_images) - 1))
st.session_state.idx = idx

img_path = all_images[idx]
stem = img_path.stem

st.markdown("---")
st.subheader(f"🖼 이미지 {idx+1}/{len(all_images)} : {img_path.name}")

# ---------------------------------------------
# 5) 이미지 로드 (원본 + 표시용 크기 계산)
# ---------------------------------------------
image = Image.open(img_path).convert("RGB")
orig_w, orig_h = image.size  # 원본 크기

# 화면에서 너무 크게 안 나오게 축소 (가로/세로 최대 800px)
MAX_SIDE = 800
scale = min(MAX_SIDE / orig_w, MAX_SIDE / orig_h, 1.0)  # 1.0 이하만
disp_w = int(orig_w * scale)
disp_h = int(orig_h * scale)
disp_image = image.resize((disp_w, disp_h))

# ---------------------------------------------
# 6) YOLO 예측 (원본 이미지 기준)
# ---------------------------------------------
conf_thres = st.slider("YOLO confidence threshold", 0.0, 1.0, 0.25, 0.05)

results = model.predict(image, conf=conf_thres, verbose=False)[0]

pred_xyxy = []
pred_clses = []
pred_confs = []

if results.boxes is not None and len(results.boxes) > 0:
    pred_xyxy = results.boxes.xyxy.cpu().numpy()   # (N,4), 원본좌표
    pred_clses = results.boxes.cls.cpu().numpy()   # (N,)
    pred_confs = results.boxes.conf.cpu().numpy()  # (N,)

# YOLO 예측 결과 이미지 (참고용)
if len(pred_xyxy) > 0:
    annot_bgr = results.plot()          # BGR numpy
    annot_rgb = annot_bgr[:, :, ::-1]
    annot_pil = Image.fromarray(annot_rgb).resize((disp_w, disp_h))
    st.image(annot_pil, caption="YOLO 예측 결과 (참고용)")
else:
    st.info("⚠ YOLO가 박스를 하나도 찾지 못했습니다. (conf를 낮춰보세요 or 그냥 수동으로 그리기)")

# ---------------------------------------------
# 7) YOLO 예측 박스 리스트 (사용 여부 + 클래스 수정)
# ---------------------------------------------
st.subheader("📌 YOLO 예측 박스 (사용 여부 / 클래스 수정)")

yolo_use_flags = []
yolo_cls_choices = []

for i, (box, cls_id, conf) in enumerate(zip(pred_xyxy, pred_clses, pred_confs)):
    x1, y1, x2, y2 = box
    st.markdown(f"### [YOLO Box {i+1}] conf={conf:.2f}")
    st.write(f"원본 좌표: x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}")

    use = st.checkbox(
        "이 예측 박스 사용하기",
        value=True,
        key=f"yolo_use_{idx}_{i}",
    )
    yolo_use_flags.append(use)

    default_cls = int(cls_id)
    cls_num = st.number_input(
        "클래스 번호 (YOLO 예측 수정 가능)",
        min_value=0,
        value=default_cls,
        step=1,
        key=f"yolo_cls_{idx}_{i}",
    )
    st.caption(f"선택된 클래스: {cls_num} ({CLASS_NAMES.get(int(cls_num), 'unknown')})")

    yolo_cls_choices.append(int(cls_num))
    st.write("---")

# ---------------------------------------------
# 8) Canvas (마우스로 박스 그리기) - 축소된 이미지 사용
# ---------------------------------------------
st.subheader("✏ YOLO가 못 잡은 박스는 여기에서 직접 그리기")
st.caption("이미지 위에서 마우스로 드래그해서 박스를 그리세요 (Rect 모드).")

canvas_result = st_canvas(
    fill_color="rgba(0, 255, 0, 0.2)",   # 박스 내부 색
    stroke_color="#00FF00",              # 박스 테두리 색
    stroke_width=1,                      # 테두리 얇게
    background_image=disp_image,         # 축소된 이미지
    update_streamlit=True,
    height=disp_h,
    width=disp_w,
    drawing_mode="rect",                 # 사각형 그리기 모드
    key=f"canvas_{idx}",
)

# ---------------------------------------------
# 9) Canvas에서 그린 박스 가져오기 (표시용 좌표)
# ---------------------------------------------
manual_boxes_disp = []  # (x1_disp, y1_disp, x2_disp, y2_disp)

if canvas_result.json_data is not None:
    objects = canvas_result.json_data.get("objects", [])
    for obj in objects:
        if obj.get("type") != "rect":
            continue

        left = obj.get("left", 0)
        top = obj.get("top", 0)
        width = obj.get("width", 0)
        height = obj.get("height", 0)

        # scale이 있을 경우 반영
        scale_x_obj = obj.get("scaleX", 1)
        scale_y_obj = obj.get("scaleY", 1)
        width *= scale_x_obj
        height *= scale_y_obj

        x1d = left
        y1d = top
        x2d = left + width
        y2d = top + height

        # 표시용 이미지 크기에 맞게 클램프
        x1d = max(0, min(x1d, disp_w - 1))
        x2d = max(0, min(x2d, disp_w - 1))
        y1d = max(0, min(y1d, disp_h - 1))
        y2d = max(0, min(y2d, disp_h - 1))

        if x2d > x1d and y2d > y1d:
            manual_boxes_disp.append((x1d, y1d, x2d, y2d))

st.write(f"✏ 직접 그린 박스 개수: {len(manual_boxes_disp)}개")

# 수동 박스별 클래스 지정
manual_cls_choices = []

for j, (x1d, y1d, x2d, y2d) in enumerate(manual_boxes_disp):
    st.markdown(f"### [수동 박스 {j+1}]")
    st.write(f"[표시용 좌표] x1={x1d:.1f}, y1={y1d:.1f}, x2={x2d:.1f}, y2={y2d:.1f}")

    default_cls = 0
    cls_num = st.number_input(
        "클래스 번호 (수동 박스)",
        min_value=0,
        value=default_cls,
        step=1,
        key=f"manual_cls_{idx}_{j}",
    )
    st.caption(f"선택된 클래스: {cls_num} ({CLASS_NAMES.get(int(cls_num), 'unknown')})")

    manual_cls_choices.append(int(cls_num))
    st.write("---")

# ---------------------------------------------
# Helper: xyxy → YOLO (cx,cy,w,h) 변환 (원본 크기 기준)
# ---------------------------------------------
def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return cx, cy, bw, bh

# ---------------------------------------------
# 10) 라벨 저장 버튼 (YOLO + 수동 박스 모두 저장)
# ---------------------------------------------
if st.button("💾 이 이미지 YOLO 라벨 저장하기"):
    lines = []

    # 1) YOLO 예측 박스 중 사용 체크된 것만 저장 (원본 좌표 기준)
    for use, cls, box in zip(yolo_use_flags, yolo_cls_choices, pred_xyxy):
        if not use:
            continue
        x1, y1, x2, y2 = box
        cx, cy, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, orig_w, orig_h)
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    # 2) 수동 박스도 저장 (표시용 → 원본 좌표 변환)
    for (cls, (x1d, y1d, x2d, y2d)) in zip(manual_cls_choices, manual_boxes_disp):
        x1 = x1d * orig_w / disp_w
        x2 = x2d * orig_w / disp_w
        y1 = y1d * orig_h / disp_h
        y2 = y2d * orig_h / disp_h

        cx, cy, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, orig_w, orig_h)
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    if len(lines) == 0:
        st.error("⚠ 최소 하나의 박스는 있어야 저장할 수 있습니다.")
    else:
        save_path = SAVE_LABEL_DIR / f"{stem}.txt"
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        st.success(f"저장 완료! → {save_path}")

# ---------------------------------------------
# 11) 이전 / 다음 이미지 이동
# ---------------------------------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("⬅ 이전 이미지"):
        st.session_state.idx = max(0, idx - 1)
        st.rerun()

with col2:
    if st.button("다음 이미지 ➡"):
        st.session_state.idx = min(len(all_images) - 1, idx + 1)
        st.rerun()
