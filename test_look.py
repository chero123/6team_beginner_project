#모델이 예측한 테스트 데이터를 클래스별로 시각화하는 streamlit

import os
from pathlib import Path
from collections import defaultdict

import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ---------------------------------------------
# 1) 경로 설정 (환경에 맞게 확인)
# ---------------------------------------------
BASE_DIR = Path(r"C:\Users\sangj\workspace\6team_beginner_project")

TEST_DIR = BASE_DIR / "data_ai06" / "test_images"  # ✅ Kaggle test 이미지 폴더
MODEL_PATH = BASE_DIR / r"runs\detect\train17\weights\best.pt"  # 쓰고 싶은 모델 경로

# ---------------------------------------------
# 2) YOLO 모델 로드
# ---------------------------------------------
model = YOLO(str(MODEL_PATH))
CLASS_NAMES = model.names          # 예: {0:'pill_0', 1:'pill_1', ...}

st.title("💊 테스트 약 분석기 (클래스별 자신 있는 약 모아보기)")

st.write(f"모델 경로: `{MODEL_PATH}`")
st.write(f"클래스 개수: **{len(CLASS_NAMES)}개**")
st.write(f"CLASS_NAMES: {CLASS_NAMES}")

# ---------------------------------------------
# 3) 테스트 이미지 목록 불러오기
# ---------------------------------------------
all_images = []
for ext in ("*.png", "*.jpg", "*.jpeg"):
    all_images.extend(TEST_DIR.glob(ext))

all_images = sorted(all_images)

if not all_images:
    st.error(f"❌ 테스트 폴더에 이미지가 없습니다: {TEST_DIR}")
    st.stop()

st.write(f"📁 테스트 이미지 개수: **{len(all_images)}장**")

# ---------------------------------------------
# 4) 사이드바 - conf 기준 설정
# ---------------------------------------------
st.sidebar.header("⚙ 분석 옵션")

high_conf_th = st.sidebar.slider(
    "자신 있는 예측 기준(conf)",
    min_value=0.0,
    max_value=1.0,
    value=0.7,      # 여기 0.8~0.9 정도로 올리면 더 빡세게 필터링됨
    step=0.05,
)

st.sidebar.write("※ 이 값 이상인 약만 '자신 있는 예측'으로 간주합니다.")

# ---------------------------------------------
# 5) 전체 테스트 이미지에서 약(박스) 단위로 예측 수행
#    - 앱 켜지면 자동으로 한 번 실행
# ---------------------------------------------
def run_full_detection():
    """
    모든 테스트 이미지에서 YOLO로 예측하고,
    약(박스) 하나당 하나의 dict로 저장.
    """
    detections = []  # [{img_path, bbox, conf, cls}, ...]

    prog = st.progress(0.0, text="모든 테스트 이미지에서 약 탐지 중...")

    for i, img_path in enumerate(all_images):
        img = Image.open(img_path).convert("RGB")

        # conf 기준은 아주 낮게(0.05) 잡아서 웬만하면 다 받는다.
        # 실제 '자신 있는 약' 여부는 나중에 high_conf_th로 나눌 거라 여기선 약하게.
        res = model.predict(img, conf=0.05, verbose=False)[0]

        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()  # (N, 4) - x1,y1,x2,y2
            confs = res.boxes.conf.cpu().numpy()  # (N,)
            clses = res.boxes.cls.cpu().numpy()   # (N,)

            for box, conf, cls in zip(boxes, confs, clses):
                x1, y1, x2, y2 = [int(v) for v in box]
                detections.append(
                    {
                        "img_path": img_path,
                        "bbox": (x1, y1, x2, y2),
                        "conf": float(conf),
                        "cls": int(cls),
                    }
                )
        # 박스가 하나도 없는 이미지는 여기선 별도 저장 안 하고 넘어감

        prog.progress((i + 1) / len(all_images))

    st.session_state.detections = detections
    st.success(f"✅ 전체 탐지 완료! 총 **{len(detections)}개** 약 박스 탐지")


# 앱 처음 켜질 때 한 번만 돌리기
if "detections" not in st.session_state:
    run_full_detection()

detections = st.session_state.detections

st.markdown("---")

# ---------------------------------------------
# 6) conf 기준으로 '자신 있는 약' vs '애매한 약' 나누기
#    - 약(박스) 단위로 나눈다!
# ---------------------------------------------
confident_pills = [
    d for d in detections if d["conf"] >= high_conf_th
]

uncertain_pills = [
    d for d in detections if d["conf"] < high_conf_th
]

st.markdown("## 📊 약(박스) 단위 요약")

st.write(
    f"✔ **자신 있게 예측한 약(박스)** (conf ≥ {high_conf_th:.2f}): "
    f"**{len(confident_pills)}개**"
)
st.write(
    f"⚠ **애매하게 예측한 약(박스)** (conf < {high_conf_th:.2f}): "
    f"**{len(uncertain_pills)}개**"
)
st.markdown("---")

# ---------------------------------------------
# 7) 자신 있는 약: 클래스별로 묶어서 보기
#    - 각 약은 원본에서 잘라낸 crop으로 보여줌
# ---------------------------------------------
st.markdown("### ✅ 자신 있는 약 — 클래스별 그룹")

by_class = defaultdict(list)
for d in confident_pills:
    by_class[d["cls"]].append(d)

if not by_class:
    st.info("conf 기준이 너무 높아서 자신 있는 약이 없습니다. 슬라이더를 조금 낮춰보세요.")
else:
    for cls_idx, items in sorted(by_class.items(), key=lambda x: x[0]):
        cls_name = CLASS_NAMES.get(int(cls_idx), f"pill_{cls_idx}")
        with st.expander(f"Class {cls_idx} - {cls_name}  ({len(items)}개 약)", expanded=False):
            cols = st.columns(5)
            for i, d in enumerate(items):
                with cols[i % 5]:
                    img = Image.open(d["img_path"]).convert("RGB")
                    x1, y1, x2, y2 = d["bbox"]
                    crop = img.crop((x1, y1, x2, y2))  # 약 부분만 자르기
                    crop = crop.resize((200, 200))
                    st.image(
                        crop,
                        caption=(
                            f"{d['img_path'].name}\n"
                            f"conf={d['conf']:.2f}"
                        ),
                    )

st.markdown("---")

# ---------------------------------------------
# 8) 애매한 약들 모아서 보기
#    - conf 낮은 약(박스)들 전부 한 데 모음
# ---------------------------------------------
st.markdown("### ⚠ 애매하게 예측한 약 모음")

if not uncertain_pills:
    st.write("🎉 애매한 약이 없습니다!")
else:
    cols2 = st.columns(5)
    for i, d in enumerate(uncertain_pills):
        with cols2[i % 5]:
            img = Image.open(d["img_path"]).convert("RGB")
            x1, y1, x2, y2 = d["bbox"]
            crop = img.crop((x1, y1, x2, y2))
            crop = crop.resize((200, 200))

            cname = CLASS_NAMES.get(int(d["cls"]), f"pill_{d['cls']}")

            st.image(
                crop,
                caption=(
                    f"{d['img_path'].name}\n"
                    f"cls={cname}, conf={d['conf']:.2f}"
                ),
            )
