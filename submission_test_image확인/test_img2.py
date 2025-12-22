import streamlit as st
import pandas as pd
import cv2
from PIL import Image
import os
import math

# =======================
# 설정
# =======================
CSV_PATH = "/home/ohs3201/6team_beginner_project/project_ver3/output/submit.csv"
IMAGE_DIR = "/home/ohs3201/6team_beginner_project/data/test_images"
RESULT_DIR = "/home/ohs3201/6team_beginner_project/submission_test_image확인/result"

NUM_COLS = 6
NUM_ROWS = 6
IMAGES_PER_PAGE = NUM_COLS * NUM_ROWS

st.set_page_config(layout="wide")
st.title("submission 결과 시각화")

# =======================
# CSV 로드
# =======================
df = pd.read_csv(CSV_PATH)

# =======================
# 카테고리 선택
# =======================
category_ids = sorted(df["category_id"].unique())

category_id = st.sidebar.radio(
    "Category 선택",
    category_ids,
    index=0
)

cat_df = df[df["category_id"] == category_id]

# =======================
# Score 기준값
# =======================
score_threshold = st.number_input(
    "Score 기준값",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

# =======================
# image_id별 score 처리
# =======================
image_score_list_df = (
    cat_df.groupby("image_id")["score"]
    .apply(list)
    .reset_index()
)

image_score_df = (
    cat_df.groupby("image_id")["score"]
    .max()
    .reset_index()
)

# =======================
# 중복 이미지
# =======================
dup_df = image_score_list_df[
    image_score_list_df["score"].apply(len) >= 2
]
dup_image_ids = dup_df["image_id"].tolist()

# =======================
# 상단 요약
# =======================
st.markdown(
    f"<div style='font-size:20px; font-weight:800;'>Category ID: {category_id}</div>",
    unsafe_allow_html=True
)

top_cols = st.columns([0.6, 5.4])

with top_cols[0]:
    rep_dir = os.path.join(RESULT_DIR, str(category_id))
    if os.path.exists(rep_dir):
        imgs = [
            f for f in os.listdir(rep_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if imgs:
            rep_img = Image.open(
                os.path.join(rep_dir, imgs[0])
            ).resize((300, 300))
            st.image(rep_img)

with top_cols[1]:
    st.markdown(
        f"""
        <div style="font-size:22px; line-height:1.6;">
            Total Images : {len(image_score_df)}<br>
            Score Range : {image_score_df.score.min():.3f} ~ {image_score_df.score.max():.3f}<br>
            Score &lt; {score_threshold} : {(image_score_df.score < score_threshold).sum()}<br>
            Score ≥ {score_threshold} : {(image_score_df.score >= score_threshold).sum()}
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()

# =======================
# 필터 / 정렬
# =======================
col1, col2 = st.columns([1, 3])

with col1:
    dup_only = st.checkbox("2개 이상 중복 이미지만 보기")

with col2:
    sort_option = st.radio(
        "정렬 기준",
        ["Score 낮은 순", "Image ID 낮은 순"],
        horizontal=True
    )

# =======================
# 정렬
# =======================
if sort_option == "Score 낮은 순":
    image_score_df = image_score_df.sort_values("score")
else:
    image_score_df = image_score_df.sort_values("image_id")

sorted_ids = image_score_df["image_id"].tolist()

# =======================
# 중복 필터 (핵심 수정)
# =======================
if dup_only:
    sorted_ids = [i for i in sorted_ids if i in dup_image_ids]

# =======================
# 페이지네이션
# =======================
total_pages = math.ceil(len(sorted_ids) / IMAGES_PER_PAGE)

page = st.number_input(
    "Page",
    min_value=1,
    max_value=max(total_pages, 1),
    value=1,
    step=1
)

page_image_ids = sorted_ids[
    (page - 1) * IMAGES_PER_PAGE : page * IMAGES_PER_PAGE
]

# =======================
# TAB
# =======================
tab1, tab2 = st.tabs(["결과 이미지", "원본 이미지"])

# =======================
# 결과 이미지
# =======================
with tab1:
    for r in range(NUM_ROWS):
        cols = st.columns(NUM_COLS)

        for c in range(NUM_COLS):
            idx = r * NUM_COLS + c
            if idx >= len(page_image_ids):
                break

            image_id = page_image_ids[idx]
            img_path = os.path.join(IMAGE_DIR, f"{image_id}.png")

            with cols[c]:
                if not os.path.exists(img_path):
                    continue

                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                rows = cat_df[cat_df["image_id"] == image_id]

                for _, row in rows.iterrows():
                    cv2.rectangle(
                        img,
                        (int(row.bbox_x), int(row.bbox_y)),
                        (int(row.bbox_x + row.bbox_w), int(row.bbox_y + row.bbox_h)),
                        (255, 0, 0),
                        3
                    )

                scores = rows["score"].tolist()

                score_html = []
                for s in scores:
                    if s < score_threshold:
                        score_html.append(f"<span style='color:red;'>{s:.3f}</span>")
                    else:
                        score_html.append(f"{s:.3f}")

                scores_str = ", ".join(score_html)

                st.markdown(
                    f"""
                    <div style="font-size:14px; font-weight:600;">
                        <span style="color:black;">{image_id}</span>
                        (<span>{scores_str}</span>)
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.image(img, use_container_width=True)

                if st.button("원본 보기", key=f"btn_{image_id}"):
                    st.session_state["selected_image"] = image_id

# =======================
# 원본 이미지
# =======================
with tab2:
    if "selected_image" in st.session_state:
        image_id = st.session_state["selected_image"]
        img_path = os.path.join(IMAGE_DIR, f"{image_id}.png")
        st.markdown(f"### 원본 이미지 : {image_id}")
        st.image(Image.open(img_path), use_container_width=True)
    else:
        st.info("결과 이미지 탭에서 이미지를 선택하세요.")
