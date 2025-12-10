import os
import shutil

# ★★★ 여기만 네 환경에 맞게 바꿔줘 ★★★
BASE_DIR = r"C:\Users\sangj\workspace\6team_beginner_project"

# 원본 전체 이미지가 있는 폴더 (651장 있던 곳)
ORIGINAL_IMG_DIR = r"C:\Users\sangj\workspace\6team_beginner_project\data_ai06\train_images"


# 라벨(txt) 있는 폴더들 (YOLO용)
LABEL_DIRS = [
    os.path.join(BASE_DIR, "yolo_dataset", "labels", "train"),
    os.path.join(BASE_DIR, "yolo_dataset", "labels", "val"),
]

# 라벨 없는 이미지들을 모아둘 폴더
UNLABELED_DIR = os.path.join(BASE_DIR, "unlabeled_images")
os.makedirs(UNLABELED_DIR, exist_ok=True)

# 1) 원본 모든 이미지 stem(확장자 뺀 이름) 수집
all_image_stems = set()
for root, dirs, files in os.walk(ORIGINAL_IMG_DIR):
    for fname in files:
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            stem, _ = os.path.splitext(fname)
            all_image_stems.add(stem)

print(f"원본 이미지 개수: {len(all_image_stems)}")

# 2) 라벨 있는 이미지 stem 수집 (txt 파일 기준)
labeled_stems = set()
for label_dir in LABEL_DIRS:
    if not os.path.isdir(label_dir):
        continue
    for fname in os.listdir(label_dir):
        if fname.lower().endswith(".txt"):
            stem, _ = os.path.splitext(fname)
            labeled_stems.add(stem)

print(f"라벨 있는 이미지 개수: {len(labeled_stems)}")

# 3) 라벨 없는 stem = 전체 - 라벨 있는 것
unlabeled_stems = sorted(all_image_stems - labeled_stems)
print(f"라벨 없는 이미지 개수: {len(unlabeled_stems)}")

# 4) 라벨 없는 이미지들을 unlabeled_images/ 로 복사
copied = 0
for stem in unlabeled_stems:
    src_path = None
    for ext in [".jpg", ".jpeg", ".png"]:
        cand = os.path.join(ORIGINAL_IMG_DIR, stem + ext)
        if os.path.exists(cand):
            src_path = cand
            break

    if src_path is None:
        print(f"[경고] {stem}.* 파일을 {ORIGINAL_IMG_DIR}에서 찾지 못함, 건너뜀")
        continue

    dst_path = os.path.join(UNLABELED_DIR, os.path.basename(src_path))
    shutil.copy2(src_path, dst_path)
    copied += 1

print(f"✅ unlabeled_images 폴더에 {copied}개 복사 완료!")
