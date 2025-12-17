#이미지는 없고 어노테이션만 있는 데이터들을 모아서 확인해보기 위해 하나의 json으로 모아주는 코드

import os
import json
from pathlib import Path

# 프로젝트 루트
BASE = Path(r"C:\Users\sangj\workspace\6team_beginner_project")

data_root = BASE / "data_ai06"
ann_root = data_root / "train_annotations"

# orphan json 목록이 들어있는 txt (앞에서 만든 json_for_nowhere.txt)
orphan_list_path = BASE / "json_for_nowhere.txt"

# 출력: orphan json들을 전부 모은 하나의 json 파일
out_path = BASE / "orphan_merged.json"

# 1) orphan json 경로들 읽기
with open(orphan_list_path, "r", encoding="utf-8") as f:
    orphan_paths = [line.strip() for line in f.readlines() if line.strip()]

print("이미지 없는 orphan json 개수:", len(orphan_paths))

merged = []  # 여기에 각 json의 전체 내용을 그대로 넣을 거야

# 2) 각 orphan json 열어서 merged 리스트에 추가
for rel_path in orphan_paths:
    json_path = ann_root / rel_path

    if not json_path.exists():
        print(f"[WARN] JSON 파일 없음: {json_path}")
        continue

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 원본 json 내용 그대로 + 경로 정보만 살짝 추가해줌 (원하면 빼도 됨)
    merged.append({
        "_source_path": str(rel_path),  # 어디서 온 json인지 표시
        **data
    })

# 3) 하나의 json 파일로 저장 (리스트 형태)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

print("\n👉 orphan_merged.json 생성 완료!")
print("경로:", out_path)
