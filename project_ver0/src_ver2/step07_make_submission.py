import pandas as pd

# 🔥 변환하고 싶은 파일들
files = [
    "/home/ohs3201/6team_beginner_project/results/submission/ver2/FasterRCNN_ver2.csv",
    "/home/ohs3201/6team_beginner_project/results/submission/ver2/YOLOv8L_continue_ver2.csv",
    "/home/ohs3201/6team_beginner_project/results/submission/ver2/final_ensemble_WBF_ver2.csv"
]

# 🔥 저장될 출력 파일들
out_files = [
    "/home/ohs3201/6team_beginner_project/results/submission/ver2/FasterRCNN_submit_ver2.csv",
    "/home/ohs3201/6team_beginner_project/results/submission/ver2/YOLOv8L_continue_submit_ver2.csv",
    "/home/ohs3201/6team_beginner_project/results/submission/ver2/final_WBF_submit_ver2.csv"
]

for in_path, out_path in zip(files, out_files):
    df = pd.read_csv(in_path)
    df.insert(0, "annotation_id", range(1, len(df) + 1))
    df.to_csv(out_path, index=False)
    print(f"✨ Saved submission file → {out_path}")