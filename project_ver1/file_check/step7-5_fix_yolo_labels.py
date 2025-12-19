import os
from collections import Counter

LBL_TRAIN = "/home/ohs3201/work/step7_yolov8/labels/train"
LBL_VAL   = "/home/ohs3201/work/step7_yolov8/labels/val"

def scan(lbl_dir, split):
    c = Counter()
    bad_lines = 0
    bad_cls = 0
    files = 0

    for f in os.listdir(lbl_dir):
        if not f.endswith(".txt"):
            continue
        files += 1
        p = os.path.join(lbl_dir, f)
        with open(p, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    bad_lines += 1
                    continue
                try:
                    cls = int(parts[0])
                except:
                    bad_lines += 1
                    continue
                c[cls] += 1

                # 좌표 간단 검증(너무 과격하면 학습불안정)
                try:
                    vals = list(map(float, parts[1:]))
                    if any((v < -0.01 or v > 1.01) for v in vals):
                        bad_lines += 1
                except:
                    bad_lines += 1

    print(f"\n[{split}] label files: {files}")
    print(f"[{split}] unique classes: {len(c)}")
    if c:
        print(f"[{split}] cls min/max: {min(c)} .. {max(c)}")
        print(f"[{split}] top10: {c.most_common(10)}")
    print(f"[{split}] bad lines (format/coord): {bad_lines}")

def main():
    assert os.path.isdir(LBL_TRAIN), f"missing: {LBL_TRAIN}"
    assert os.path.isdir(LBL_VAL),   f"missing: {LBL_VAL}"
    scan(LBL_TRAIN, "TRAIN")
    scan(LBL_VAL, "VAL")
    print("\n✅ STEP 7-5 VALIDATION DONE")

if __name__ == "__main__":
    main()