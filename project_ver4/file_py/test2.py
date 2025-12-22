from collections import Counter
import json

coco = json.load(open(
    "/home/ohs3201/6team_beginner_project/project_ver4/coco/train_coco_trainid_SAFE_FULL_IMAGES.json"
))

cnt = Counter(a["category_id"] for a in coco["annotations"])
print("COCO 실제 등장 train_id 수:", len(cnt))
print("하위 20개:", cnt.most_common()[-20:])