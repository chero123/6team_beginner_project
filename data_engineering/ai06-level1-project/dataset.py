import os
import json
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PillDataset(Dataset):
    def __init__(self, root, split="train"):
        self.root = root
        # 원래 쓰던 경로들
        self.img_dir = os.path.join(root, "train_images")
        self.ann_dir = os.path.join(root, "train_annotations")

        # 1) train / val 파일 리스트 읽기
        split_path = os.path.join(root, "split.json")
        with open(split_path, "r", encoding="utf-8") as f:
            split_info = json.load(f)
        self.json_files = split_info[split]

        # 2) category_id_mapping.json 읽기 (old_id -> 약 이름)
        mapping_path = os.path.join(root, "category_id_mapping.json")
        with open(mapping_path, "r", encoding="utf-8") as f:
            cat_raw = json.load(f)

        # 문자열 키를 int로 변환해서 정렬
        self.old_ids = sorted([int(k) for k in cat_raw.keys()])
        # old_id(int) -> 0 ~ num_classes-1
        self.id_map = {old_id: idx for idx, old_id in enumerate(self.old_ids)}
        self.class_names = [cat_raw[str(old_id)] for old_id in self.old_ids]
        self.num_classes = len(self.old_ids)

        # 3) 이미지 파일 이름 → 실제 경로 인덱스 만들기
        #    train_images / test_images 아래를 전부 뒤져서
        #    "K-0033...png" 같은 파일명을 키로 쓰는 dict 생성
        self.img_index = {}
        img_root_dirs = [
            os.path.join(root, "train_images"),
            os.path.join(root, "test_images"),
        ]

        for base_dir in img_root_dirs:
            if not os.path.isdir(base_dir):
                continue
            for cur_root, dirs, files in os.walk(base_dir):
                for fname in files:
                    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                        # 같은 이름이 여러 폴더에 있으면 제일 먼저 찾은 것만 사용
                        if fname not in self.img_index:
                            self.img_index[fname] = os.path.join(cur_root, fname)

        print(f"[PillDataset] indexed {len(self.img_index)} images")

        # 4) Albumentations 변환 설정
        if split == "train":
            self.transform = A.Compose(
                [
                    A.LongestMaxSize(max_size=1024),
                    A.PadIfNeeded(
                        min_height=1024,
                        min_width=1024,
                        border_mode=cv2.BORDER_CONSTANT
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                    A.ShiftScaleRotate(
                        shift_limit=0.02,
                        scale_limit=0.1,
                        rotate_limit=10,
                        border_mode=cv2.BORDER_CONSTANT,
                        p=0.5
                    ),
                    A.Normalize(mean=(0.5, 0.5, 0.5),
                                std=(0.5, 0.5, 0.5)),
                    ToTensorV2()
                ],
                bbox_params=A.BboxParams(
                    format="coco",   # [x, y, w, h]
                    label_fields=["labels"],
                    min_area=1,
                    min_visibility=0.1,
                )
            )
        else:
            self.transform = A.Compose(
                [
                    A.LongestMaxSize(max_size=1024),
                    A.PadIfNeeded(
                        min_height=1024,
                        min_width=1024,
                        border_mode=cv2.BORDER_CONSTANT
                    ),
                    A.Normalize(mean=(0.5, 0.5, 0.5),
                                std=(0.5, 0.5, 0.5)),
                    ToTensorV2()
                ],
                bbox_params=A.BboxParams(
                    format="coco",
                    label_fields=["labels"],
                    min_area=1,
                    min_visibility=0.0,
                )
            )

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        # 1) json 파일 열기
        json_rel = self.json_files[idx]
        json_path = os.path.join(self.ann_dir, json_rel)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 2) 이미지 정보 읽기 (COCO 형식 가정)
        img_info = data["images"][0]
        img_name_in_json = img_info["file_name"]
        # 혹시 file_name에 폴더가 포함돼 있어도 파일명만 따서 사용
        img_file = os.path.basename(img_name_in_json)

        # 우리가 만든 인덱스에서 실제 경로 찾기
        img_path = self.img_index.get(img_file, None)
        if img_path is None:
            raise FileNotFoundError(f"이미지 인덱스에 없음: {img_file}")

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"cv2.imread 실패: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 3) annotation 읽기
        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in data["annotations"]:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue

            old_cid = int(ann["category_id"])
            cid = self.id_map[old_cid]  # 0 ~ num_classes-1

            boxes.append([x, y, w, h])  # coco 형식
            labels.append(cid)
            areas.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))

        # 4) Albumentations 적용
        transformed = self.transform(
            image=img,
            bboxes=boxes,
            labels=labels,
        )

        img_t    = transformed["image"]
        boxes_t  = transformed["bboxes"]
        labels_t = transformed["labels"]

        # 5) [x, y, w, h] -> [x1, y1, x2, y2]
        boxes_xyxy = []
        for (x, y, w, h) in boxes_t:
            boxes_xyxy.append([x, y, x + w, y + h])

        boxes_xyxy = torch.as_tensor(boxes_xyxy, dtype=torch.float32)
        labels_t   = torch.as_tensor(labels_t, dtype=torch.int64)
        areas_t    = torch.as_tensor(areas[: len(boxes_xyxy)], dtype=torch.float32)
        iscrowd_t  = torch.as_tensor(iscrowd[: len(boxes_xyxy)], dtype=torch.int64)

        target = {
            "boxes": boxes_xyxy,
            "labels": labels_t,
            "image_id": torch.tensor([idx]),
            "area": areas_t,
            "iscrowd": iscrowd_t,
        }

        return img_t, target
