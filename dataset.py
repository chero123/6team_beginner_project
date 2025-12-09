import os
import json
import cv2
import torch
from torch.utils.data import Dataset


class PillDataset(Dataset):
    def __init__(self, root, split="train"):
        self.root = root
        self.img_dir = os.path.join(root, "train_images")
        self.ann_dir = os.path.join(root, "train_annotations")

        # 1) train / val 파일 리스트
        split_path = os.path.join(root, "split.json")
        with open(split_path, "r", encoding="utf-8") as f:
            split_info = json.load(f)
        self.json_files = split_info[split]

        # 2) category_id_mapping.json (old_id -> 이름)
        mapping_path = os.path.join(root, "category_id_mapping.json")
        with open(mapping_path, "r", encoding="utf-8") as f:
            cat_raw = json.load(f)

        self.old_ids = sorted([int(k) for k in cat_raw.keys()])
        self.id_map = {old_id: idx for idx, old_id in enumerate(self.old_ids)}
        self.class_names = [cat_raw[str(old_id)] for old_id in self.old_ids]
        self.num_classes = len(self.old_ids)

        # 3) 이미지 인덱스 만들기 (파일명 -> 실제 경로)
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
                        if fname not in self.img_index:
                            self.img_index[fname] = os.path.join(cur_root, fname)

        print(f"[PillDataset] indexed {len(self.img_index)} images")

        # 리사이즈 타겟 크기
        self.target_size = 640

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        # 1) json 열기
        json_rel = self.json_files[idx]
        json_path = os.path.join(self.ann_dir, json_rel)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 2) 이미지 정보
        img_info = data["images"][0]
        img_name_in_json = img_info["file_name"]
        img_file = os.path.basename(img_name_in_json)

        img_path = self.img_index.get(img_file, None)
        if img_path is None:
            print(f"[경고] 이미지 인덱스에 없음, 스킵: {img_file}")
            return self.__getitem__((idx + 1) % len(self))

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"cv2.imread 실패: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 원본 크기
        h0, w0 = img.shape[:2]

        # 3) annotation 읽기 (원본 좌표 기준)
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

            # 나중에 스케일링할 거라 일단 저장만
            boxes.append([x, y, w, h])
            labels.append(cid)
            iscrowd.append(ann.get("iscrowd", 0))

        # 4) 이미지 리사이즈 + 박스도 같이 스케일링
        tgt = self.target_size
        scale_x = tgt / w0
        scale_y = tgt / h0

        img_resized = cv2.resize(img, (tgt, tgt))

        boxes_xyxy = []
        areas_scaled = []

        for (x, y, w, h) in boxes:
            x1 = x * scale_x
            y1 = y * scale_y
            x2 = (x + w) * scale_x
            y2 = (y + h) * scale_y
            boxes_xyxy.append([x1, y1, x2, y2])
            areas_scaled.append((x2 - x1) * (y2 - y1))

        # 5) 텐서로 변환
        img_t = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0

        if len(boxes_xyxy) > 0:
            boxes_xyxy = torch.as_tensor(boxes_xyxy, dtype=torch.float32)
            labels_t   = torch.as_tensor(labels, dtype=torch.int64)
            areas_t    = torch.as_tensor(areas_scaled, dtype=torch.float32)
            iscrowd_t  = torch.as_tensor(iscrowd[: len(boxes_xyxy)], dtype=torch.int64)
        else:
            # 박스가 없는 이미지도 안전하게 처리
            boxes_xyxy = torch.zeros((0, 4), dtype=torch.float32)
            labels_t   = torch.zeros((0,), dtype=torch.int64)
            areas_t    = torch.zeros((0,), dtype=torch.float32)
            iscrowd_t  = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_xyxy,
            "labels": labels_t,
            "image_id": torch.tensor([idx]),
            "area": areas_t,
            "iscrowd": iscrowd_t,
        }

        return img_t, target
