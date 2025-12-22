NUM_CLASSES = 28   # 실제 클래스
model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, NUM_CLASSES + 1)