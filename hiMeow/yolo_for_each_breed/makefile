.PHONY: all train_blepharitis train_conjunctivitis train_corneal_sequestrum train_corneal_ulcer train_non_ulcerative_keratitis

# 모든 질병 학습 실행
all: train_blepharitis train_conjunctivitis train_corneal_sequestrum train_corneal_ulcer train_non_ulcerative_keratitis

# Blepharitis 데이터 학습
train_blepharitis:
	@echo "Training YOLO Classification for Blepharitis..."
	yolo classify train data=/content/drive/MyDrive/yolo_dataset/코리아_숏헤어/Blepharitis/datasets model=yolo11n-cls.pt epochs=100 imgsz=64

# Conjunctivitis 데이터 학습
train_conjunctivitis:
	@echo "Training YOLO Classification for Conjunctivitis..."
	yolo classify train data=/content/drive/MyDrive/yolo_dataset/Conjunctivitis/datasets model=yolo11n-cls.pt epochs=100 imgsz=64

# Corneal Sequestrum 데이터 학습
train_corneal_sequestrum:
	@echo "Training YOLO Classification for Corneal Sequestrum..."
	yolo classify train data=/content/drive/MyDrive/yolo_dataset/코리아_숏헤어/Corneal_Secquestrum/datasets model=yolo11n-cls.pt epochs=100

# Corneal Ulcer 데이터 학습
train_corneal_ulcer:
	@echo "Training YOLO Classification for Corneal Ulcer..."
	yolo classify train data=/content/drive/MyDrive/yolo_dataset/Corneal_Ulcer/datasets model=yolo11n-cls.pt epochs=100 imgsz=64

train_non_ulcerative_keratitis:
	@echo "Training YOLO Classification for Non-Ulcerative Keratitis..."
	yolo classify train data=/content/drive/MyDrive/yolo_dataset/Non_Ulcerative_Keratitis/datasets model=yolo11n-cls.pt epochs=100 imgsz=64
