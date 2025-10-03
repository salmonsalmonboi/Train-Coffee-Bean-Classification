# ==========================================================
# ============ MODEL A: GOOD-SPECIALIST TRAINING ============
# ==========================================================

from ultralytics import YOLO 

# Dataset yaml: datasets/dataset_good_specialist/data.yaml
# classes: A, AA, AAA, B, Dry, Honey, Pea berry, Wash

# ----- Phase 1: Freeze -----
model = YOLO('yolo/yolo12s.pt')
train_results = model.train(
    data="C:/Users/jokoz/Documents/Vscode-work/datasets/dataset_good_specialist/data.yaml",
    epochs=25,                               # เพิ่มเล็กน้อย เพราะคลาสย่อยหลายกลุ่ม
    freeze=10,
    imgsz=672,                               # ใหญ่กว่า 640 นิดเพื่อช่วยแยกเกรด/texture
    rect=False,
    device="0", workers=0,
    patience=15,
    lr0=1e-3, lrf=0.01, warmup_epochs=3, warmup_momentum=0.8,
    momentum=0.9, cos_lr=True,
    batch=8, amp=True,

    weight_decay=8e-3,
    optimizer="AdamW",

    # Augment: เน้น photometric มากกว่า geometry (กันสับสน Dry/Honey/Wash)
    augment=True,
    hsv_h=0.012, hsv_s=0.30, hsv_v=0.22,
    fliplr=0.5, flipud=0.0,
    mosaic=0.20, mixup=0.10,
    scale=0.22, translate=0.12, degrees=4.0,
    shear=0.0, perspective=0.0,
    close_mosaic=7,
    label_smoothing=0.05,

    project="Coffee-Bean-Hierarchical-Classification",
    name="good-specialist-phase1-train(yolov12s)",
    exist_ok=True
)

# ----- Phase 2: Finetune -----
model = YOLO('Coffee-Bean-Hierarchical-Classification/good-specialist-phase1-train(yolov12s)/weights/best.pt')

train_results = model.train(
    data="C:/Users/jokoz/Documents/Vscode-work/datasets/dataset_good_specialist/data.yaml",
    epochs=90,
    freeze=0,
    imgsz=736,
    rect=True,
    device="0", workers=0,
    patience=25,
    lr0=5e-4, lrf=0.01, momentum=0.9,
    cos_lr=True,
    batch=8,

    weight_decay=8e-4,
    optimizer="AdamW",

    augment=True,
    hsv_h=0.010, hsv_s=0.25, hsv_v=0.18,
    fliplr=0.5, flipud=0.0,
    scale=0.18, translate=0.10, degrees=3.0,
    shear=0.0, perspective=0.0,
    mosaic=0.10,
    amp=True,
    label_smoothing=0.03,

    project="Coffee-Bean-Hierarchical-Classification",
    name="good-specialist-phase2-finetune(yolov12s)",
    exist_ok=True
)

metrics = model.val()
print("Good-Specialist Training Completed. Metrics:", metrics)
