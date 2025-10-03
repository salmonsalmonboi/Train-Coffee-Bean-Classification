# ==========================================================
# ========== MODEL B: DEFECT-SPECIALIST TRAINING ============
# ==========================================================

from ultralytics import YOLO 

# Dataset yaml: datasets/dataset_defect_specialist/data.yaml
# classes: Black, Chipped, Elephant ear, Faded, Split, Triangle, Weevil-infested

# ----- Phase 1: Freeze -----
model = YOLO('yolo/yolo12s.pt')
train_results = model.train(
    data="C:/Users/jokoz/Documents/Vscode-work/datasets/dataset_defect_specialist/data.yaml",
    epochs=25,
    freeze=10,
    imgsz=704,                               # defect บางชนิดเป็น texture/รูเล็ก → ให้ละเอียดขึ้น
    rect=False,
    device="0", workers=0,
    patience=15,
    lr0=1e-3, lrf=0.01, warmup_epochs=3, warmup_momentum=0.8,
    momentum=0.9, cos_lr=True,
    batch=8, amp=True,

    weight_decay=8e-3,
    optimizer="AdamW",

    # Augment: ระวังอย่าบิดรูปทรงตำหนิ (แตก/รู/ความซีด)
    augment=True,
    hsv_h=0.010, hsv_s=0.28, hsv_v=0.18,
    fliplr=0.5, flipud=0.0,
    mosaic=0.18,            # น้อยลงเพื่อคง pattern รอยตำหนิ
    mixup=0.08,
    scale=0.20, translate=0.12, degrees=4.0,
    shear=0.0, perspective=0.0,
    close_mosaic=8,
    label_smoothing=0.04,

    project="Coffee-Bean-Hierarchical-Classification",
    name="defect-specialist-phase1-train(yolov12s)",
    exist_ok=True
)

# ----- Phase 2: Finetune -----
model = YOLO('Coffee-Bean-Hierarchical-Classification/defect-specialist-phase1-train(yolov12s)/weights/best.pt')

train_results = model.train(
    data="C:/Users/jokoz/Documents/Vscode-work/datasets/dataset_defect_specialist/data.yaml",
    epochs=100,
    freeze=0,
    imgsz=800,               # ดันละเอียดอีกนิดเพื่อ Triangle/Split/Weevil
    rect=True,
    device="0", workers=0,
    patience=30,
    lr0=4e-4, lrf=0.01, momentum=0.9,
    cos_lr=True,
    batch=6,                 # imgsz ใหญ่ขึ้น → batch ลดลงกัน VRAM
    amp=True,

    weight_decay=8e-4,
    optimizer="AdamW",

    augment=True,
    hsv_h=0.010, hsv_s=0.25, hsv_v=0.16,
    fliplr=0.5, flipud=0.0,
    scale=0.16, translate=0.10, degrees=3.0,
    shear=0.0, perspective=0.0,
    mosaic=0.08, mixup=0.06,
    label_smoothing=0.03,

    project="Coffee-Bean-Hierarchical-Classification",
    name="defect-specialist-phase2-finetune(yolov12s)",
    exist_ok=True
)

metrics = model.val()
print("Defect-Specialist Training Completed. Metrics:", metrics)
