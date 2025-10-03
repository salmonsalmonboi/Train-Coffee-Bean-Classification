# ==========================================================
# ============== MODEL 1: GATEKEEPER TRAINING ==============
# ==========================================================

from ultralytics import YOLO 

# ----- Phase 1: Train with frozen backbone ------
model = YOLO('yolo/yolo12s.pt')
train_results = model.train(
    data="C:/Users/jokoz/Documents/Vscode-work/datasets/dataset_gatekeeper/data.yaml",  # 2 classes: Good, Defect
    epochs=20,
    freeze=10,                               # แช่แข็ง backbone 10 ชั้นแรก
    imgsz=640,
    rect=False,                              # ให้สุ่มตัด/แพดอิสระช่วงปรับหัว
    device="0",
    workers=0,
    patience=12,
    lr0=1e-3, lrf=0.01, warmup_epochs=3, warmup_momentum=0.8,
    momentum=0.9, cos_lr=True,
    batch=8,
    amp=True,

    # Regularization / Optimizer
    weight_decay=1e-2,
    optimizer="AdamW",

    # Augmentation (สายพานจริง: แสง/เงาเบา ๆ, geometry น้อย)
    augment=True,
    hsv_h=0.012, hsv_s=0.30, hsv_v=0.20,
    flipud=0.0, fliplr=0.5,
    mosaic=0.25,           # ลดความแรงของ mosaic
    mixup=0.10,
    scale=0.25, translate=0.15, degrees=5.0,
    shear=0.0, perspective=0.0,
    close_mosaic=7,
    label_smoothing=0.05,

    project="Coffee-Bean-Hierarchical-Classification",
    name="gatekeeper-phase1-train(yolov12s)",
    exist_ok=True
)

# ----- Phase 2: Fine-tune -----
model = YOLO('Coffee-Bean-Hierarchical-Classification/gatekeeper-phase1-train(yolov12s)/weights/best.pt')

train_results = model.train(
    data="C:/Users/jokoz/Documents/Vscode-work/datasets/dataset_gatekeeper/data.yaml",
    epochs=80,
    freeze=0,
    imgsz=704,            # ลดจาก 832 -> 704 เพื่อสปีดขึ้นโดยแม่นยังดี
    rect=True,            # rectangular training ให้ภาพไม่ถูกครอปสี่เหลี่ยมจัตุรัส
    device="0",
    workers=0,
    patience=25,
    lr0=5e-4, lrf=0.01, momentum=0.9,
    cos_lr=True,
    batch=8,

    weight_decay=1e-3,
    optimizer="AdamW",

    augment=True,
    hsv_h=0.01, hsv_s=0.25, hsv_v=0.18,
    flipud=0.0, fliplr=0.5,
    scale=0.18, translate=0.10, degrees=4.0,
    shear=0.0, perspective=0.0,
    mosaic=0.10,          # คงไว้เล็กน้อยเพื่อความหลากหลายแต่ไม่ทำลายรูปทรง
    amp=True,
    label_smoothing=0.03,

    project="Coffee-Bean-Hierarchical-Classification",
    name="gatekeeper-phase2-finetune(yolov12s)",
    exist_ok=True
)

metrics = model.val()
print("Gatekeeper Model Training Completed. Metrics:", metrics)
