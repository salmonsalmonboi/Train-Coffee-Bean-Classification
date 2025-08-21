from ultralytics import YOLO 

# **Load YOLO model**
model = YOLO('yolo12s.pt')

# **Train the model**
train_results = model.train (
    data     = "datasets/dataset_baseline/data.yaml",         # Path to dataset YAML
    epochs   = 100,                          # Number of training epochs
    imgsz    = 640,                          # Training image size (Width x Height) 
    device   = "0",                          # Device to run on (e.g., 0 for GPU, 'cpu' for CPU)
    workers  = 0,                            # Number of workers (set to 0 to avoid multiprocessing issues)
    patience = 50,                           # Early stopping patience
    lr0      = 0.001,                        # Initial learning rate
    lrf      = 0.01,                         # Final learning rate multiplier
    momentum = 0.9,                          # เพิ่ม Momentum (ช่วยให้ SGD ทำงานดีขึ้น)
    cos_lr   = True,                         # Cosine Annealing ลดค่า LR ตามเส้นโค้งโคไซน์ ทำให้โมเดลเรียนรู้ได้เสถียรขึ้น
    batch = 8,
    
    # **Regularization**
    weight_decay = 0.0005,                   # ใช้ L2 Regularization
    optimizer    = "AdamW",                  # optimization
    
    # **Augmentation**
    augment      = True,                     # Ensure augmentations are enabled
    hsv_h        = 0.015, 
    hsv_s        = 0.4, 
    hsv_v        = 0.2,                      # เพิ่ม Color Augmentation
    flipud       = 0.2, 
    fliplr       = 0.5,                      # เพิ่ม Perspective + Flip
    mosaic       = 0.6,                      # เปิดใช้ Mosaic Augmentation
    mixup        = 0.1,                      # เพิ่ม MixUp Augmentation
    scale        = 0.2,                      # ปรับขนาดภาพแบบสุ่ม
    translate    = 0.05,                     # เลื่อนภาพแบบสุ่ม
    shear        = 0.05                      # เพิ่ม Shear Augmentation

)

# **Validate the model**
metrics = model.val()
print("Training completed. Metrics:", metrics)


