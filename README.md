# ☕ Coffee Bean Classification Baseline (15 Classes)

โปรเจกต์นี้เป็นโค้ดสำหรับ **เทรนโมเดล YOLO** เพื่อตรวจจับและจำแนกเมล็ดกาแฟออกเป็น **15 คลาส**  
โดยไฟล์หลักที่ใช้คือ `Train_Baseline_15Class.py`  

---

## 📂 Dataset

> **หมายเหตุ:** Dataset ไม่ได้รวมอยู่ใน repo (ถูก ignore ไว้ใน `.gitignore`)

1. ดาวน์โหลด dataset ได้จาก [Roboflow Project Link](https://universe.roboflow.com/chetsada-rngwe/coffee-bean-webcam) 
2. หลังจากดาวน์โหลด ให้วางโฟลเดอร์ dataset ไว้ในโครงสร้างดังนี้:

  
```
datasets/
├── train/
│ └── images/
│ └── labels/
├── valid/
│ └── images/
│ └── labels/
└── test/
└── images/
└── labels/
```

- ภายในจะมีไฟล์ `data.yaml` กำหนดโครงสร้าง dataset เช่น path, จำนวนคลาส และชื่อคลาส

ตัวอย่าง `data.yaml`:
```yaml
train: datasets/train/images
val: datasets/valid/images
test: datasets/test/images

nc: 15
names: ["A", "AA", "AAA", "B", "Black", "Chipped", "Dry", "Elephant ear",
      "Faded", "Honey", "Pea berry", "Split", "Triangle", "Wash", "Weevil-infested"]



