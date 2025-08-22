# ☕ Coffee Bean Classification 

โปรเจกต์นี้เป็นโค้ดสำหรับ **เทรนโมเดล YOLO** เพื่อตรวจจับและจำแนกเมล็ดกาแฟบนสายพาน

---

## 📂 Dataset

> **หมายเหตุ:** Dataset ไม่ได้รวมอยู่ใน repo (ถูก ignore ไว้ใน `.gitignore`)

1. ดาวน์โหลด dataset 15 Classหรือ Gatekeeper ได้จาก [Roboflow Project Link](https://universe.roboflow.com/chetsada-rngwe/coffee-bean-webcam)
2. ดาวน์โหลด dataset Model A ได้จาก [Roboflow Project Link](https://universe.roboflow.com/chetsada-rngwe/coffee-bean-webcam-good-b2oel)
3. ดาวน์โหลด dataset Model B ได้จาก [Roboflow Project Link](https://universe.roboflow.com/chetsada-rngwe/coffee-bean-webcam-defect-nn0fe)
4. หลังจากดาวน์โหลด ให้วางโฟลเดอร์ dataset ไว้ในโครงสร้างดังนี้:

  
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

ตัวอย่าง `data.yaml` แบบ 15 Class:
```yaml
train: datasets/train/images
val: datasets/valid/images
test: datasets/test/images

nc: 15
names: ["A", "AA", "AAA", "B", "Black", "Chipped", "Dry", "Elephant ear",
      "Faded", "Honey", "Pea berry", "Split", "Triangle", "Wash", "Weevil-infested"]
```



