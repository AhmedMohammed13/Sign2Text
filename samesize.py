import os
import cv2

DATA_DIR = './data'
VALID_EXTENSIONS = ('.jpg', '.jpeg')

for class_dir in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_dir)

    if not os.path.isdir(class_path):
        continue

    for img_name in os.listdir(class_path):
        # ✅ فلترة الامتدادات غير المناسبة
        if not img_name.lower().endswith(VALID_EXTENSIONS):
            continue

        img_path = os.path.join(class_path, img_name)

        # ✅ محاولة تحميل الصورة
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ لم يتم تحميل الصورة: {img_path}")
            continue

        # ✅ فقط تأكيد أن الصورة قابلة للاستخدام
        height, width = img.shape[:2]
        print(f"✅ صورة صالحة: {img_path} | الحجم: {width}x{height}")

print("✅ جميع الصور تمت مراجعتها بنجاح.")
