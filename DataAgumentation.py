'''
import os
import cv2
import numpy as np

DATA_DIR = './data'
def augment_image(image):
    augmented_images = []
    rows, cols, _ = image.shape
    
    # الصورة الأصلية (مع وصف لتجنب التعارض)
    augmented_images.append(('original', image))
    # القلب الأفقي
    flipped = cv2.flip(image, 1)
    augmented_images.append(('flipped', flipped))
    # التدوير (20 درجة)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 20, 1)
    rotated = cv2.warpAffine(image, M, (cols, rows))
    augmented_images.append(('rotated', rotated))
    # تغيير السطوع (زيادة 50%)
    brightness = cv2.convertScaleAbs(image, beta=50)
    augmented_images.append(('brightness', brightness))
    
    return augmented_images

for class_dir in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_dir)
    
    if not os.path.isdir(class_path):
        continue
    
    for img_name in os.listdir(class_path):
        if not img_name.lower().endswith(('.jpg', '.jpeg')):
            continue 
        img_path = os.path.join(class_path, img_name)
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ لم يتم تحميل الصورة: {img_path}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        augmented_images = augment_image(img_rgb)
        for aug_type, aug_img in augmented_images:
            aug_filename = f"{os.path.splitext(img_name)[0]}_{aug_type}{os.path.splitext(img_name)[1]}"
            aug_path = os.path.join(class_path, aug_filename)
            base_name, ext = os.path.splitext(img_name)
            counter = 1
            while os.path.exists(aug_path):
                aug_filename = f"{base_name}_{aug_type}_{counter}{ext}"
                aug_path = os.path.join(class_path, aug_filename)
                counter += 1
            cv2.imwrite(aug_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
            print(f"✅ تم حفظ الصورة: {aug_path}")

print("✅ تم تكبير البيانات مع الاحتفاظ بالأصلية بنجاح.")

'''


import os

DATA_DIR = './data'
AUGMENTATION_TAGS = ['_original', '_resized', '_original_resized']

deleted_count = 0

for class_dir in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_dir)
    
    if not os.path.isdir(class_path):
        continue

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        
        # إذا احتوى اسم الصورة على أي من العلامات، نحذفه
        if any(tag in img_name.lower() for tag in AUGMENTATION_TAGS):
            try:
                os.remove(img_path)
                deleted_count += 1
                print(f"🗑️ تم حذف: {img_path}")
            except Exception as e:
                print(f"❌ خطأ أثناء حذف {img_path}: {e}")

print(f"\n✅ تم حذف {deleted_count} صورة معدلة والاحتفاظ بالصور الأصلية فقط.")






