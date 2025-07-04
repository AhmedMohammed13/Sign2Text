import os

# المسار إلى مجلد الصور
folder_path = './data/21'  # عدّل حسب الحاجة

# بداية الترقيم
start_number = 0

# الامتداد المستهدف
target_extension = '.jpg'

# الحصول على الصور الحالية في المجلد بامتداد .jpg أو .jpeg
files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg'))]

# ترتيب الأسماء لعملية منظمة
files.sort()

# الترقيم مع تجاوز الملفات الموجودة
current_number = start_number
for filename in files:
    # توليد اسم جديد بدون تعارض
    while True:
        new_filename = f"{current_number}{target_extension}"
        new_path = os.path.join(folder_path, new_filename)
        if not os.path.exists(new_path):
            break
        current_number += 1  # تخطى الرقم الموجود

    # إعادة التسمية
    old_path = os.path.join(folder_path, filename)
    os.rename(old_path, new_path)
    print(f"Renamed: {filename} → {new_filename}")
    
    current_number += 1  # استعد للرقم التالي

print("✅ تم الترقيم مع تجاوز الأسماء الموجودة.")