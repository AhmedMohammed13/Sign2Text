import os

def rename_folders(start_dir, start_number=10):
    # الحصول على قائمة بالمجلدات في الدليل
    folders = [f for f in os.listdir(start_dir) if os.path.isdir(os.path.join(start_dir, f))]
    
    # ترتيب المجلدات لضمان التسلسل
    folders.sort()
    
    # إنشاء قائمة مؤقتة لتجنب تعارض الأسماء
    temp_prefix = "temp_"
    for index, folder in enumerate(folders):
        old_path = os.path.join(start_dir, folder)
        temp_path = os.path.join(start_dir, f"{temp_prefix}{index}")
        try:
            os.rename(old_path, temp_path)
            print(f"تمت إعادة تسمية المجلد مؤقتًا: {old_path} إلى {temp_path}")
        except Exception as e:
            print(f"خطأ أثناء إعادة التسمية المؤقتة لـ {old_path}: {e}")
    
    # إعادة تسمية المجلدات المؤقتة إلى الأرقام النهائية
    temp_folders = [f for f in os.listdir(start_dir) if os.path.isdir(os.path.join(start_dir, f)) and f.startswith(temp_prefix)]
    temp_folders.sort()
    
    for index, folder in enumerate(temp_folders):
        temp_path = os.path.join(start_dir, folder)
        new_name = str(start_number + index)
        new_path = os.path.join(start_dir, new_name)
        try:
            os.rename(temp_path, new_path)
            print(f"تمت إعادة تسمية المجلد {temp_path} إلى {new_path}")
        except Exception as e:
            print(f"خطأ أثناء إعادة تسمية {temp_path}: {e}")

# مثال على الاستخدام
if __name__ == "__main__":
    # استبدل هذا المسار بالمسار الفعلي للمجلدات
    directory = "./data"
    rename_folders(directory, start_number=10)