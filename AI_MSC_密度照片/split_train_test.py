import os
import shutil
import random

# 設定隨機種子以確保可重現性
random.seed(42)

# 原始資料夾路徑
base_dir = 'AI_MSC_密度照片'
# 訓練集與測試集資料夾名稱
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# 建立訓練集與測試集資料夾
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 取得所有區間資料夾
interval_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) and '_' in f]

for interval in interval_folders:
    src_folder = os.path.join(base_dir, interval)
    images = [f for f in os.listdir(src_folder) if f.lower().endswith('.tif')]
    random.shuffle(images)
    n_total = len(images)
    n_train = int(n_total * 0.8)
    train_images = images[:n_train]
    test_images = images[n_train:]

    # 建立對應的區間資料夾
    train_interval_dir = os.path.join(train_dir, interval)
    test_interval_dir = os.path.join(test_dir, interval)
    os.makedirs(train_interval_dir, exist_ok=True)
    os.makedirs(test_interval_dir, exist_ok=True)

    # 複製圖片到訓練集
    for img in train_images:
        src_path = os.path.join(src_folder, img)
        dst_path = os.path.join(train_interval_dir, img)
        shutil.copy2(src_path, dst_path)

    # 複製圖片到測試集
    for img in test_images:
        src_path = os.path.join(src_folder, img)
        dst_path = os.path.join(test_interval_dir, img)
        shutil.copy2(src_path, dst_path)

print('資料分割完成！')
