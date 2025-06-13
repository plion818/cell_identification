import os
import csv

# 設定訓練集與測試集路徑
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(script_dir, 'AI_MSC_密度照片')
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# 產生標籤清單
intervals = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
intervals.sort()  # 確保順序一致

# 建立標籤對應表
label_map = {interval: interval.replace('_', '') for interval in intervals}

# 產生csv檔案
for split, split_dir in [('train', train_dir), ('test', test_dir)]:
    csv_path = os.path.join(base_dir, f'{split}_labels.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_path', 'label'])
        for interval in intervals:
            folder = os.path.join(split_dir, interval)
            if not os.path.exists(folder):
                continue
            for img in os.listdir(folder):
                if img.lower().endswith('.tif'):
                    img_path = os.path.join(folder, img)
                    writer.writerow([img_path, label_map[interval]])
print('train_labels.csv 與 test_labels.csv 已建立')
