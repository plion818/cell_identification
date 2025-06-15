import os
import csv

# 設定訓練集與測試集路徑
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(script_dir, 'AI_MSC_密度照片')
data_dir = os.path.join(script_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# 產生標籤清單
intervals = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
intervals.sort()  # 確保順序一致

# 建立描述對應表
label_map = {interval: interval.replace('_', '') for interval in intervals}
desc_map = {interval: f"細胞生長比例為{interval.replace('_','')}%" for interval in intervals}

# 產生image-text配對csv
for split, split_dir in [('train', train_dir), ('test', test_dir)]:
    csv_path = os.path.join(data_dir, f'{split}_blip2.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_path', 'text'])
        for interval in intervals:
            folder = os.path.join(split_dir, interval)
            if not os.path.exists(folder):
                continue
            for img in os.listdir(folder):
                if img.lower().endswith('.tif'):
                    img_path = os.path.join(folder, img)
                    if split == 'train':
                        # 訓練集：圖片對應正確描述
                        writer.writerow([img_path, desc_map[interval]])
                    else:
                        # 測試集：圖片對應統一詢問
                        writer.writerow([img_path, '判斷圖片細胞生長比例?'])
print('train_blip2.csv 與 test_blip2.csv 已建立於 data 資料夾')
