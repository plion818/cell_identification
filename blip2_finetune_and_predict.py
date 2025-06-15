import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoModelForImageTextToText, TrainingArguments, Trainer
import torch

# 1. 自訂Dataset
class Blip2ImageTextDataset(Dataset):
    def __init__(self, csv_file, processor):
        self.data = pd.read_csv(csv_file)
        self.processor = processor
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image_path']
        # 若 img_path 為絕對路徑則直接用，否則用 script_dir 當前目錄拼接
        if not os.path.isabs(img_path):
            # 處理 Windows 路徑分隔符
            img_path = img_path.replace('\\', os.sep).replace('/', os.sep)
            img_path = os.path.join(script_dir, img_path)
        text = self.data.iloc[idx]['text']
        image = Image.open(img_path).convert('RGB')
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item['labels'] = self.processor.tokenizer(text, return_tensors="pt").input_ids.squeeze(0)
        return item

# 2. 設定路徑
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(script_dir, 'AI_MSC_密度照片')
data_dir = os.path.join(script_dir, 'data')
train_csv = os.path.join(data_dir, 'train_blip2.csv')
test_csv = os.path.join(data_dir, 'test_blip2.csv')

# 3. 載入 BLIP-1 影像描述模型（Hugging Face 官方推薦用法）
model_name = "Salesforce/blip-image-captioning-base"
processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
model = AutoModelForImageTextToText.from_pretrained(model_name)

# 4. 準備資料集
train_dataset = Blip2ImageTextDataset(train_csv, processor)
test_dataset = Blip2ImageTextDataset(test_csv, processor)

"""
# epoch 數（num_train_epochs）
# 定義：全訓練資料集被模型完整學習幾次。

# 小型資料集：通常 3~10 次即可，避免過度擬合。
# 大型資料集：可設大一點，但要觀察驗證損失，避免過度訓練。

# logging_steps（每幾個 batch 記錄一次損失）
"""

# 5. 訓練參數
training_args = TrainingArguments(
    output_dir="blip2_cell_growth_vlm",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    # evaluation_strategy="epoch",
    # save_strategy="epoch",
    logging_steps=10,
    learning_rate=5e-5,
    fp16=True,
    report_to="none"
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# 7. 微調
trainer.train()

# 8. 推論
result_df = pd.read_csv(test_csv)
pred_texts = []
for idx, row in result_df.iterrows():
    img_path = row['image_path']
    if not os.path.isabs(img_path):
        img_path = img_path.replace('\\', os.sep).replace('/', os.sep)
        img_path = os.path.join(script_dir, img_path)
    image = Image.open(img_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    pred_texts.append(generated_text)
result_df['predicted_text'] = pred_texts

# 9. 根據描述判斷預測區間
intervals = [d.replace('_','') for d in os.listdir(os.path.join(base_dir, 'train')) if os.path.isdir(os.path.join(base_dir, 'train', d))]
def match_interval(text):
    import re
    text_nospace = text.replace(' ', '')
    for interval in intervals:
        interval_nospace = interval.replace(' ', '')
        if interval_nospace in text_nospace:
            return interval
        # 允許描述中數字間有任意空格或符號
        # 例如 interval = '60-80'，pattern = r'6\s*0\s*[-~—－–_]?\s*8\s*0'
        m = re.match(r'^(\d+)-(\d+)$', interval)
        if m:
            n1, n2 = m.group(1), m.group(2)
            pattern = rf'{n1}\s*[-~—－–_]?\s*{n2}'
            if re.search(pattern, text):
                return interval
    return '未知'

# 10. 計算準確率（以 test_blip2.csv 的正確區間為標準）
def get_true_interval_from_path(img_path):
    # 依據圖片路徑中的資料夾名稱判斷正確區間
    # e.g. .../test/60-80_/1.tif -> 60-80
    parts = os.path.normpath(img_path).split(os.sep)
    for p in parts:
        if '-' in p and p.replace('_','').replace('-','').isdigit():
            return p.replace('_','')
    return '未知'

# 先建立 predicted_interval 欄位，再計算準確率
result_df['predicted_interval'] = result_df['predicted_text'].apply(match_interval)
result_df['true_interval'] = result_df['image_path'].apply(get_true_interval_from_path)
correct = (result_df['predicted_interval'] == result_df['true_interval']) & (result_df['true_interval'] != '未知')
accuracy = correct.sum() / (result_df['true_interval'] != '未知').sum() if (result_df['true_interval'] != '未知').sum() > 0 else 0

result_df.to_csv(os.path.join(script_dir, 'test_blip2_predictions.csv'), index=False)
print('微調與推論完成，結果已輸出到 test_blip2_predictions.csv')
print(f'模型在測試集的區間分類準確率：{accuracy:.2%}')
