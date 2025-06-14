import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering, TrainingArguments, Trainer
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

# 3. 載入BLIP-2 (改用 blip2-opt-2.7b 模型)
model_name = "Salesforce/blip2-opt-2.7b"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVisualQuestionAnswering.from_pretrained(model_name)

# 4. 準備資料集
train_dataset = Blip2ImageTextDataset(train_csv, processor)
test_dataset = Blip2ImageTextDataset(test_csv, processor)

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
    for interval in intervals:
        if interval in text:
            return interval
    return '未知'
result_df['predicted_interval'] = result_df['predicted_text'].apply(match_interval)
result_df.to_csv(os.path.join(script_dir, 'test_blip2_predictions.csv'), index=False)
print('微調與推論完成，結果已輸出到 test_blip2_predictions.csv')
