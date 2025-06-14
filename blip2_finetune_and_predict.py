import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering, TrainingArguments, Trainer
import torch

# 開啟 TF32 加速，提升 GPU 訓練/推論效能
torch.backends.cuda.matmul.allow_tf32 = True

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

# 3. 載入BLIP-2 (改用 blip2-flan-t5-xl 模型)
model_name = "Salesforce/blip2-flan-t5-xl"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVisualQuestionAnswering.from_pretrained(model_name)

# 4. 準備資料集
train_dataset = Blip2ImageTextDataset(train_csv, processor)
test_dataset = Blip2ImageTextDataset(test_csv, processor)


# 5. 手動訓練 loop
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 3
batch_size = 1
accum_steps = 4  # gradient_accumulation_steps
lr = 5e-5

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
optimizer = AdamW(model.parameters(), lr=lr)
model.train()
optimizer.zero_grad()

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    running_loss = 0.0
    for step, batch in enumerate(tqdm(train_loader)):
        for k in batch:
            batch[k] = batch[k].to(device)
        outputs = model(**batch)
        loss = outputs.loss
        loss = loss / accum_steps
        loss.backward()
        running_loss += loss.item()
        if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()
    print(f"  Loss: {running_loss/len(train_loader):.4f}")


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
