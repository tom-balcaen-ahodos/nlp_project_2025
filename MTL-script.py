import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_scheduler
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import os
import numpy as np

# Configuration
MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset class
class EmailDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.data = dataframe
        self.tokenizer = tokenizer

        self.texts = dataframe.apply(lambda x: f"Subject: {x['Subject']}\nBody: {x['TextBody']}", axis=1).tolist()
        self.is_complaint = dataframe['is_complaint'].tolist()
        self.priority = dataframe['priority'].tolist()
        self.complaint_type = dataframe['complaint_type'].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], truncation=True, padding="max_length", max_length=512, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "is_complaint": torch.tensor(self.is_complaint[idx], dtype=torch.long),
            "priority": torch.tensor(self.priority[idx], dtype=torch.long),
            "complaint_type": torch.tensor(self.complaint_type[idx], dtype=torch.long)
        }

# Model definition
class MultiTaskModel(nn.Module):
    def __init__(self, model_name, num_priorities, num_complaint_types):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size

        self.classifier_is_complaint = nn.Linear(hidden_size, 2)
        self.classifier_priority = nn.Linear(hidden_size, num_priorities)
        self.classifier_complaint_type = nn.Linear(hidden_size, num_complaint_types)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]

        is_complaint_logits = self.classifier_is_complaint(pooled_output)
        priority_logits = self.classifier_priority(pooled_output)
        complaint_type_logits = self.classifier_complaint_type(pooled_output)

        return is_complaint_logits, priority_logits, complaint_type_logits

# Load data
train_df = pd.read_csv("train_set_80_percent_20k.csv", sep=";")
test_df = pd.read_csv("test_set_20_percent_20k.csv", sep=";")

# Rename columns to match expected names in your code
column_map = {
    "Priority": "priority",
    "Subject": "Subject",
    "TextBody": "TextBody",
    "Complaint": "is_complaint",
    "Complaint_type": "complaint_type"
}

train_df = train_df.rename(columns=column_map)
test_df = test_df.rename(columns=column_map)

# Label encoding
priority_labels = train_df["priority"].unique().tolist()
complaint_type_labels = train_df["complaint_type"].unique().tolist()
priority2id = {v: i for i, v in enumerate(priority_labels)}
id2priority = {i: v for v, i in priority2id.items()}
ctype2id = {v: i for i, v in enumerate(complaint_type_labels)}
id2ctype = {i: v for v, i in ctype2id.items()}

train_df["priority"] = train_df["priority"].map(priority2id)
train_df["complaint_type"] = train_df["complaint_type"].map(ctype2id)
test_df["priority"] = test_df["priority"].map(priority2id)
test_df["complaint_type"] = test_df["complaint_type"].map(ctype2id)

# Tokenizer and datasets
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_dataset = EmailDataset(train_df, tokenizer)
test_dataset = EmailDataset(test_df, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Initialize model
model = MultiTaskModel(MODEL_NAME, len(priority_labels), len(complaint_type_labels)).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# Training loop
loss_fn = nn.CrossEntropyLoss()
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        y_is_complaint = batch["is_complaint"].to(DEVICE)
        y_priority = batch["priority"].to(DEVICE)
        y_ctype = batch["complaint_type"].to(DEVICE)

        optimizer.zero_grad()
        out_is_complaint, out_priority, out_ctype = model(input_ids, attention_mask)

        loss = loss_fn(out_is_complaint, y_is_complaint) + \
               loss_fn(out_priority, y_priority) + \
               loss_fn(out_ctype, y_ctype)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

# Evaluation
model.eval()
y_true_is_complaint = []
y_pred_is_complaint = []
y_score_is_complaint = []

y_true_priority = []
y_pred_priority = []
y_score_priority = []

y_true_ctype = []
y_pred_ctype = []
y_score_ctype = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        y_is_complaint = batch["is_complaint"]
        y_priority = batch["priority"]
        y_ctype = batch["complaint_type"]

        out_is_complaint, out_priority, out_ctype = model(input_ids, attention_mask)

        probs_is_complaint = torch.softmax(out_is_complaint, dim=-1)
        probs_priority = torch.softmax(out_priority, dim=-1)
        probs_ctype = torch.softmax(out_ctype, dim=-1)

        y_pred_is_complaint.extend(torch.argmax(probs_is_complaint, dim=-1).cpu().tolist())
        y_true_is_complaint.extend(y_is_complaint.tolist())
        y_score_is_complaint.extend(probs_is_complaint[:, 1].cpu().tolist())  # for ROC

        y_pred_priority.extend(torch.argmax(probs_priority, dim=-1).cpu().tolist())
        y_true_priority.extend(y_priority.tolist())
        y_score_priority.extend(probs_priority.cpu().tolist())

        y_pred_ctype.extend(torch.argmax(probs_ctype, dim=-1).cpu().tolist())
        y_true_ctype.extend(y_ctype.tolist())
        y_score_ctype.extend(probs_ctype.cpu().tolist())

# Classification reports
print("\nComplaint Detection Report:")
print(classification_report(y_true_is_complaint, y_pred_is_complaint))

print("\nPriority Classification Report:")
print(classification_report(y_true_priority, y_pred_priority, target_names=priority_labels))

print("\nComplaint Type Classification Report:")
print(classification_report(y_true_ctype, y_pred_ctype, target_names=complaint_type_labels))

# ROC and PR curves (binary for complaint detection)
fpr, tpr, _ = roc_curve(y_true_is_complaint, y_score_is_complaint)
precision, recall, _ = precision_recall_curve(y_true_is_complaint, y_score_is_complaint)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc(fpr, tpr):.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Complaint Detection")
plt.legend()
plt.savefig("roc_complaint_detection.png")

plt.figure()
plt.plot(recall, precision, label="PR Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Complaint Detection")
plt.legend()
plt.savefig("pr_complaint_detection.png")
