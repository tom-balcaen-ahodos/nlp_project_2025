import argparse
import os
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from datasets import Dataset
from sklearn.preprocessing import label_binarize

model_prefix = 'multiclass_priority'
version_suffix = '_v2'
filepath_suffix = '_200k'
adapter_path = "./distilbert_email_priority_classifier_adapter_v2_200k"
model_path = f"./distilbert_lora_email_priority_classifier{version_suffix}/checkpoint_30807"
# confusion_matrix_fileName = f"./confusion_matrix_bin_class{version_suffix}{filepath_suffix}.png"
# roc_filename = f"./roc_curve_inf_bin_class{version_suffix}_150_train{filepath_suffix}.png"à
# pr_filename = f"./pr_curve_inf_bin_class{version_suffix}_150_train{filepath_suffix}.png"
csv_file_path = f'./test_set_20_percent{filepath_suffix}.csv'
# --- Argument Parser ---
parser = argparse.ArgumentParser(description='Email Classifier Inference')
parser.add_argument('--test_file', type=str, default=csv_file_path, 
                        help='Path to the test CSV file')
parser.add_argument('--model_path', type=str, default=model_path, 
                    help='')
parser.add_argument('--output_dir', type=str, default=f'./evaluation_results_multiclass_priority{model_prefix}{version_suffix}', 
                    help='Directory to save evaluation results')
args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

# --- Setup ---
model_name = "distilbert-base-uncased"
id2label = {
    0: 'Medium', 1: 'Low', 2: 'Normal', 3: 'Urgent',
    4: 'nan', 5: 'Immediate', 6: 'Critical'
}
label2id = {v: k for k, v in id2label.items()}
class_names = list(id2label.values())
num_classes = len(class_names)

# --- Load Tokenizer and Model ---
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_classes,
    id2label=id2label,
    label2id=label2id
)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --- Load and Prepare Dataset ---
df = pd.read_csv(args.test_file, sep=';')
df = df.fillna({'Priority': 'nan'})
df['text'] = "Subject: " + df['Subject'].astype(str) + "\nBody: " + df['TextBody'].astype(str)
df['label'] = df['Priority'].map(label2id).fillna(label2id['nan']).astype(int)

dataset = Dataset.from_pandas(df[['text', 'label']])
tokenized = dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding=True, max_length=512), batched=True)
tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# --- Inference ---
all_logits = []
all_labels = []

with torch.no_grad():
    for batch in torch.utils.data.DataLoader(tokenized, batch_size=32):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

# --- Evaluation ---
logits = torch.cat(all_logits)
labels = torch.cat(all_labels)
probs = torch.nn.functional.softmax(logits, dim=1)
preds = torch.argmax(probs, dim=1)

# Confusion Matrix
cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
plt.close()

# Classification Report
report = classification_report(labels, preds, target_names=class_names, digits=4)
with open(os.path.join(args.output_dir, 'classification_report.txt'), 'w') as f:
    f.write(report)
print(report)

# ROC Curve (one-vs-rest)
labels_bin = label_binarize(labels, classes=list(range(num_classes)))
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(labels_bin[:, i], probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'roc_curves.png'))
plt.close()

# Precision-Recall Curves
plt.figure(figsize=(10, 8))
for i in range(num_classes):
    precision, recall, _ = precision_recall_curve(labels_bin[:, i], probs[:, i])
    plt.plot(recall, precision, label=f'{class_names[i]}')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'precision_recall_curves.png'))
plt.close()

print(f"\n✅ Evaluation results saved in: {args.output_dir}")
