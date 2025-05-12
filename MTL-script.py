# Multi-Task Learning (MTL) setup for complaint detection + complaint type classification
import torch
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from transformers import default_data_collator
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from datasets import Dataset
from huggingface_hub import login
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
import numpy as np

login(token = '')

print('CUDA? ', torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

url_sufix = '_200k'
version_suffix = '_v1'
extra_info = '_complaints_only_multi_task_learning'
csv_file_path = f'train_set_80_percent_complaints_only{url_sufix}.csv'
adapter_path = f"./distilbert_multi_task_learning{version_suffix}{url_sufix}{extra_info}"
confusion_matrix_fileName = f"./confusion_matrix_multi_task_learning/confusion_matrix_multi_task_learning{url_sufix}{version_suffix}{extra_info}.png"
roc_filename = f"./roc_curve_multi_task_learning/roc_curve_multi_task_learning{version_suffix}{url_sufix}{extra_info}.png"
pr_filename = f"./pr_curve_multi_task_learning/pr_curve_multi_task_learning{version_suffix}{url_sufix}{extra_info}.png"

label_map = {
    'packaging': 0,
    'logistics': 1,
    'quality': 2,
    'pricing errors': 3
}

class MultiTaskModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)

        # Binary classifier head
        self.binary_classifier = nn.Linear(self.encoder.config.hidden_size, 1)

        # Multi-class classifier head
        self.multiclass_classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(self, input_ids=None, attention_mask=None, labels_binary=None, labels_multiclass=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token representation
        pooled_output = self.dropout(pooled_output)

        logits_binary = self.binary_classifier(pooled_output).squeeze(-1)
        logits_multiclass = self.multiclass_classifier(pooled_output)

        loss = None
        if labels_binary is not None and labels_multiclass is not None:
            loss_fn_bce = nn.BCEWithLogitsLoss()
            loss_fn_ce = nn.CrossEntropyLoss()
            loss_binary = loss_fn_bce(logits_binary, labels_binary.float())
            loss_multiclass = loss_fn_ce(logits_multiclass, labels_multiclass)
            loss = loss_binary + loss_multiclass  # Equal weighting

        return {
            'loss': loss,
            'logits_binary': logits_binary,
            'logits_multiclass': logits_multiclass
        }

def compute_and_plot_confusion_matrix(preds, labels, class_names, filename):
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    return cm


def print_metrics_from_confusion_matrix(cm, class_names):
    print("\nPer-class metrics:")
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 45)

    for i, class_name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"{class_name:<15} {precision:.4f}     {recall:.4f}     {f1:.4f}")


def plot_roc_curve(labels_bin, probs, class_names, filename):
    num_classes = len(class_names)
    fpr, tpr, roc_auc = {}, {}, {}
    plt.figure(figsize=(10, 8))

    for i in range(num_classes):
        try:
            fpr[i], tpr[i], _ = roc_curve(labels_bin[:, i], probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})")
        except ValueError:
            roc_auc[i] = float('nan')
            print(f"⚠️ ROC AUC could not be calculated for class '{class_names[i]}' (possibly missing positive samples)")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print("\nPer-class ROC AUC:")
    for i in range(num_classes):
        print(f"{class_names[i]:<15}: {roc_auc[i]:.4f}")
    print(f"\nMacro-average ROC AUC: {np.nanmean(list(roc_auc.values())):.4f}")


def plot_pr_curve(labels_bin, probs, class_names, filename):
    num_classes = len(class_names)
    ap_scores = {}
    plt.figure(figsize=(10, 8))

    for i in range(num_classes):
        try:
            precision, recall, _ = precision_recall_curve(labels_bin[:, i], probs[:, i])
            ap_scores[i] = average_precision_score(labels_bin[:, i], probs[:, i])
            plt.plot(recall, precision, label=f"{class_names[i]} (AP = {ap_scores[i]:.2f})")
        except ValueError:
            ap_scores[i] = float('nan')
            print(f"⚠️ Average precision could not be calculated for class '{class_names[i]}'")

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print("\nPer-class Average Precision:")
    for i in range(num_classes):
        print(f"{class_names[i]:<15}: {ap_scores[i]:.4f}")
    print(f"\nMacro-average Precision: {np.nanmean(list(ap_scores.values())):.4f}")

def run_inference(trainer, eval_dataset):
    trainer.model.eval()
    all_preds, all_labels, all_probs = [], [], []
    eval_dataloader = trainer.get_eval_dataloader(eval_dataset)

    with torch.no_grad():
        for batch in eval_dataloader:
            inputs = {k: v.to(trainer.model.device) for k, v in batch.items()}
            if 'labels' in inputs:
                labels = inputs['labels'].detach().cpu()
            elif 'label' in inputs:
                labels = inputs['label'].detach().cpu()
            else:
                print(f"Warning: No label found in batch: {list(inputs.keys())}")
                continue

            outputs = trainer.model(**inputs)
            logits = outputs.logits.detach().cpu()
            probs = torch.nn.functional.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            print('lables', labels)
            all_preds.append(preds)
            all_labels.append(labels)
            all_probs.append(probs)

    return (
        torch.cat(all_preds),
        torch.cat(all_labels),
        torch.cat(all_probs),
    )

def evaluate_with_metrics_and_plots(trainer, eval_dataset,
                                     roc_filename=roc_filename,
                                     pr_filename=pr_filename,
                                     cm_filename=confusion_matrix_fileName):
    all_preds, all_labels, all_probs = run_inference(trainer, eval_dataset)
    class_names = list(trainer.model.config.id2label.values())

    print(f"\n📊 Classification Report:\n")
    labels = list(range(len(class_names)))  # [0, 1, 2, 3, 4]
    print(classification_report(all_labels, all_preds, labels=labels, target_names=class_names, digits=5))

    cm = compute_and_plot_confusion_matrix(all_preds, all_labels, class_names, cm_filename)
    print_metrics_from_confusion_matrix(cm, class_names)

    labels_bin = label_binarize(all_labels.numpy(), classes=list(range(len(class_names))))
    plot_roc_curve(labels_bin, all_probs.numpy(), class_names, roc_filename)
    plot_pr_curve(labels_bin, all_probs.numpy(), class_names, pr_filename)

    return cm


# Custom compute_metrics for both tasks
def compute_mtl_metrics(pred):
    from sklearn.metrics import accuracy_score, f1_score
    preds_binary = torch.sigmoid(torch.tensor(pred.predictions["logits_binary"])) >= 0.5
    preds_multiclass = torch.argmax(torch.tensor(pred.predictions["logits_multiclass"]), dim=1)

    labels_binary = pred.label_ids["labels_binary"]
    labels_multiclass = pred.label_ids["labels_multiclass"]

    metrics = {
        "accuracy_binary": accuracy_score(labels_binary, preds_binary.numpy()),
        "f1_binary": f1_score(labels_binary, preds_binary.numpy()),
        "accuracy_multiclass": accuracy_score(labels_multiclass, preds_multiclass.numpy()),
        "f1_multiclass": f1_score(labels_multiclass, preds_multiclass.numpy(), average='weighted')
    }
    return metrics

# Prepare your dataset's mapping function to return both labels
def prepare_mtl_features(example):
    example['text'] = f"Subject: {example['Subject']}\nBody: {example['TextBody']}"
    example['labels_binary'] = 1 if str(example['Complaint']).lower() == 'true' else 0
    example['labels_multiclass'] = label_map.get(str(example['Complaint_type']).lower(), 0)
    return example

# Data collator for dictionary-based labels
def custom_data_collator(features):
    from transformers import default_data_collator
    batch = default_data_collator(features)
    batch['labels_binary'] = batch.pop('labels_binary')
    batch['labels_multiclass'] = batch.pop('labels_multiclass')
    return batch

# Training step setup
def train_mtl_model():
    model_name = "distilbert-base-uncased"
    model = MultiTaskModel(model_name=model_name, num_classes=4)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load and prepare dataset
    df = pd.read_csv(csv_file_path, sep=';')
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(prepare_mtl_features)
    dataset = dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=512), batched=True)
    dataset = dataset.remove_columns(['Subject', 'TextBody', 'Complaint_type', 'Complaint', 'text', '__index_level_0__'])
    dataset.set_format(type='torch')

    split = dataset.train_test_split(test_size=0.1)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    training_args = TrainingArguments(
        output_dir=adapter_path,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy_multiclass",
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=custom_data_collator,
        compute_metrics=compute_mtl_metrics,
    )

    trainer.train()
    trainer.evaluate()
    model.save_pretrained(adapter_path)