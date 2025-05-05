import pandas as pd
from datasets import Dataset, Features, Value, ClassLabel, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    IntervalStrategy,
    DataCollatorWithPadding # Good practice to explicitly use data collator
)
import numpy as np
import evaluate # Import the evaluate library
from peft import LoraConfig, get_peft_model, TaskType # Import PEFT components
from huggingface_hub import login
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize

login(token = '')

print('CUDA? ', torch.cuda.is_available())
print("TrainingArguments comes from:", TrainingArguments.__module__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

url_sufix = '_200k'
version_suffix = '_v1'
extra_info = '_complaints_only'
csv_file_path = f'train_set_80_percent_complaints_only{url_sufix}.csv'
adapter_path = f"./distilbert_email_complaint_type_classifier_adapter{version_suffix}{url_sufix}{extra_info}"
confusion_matrix_fileName = f"./confusion_matrix_complaint_type_class/confusion_matrix_complaint_type_class{url_sufix}{version_suffix}{extra_info}.png"
roc_filename = f"./roc_curve_dl_train/roc_curve_roberta_complaint_type_multi_class{version_suffix}{url_sufix}{extra_info}.png"
pr_filename = f"./pr_curve_dl_train/pr_curve_roberta_complaint_type_multi_class{version_suffix}{url_sufix}{extra_info}.png"

try:
    # Read CSV into a DataFrame named 'df' (or choose another name)
    train_dataset = pd.read_csv(csv_file_path, sep=';')
    print(f"Successfully loaded {csv_file_path} into DataFrame")
    print("DataFrame Columns:", train_dataset.columns)
except FileNotFoundError:
    print(f"Error: CSV file not found at {csv_file_path}")
    exit()
except pd.errors.ParserError as e:
     print(f"ParserError loading CSV: {e}. Check delimiter and quotes.")
     exit()
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# --- 2. Create the combined text input and numerical labels ---
def prepare_features(ds):
    ds['text'] = "Subject: " + str(ds['Subject']) + "\nBody: " + str(ds['TextBody'])
    # ds['label'] = 1 if str(ds['Complaint']).lower() == 'true' else 0
    complaint_type = str(ds['Complaint_type']).lower()
    ds['label'] = label_map.get(complaint_type, 0)
    return ds

label_map = {
    'packaging': 0, 
    'logistics': 1, 
    'quality': 2, 
    'not applicable': 3, 
    'pricing errors': 4
}


features = Features({
    'Subject': Value('string'),
    'TextBody': Value('string'),
    'Complaint': Value('bool'),
    'text': Value('string'),
    'label': ClassLabel(names=['Packaging', 'Logistics', 'Quality', 'Not applicable', 'Pricing errors'])
})


# --- 4. Convert Pandas DataFrame to Hugging Face Dataset ---
try:
    # Create the Hugging Face Dataset FROM the DataFrame 'df'
    original_dataset = Dataset.from_pandas(train_dataset)
    print("Successfully converted DataFrame to Hugging Face Dataset.")
except Exception as e:
    print(f"Error creating Dataset from DataFrame: {e}")
    exit()

try:
    # Apply .map to the 'original_dataset' (Hugging Face Dataset object)
    processed_dataset = original_dataset.map(
        prepare_features,
        batched=False,
        remove_columns=['Subject', 'TextBody', 'Complaint_type'],
        #features=features
    )
    print("Successfully mapped 'prepare_features' onto the dataset.")
    print("\nProcessed Dataset Example:")
    print(processed_dataset[0])
except Exception as e:
    print(f"Error during dataset mapping: {e}")
    # Print details about the dataset structure just before mapping if needed
    print("Original dataset structure:", original_dataset)
    exit()

# --- 3. Load Tokenizer ---
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# --- 4. Tokenize the Dataset ---
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512) # Let DataCollator handle padding

tokenized_dataset = processed_dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])

# Split dataset (example) - DO THIS PROPERLY for real training
train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# --- 5. Define LoRA Configuration ---
# **Important:** Adjust target_modules for DistilBERT
# Common layers to target in BERT-like models are query, key, value projections
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_lin", "k_lin", "v_lin"], # Target specific layers in DistilBERT Attention
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS # CRITICAL: Ensures classification head is usable and trainable
)

# --- 6. Load Base Model and Apply LoRA ---
# Load the base model *without* the PEFT modifications first
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=5,
    id2label={0: 'packaging', 1: 'logistics', 2: 'quality', 3: 'not applicable', 4: 'pricing errors'},
    label2id={'packaging': 0, 'logistics': 1, 'quality': 2, 'not applicable': 3, 'pricing errors': 4}
    # Add label2id and id2label for clarity, Trainer might infer but explicit is better
    # id2label={0: 'false', 1: 'true'},
    # label2id={'false': 0, 'true': 1}
)

# Apply LoRA configuration to the base model
model = get_peft_model(base_model, lora_config)

# **Verify:** Print trainable parameters to see the effect of LoRA
model.print_trainable_parameters()
# Expected output: trainable params: ... || all params: ... || trainable%: significantly less than 100%

# --- 7. Prepare Trainer ---
# Data collator handles dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define Training Arguments
# (Using smaller batch size as an example, adjust based on your GPU)
training_args = TrainingArguments(
    output_dir=adapter_path,
    learning_rate=2e-4, # LoRA often benefits from slightly higher LR than full fine-tuning
    num_train_epochs=3,
    per_device_train_batch_size=8,  # Adjust based on GPU memory
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    logging_dir='./logs_lora',
    logging_steps=50, # Log more often maybe
    eval_strategy=IntervalStrategy.EPOCH,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy", # Define a metric if using load_best_model_at_end
    # push_to_hub=False, # Set to True to upload model/adapter
)

# Define evaluation metric (example: accuracy)
# Load the metric using evaluate.load()
accuracy_metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Ensure predictions are derived correctly (usually argmax for classification)
    if isinstance(predictions, tuple): # Handle cases where model outputs more than just logits
        predictions = predictions[0]
    predictions = np.argmax(predictions, axis=1)

    # Use the loaded metric object's compute method
    return accuracy_metric.compute(predictions=predictions, references=labels)

def evaluate_with_confusion_matrix_pytorch(trainer, eval_dataset):
    # Set model to evaluation mode
    trainer.model.eval()
    
    # Create empty tensors to accumulate predictions and labels
    all_preds = []
    all_labels = []
    
    # Get data loader
    eval_dataloader = trainer.get_eval_dataloader(eval_dataset)
    
    # Debug: Print the keys in the first batch to see the actual structure
    first_batch = next(iter(eval_dataloader))
    print("Batch keys:", first_batch.keys())
    
    # Disable gradient calculation for inference
    with torch.no_grad():
        for batch in eval_dataloader:
            # Move batch to same device as model
            inputs = {k: v.to(trainer.model.device) for k, v in batch.items()}
            
            # Extract labels before passing to model (don't pop labels)
            if 'labels' in inputs:
                labels = inputs['labels'].cpu()
            elif 'label' in inputs:
                labels = inputs['label'].cpu()
            else:
                # If your dataset has a different structure, find the correct key
                print(f"Warning: Could not find labels in batch. Available keys: {list(inputs.keys())}")
                continue
            
            # Forward pass
            outputs = trainer.model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu()
            
            # Store batch results
            all_preds.append(preds)
            all_labels.append(labels)
    
    # Concatenate all batches
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # Get class names
    class_names = list(trainer.model.config.id2label.values())
    num_classes = len(class_names)
    
    # Create confusion matrix
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(all_labels, all_preds):
        cm[t.item(), p.item()] += 1
    
    # Convert to numpy for plotting
    cm_np = cm.numpy()
    
    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_np, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(confusion_matrix_fileName, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate accuracy, precision, recall, F1 manually
    accuracy = torch.sum(all_preds == all_labels).item() / len(all_labels)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Per-class metrics
    print("\nPer-class metrics:")
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 45)
    
    for i in range(num_classes):
        # True positives: diagonal elements
        tp = cm[i, i].item()
        # All predicted as this class (column sum)
        all_pred_as_i = torch.sum(cm[:, i]).item()
        # All actual of this class (row sum)
        all_actual_i = torch.sum(cm[i, :]).item()
        
        precision = tp / all_pred_as_i if all_pred_as_i > 0 else 0
        recall = tp / all_actual_i if all_actual_i > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{class_names[i]:<15} {precision:.4f}     {recall:.4f}     {f1:.4f}")
    
    return cm

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

# Initialize the standard Trainer with the PEFT model
trainer = Trainer(
    model=model, # Pass the PEFT model
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics, # Add the compute_metrics function
)

# --- 8. Start Training ---
print("Starting LoRA training...")
trainer.train()

# --- 9. Evaluate the model ---
# cm = evaluate_with_confusion_matrix_pytorch(trainer, eval_dataset)
cm = evaluate_with_metrics_and_plots(trainer, eval_dataset)

# --- Optional: Save the trained adapter ---
# After training, you can save just the trained LoRA adapter weights
model.save_pretrained(adapter_path)
print(f"LoRA adapter saved to {adapter_path}")

# To load later:
# from peft import PeftModel
# config = LoraConfig.from_pretrained(adapter_path)
# base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, id2label={0: 'false', 1: 'true'}, label2id={'false': 0, 'true': 1})
# loaded_model = PeftModel.from_pretrained(base_model, adapter_path)