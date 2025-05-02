import pandas as pd
from datasets import Dataset, Features, Value, ClassLabel, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding # Good practice to explicitly use data collator
)
import numpy as np
import evaluate # Import the evaluate library
from peft import LoraConfig, get_peft_model, TaskType # Import PEFT components
from huggingface_hub import login
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

login(token = '')

csv_file_path = 'train_set_small_df_filtered.csv'
adapter_path = "./distilbert_email_classifier_adapter_1_1"
confusion_matrix_fileName = "./confusion_matrix_bin_class_v1_1.png"

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
    ds['label'] = 1 if str(ds['Complaint']).lower() == 'true' else 0
    return ds


features = Features({
    'Subject': Value('string'),
    'TextBody': Value('string'),
    'Complaint': Value('bool'),
    'text': Value('string'),
    'label': ClassLabel(names=['false', 'true'])
})


# --- 4. Convert Pandas DataFrame to Hugging Face Dataset ---
try:
    # Create the Hugging Face Dataset FROM the DataFrame 'df'
    original_dataset = Dataset.from_pandas(train_dataset)
    print("Successfully converted DataFrame to Hugging Face Dataset.")
except Exception as e:
    exit()

try:
    # Apply .map to the 'original_dataset' (Hugging Face Dataset object)
    processed_dataset = original_dataset.map(
        prepare_features,
        batched=False,
        remove_columns=['Subject', 'TextBody', 'Complaint'],
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
    device_map="cpu",
    num_labels=2,
    # Add label2id and id2label for clarity, Trainer might infer but explicit is better
    id2label={0: 'false', 1: 'true'},
    label2id={'false': 0, 'true': 1}
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
    output_dir="./distilbert_lora_email_classifier_1_1",
    learning_rate=2e-4, # LoRA often benefits from slightly higher LR than full fine-tuning
    num_train_epochs=1,
    per_device_train_batch_size=8,  # Adjust based on GPU memory
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    logging_dir='./logs_lora',
    logging_steps=50, # Log more often maybe
    evaluation_strategy="epoch",
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

def evaluate_with_metrics_and_plots(trainer, eval_dataset, roc_filename="./roc_curve_1_1.png", pr_filename="./pr_curve_1_1.png"):
    # Set model to evaluation mode
    trainer.model.eval()
    
    # Create empty lists to accumulate predictions and labels
    all_preds = []
    all_labels = []
    all_probs = []  # For storing raw probabilities needed for ROC
    
    # Get data loader
    eval_dataloader = trainer.get_eval_dataloader(eval_dataset)
    
    # Disable gradient calculation for inference
    with torch.no_grad():
        for batch in eval_dataloader:
            # Move batch to same device as model
            inputs = {k: v.to(trainer.model.device) for k, v in batch.items()}
            
            # Extract labels before passing to model
            if 'labels' in inputs:
                labels = inputs['labels'].cpu()
            elif 'label' in inputs:
                labels = inputs['label'].cpu()
            else:
                print(f"Warning: Could not find labels in batch. Available keys: {list(inputs.keys())}")
                continue
            
            # Forward pass
            outputs = trainer.model(**inputs)
            logits = outputs.logits
            
            # Get softmax probabilities for ROC curve
            probs = torch.nn.functional.softmax(logits, dim=1).cpu()
            preds = torch.argmax(logits, dim=1).cpu()
            
            # Store batch results
            all_preds.append(preds)
            all_labels.append(labels)
            all_probs.append(probs)
    
    # Concatenate all batches
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)
    
    # Get class names
    class_names = list(trainer.model.config.id2label.values())
    num_classes = len(class_names)
    
    # --- CONFUSION MATRIX ---
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
    plt.close()
    
    # Calculate standard metrics
    accuracy = torch.sum(all_preds == all_labels).item() / len(all_labels)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Per-class metrics
    print("\nPer-class metrics:")
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 45)
    
    for i in range(num_classes):
        tp = cm[i, i].item()
        all_pred_as_i = torch.sum(cm[:, i]).item()
        all_actual_i = torch.sum(cm[i, :]).item()
        
        precision = tp / all_pred_as_i if all_pred_as_i > 0 else 0
        recall = tp / all_actual_i if all_actual_i > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{class_names[i]:<15} {precision:.4f}     {recall:.4f}     {f1:.4f}")
    
    # --- ROC CURVE ---
    # For binary classification, use probability of positive class (index 1)
    # If you have multi-class, consider one-vs-rest ROC curves
    positive_probs = all_probs[:, 1].numpy()  # Probabilities for the positive class
    true_labels = all_labels.numpy()
    
    # Calculate ROC curve and ROC area
    fpr, tpr, thresholds = roc_curve(true_labels, positive_probs)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(roc_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- PRECISION-RECALL CURVE (also very useful) ---
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(true_labels, positive_probs)
    average_precision = average_precision_score(true_labels, positive_probs)
    
    # Plot Precision-Recall curve
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'Precision-Recall curve (AP = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.savefig(pr_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nROC AUC: {roc_auc:.4f}")
    print(f"Average Precision: {average_precision:.4f}")
    
    return cm, roc_auc, average_precision

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
cm, roc_auc, average_precision = evaluate_with_metrics_and_plots(
    trainer, 
    eval_dataset, 
    roc_filename="./roc_curve_bin_class_v1_1_1.png",
    pr_filename="./pr_curve_bin_class_v1_1_1.png"
)

# --- Optional: Save the trained adapter ---
# After training, you can save just the trained LoRA adapter weights
model.save_pretrained(adapter_path)
print(f"LoRA adapter saved to {adapter_path}")

# To load later:
# from peft import PeftModel
# config = LoraConfig.from_pretrained(adapter_path)
# base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, id2label={0: 'false', 1: 'true'}, label2id={'false': 0, 'true': 1})
# loaded_model = PeftModel.from_pretrained(base_model, adapter_path)