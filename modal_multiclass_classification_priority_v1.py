import pandas as pd
from datasets import Dataset, Features, Value, ClassLabel, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import numpy as np
import evaluate
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import login
import torch
import matplotlib.pyplot as plt
import seaborn as sns

login(token = '')

url_sufix = '_20k'
csv_file_path = f'train_set_80_percent{url_sufix}.csv'
adapter_path = f"./distilbert_email_priority_classifier_adapter_1{url_sufix}"
confusion_matrix_fileName = f"confusion_matrix_priority_class{url_sufix}_V1.png"

try:
    # Read CSV into a DataFrame
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

# --- Define priority label mapping ---
# Map priority values to numerical labels
label_map = {
    'Medium': 0,
    'Low': 1,
    'Normal': 2,
    'Urgent': 3,
    'nan': 4,  # Missing values
    'Immediate': 5,
    'Critical': 6
}

# --- Create the combined text input and numerical labels ---
def prepare_features(ds):
    ds['text'] = "Subject: " + str(ds['Subject']) + "\nBody: " + str(ds['TextBody'])
    
    # Handle the priority field, managing missing values
    priority = str(ds['Priority']).strip()
    if priority.lower() == 'nan' or priority == '' or pd.isna(priority):
        ds['label'] = 4  # Use the code for nan/missing
    else:
        ds['label'] = label_map.get(priority, 4)  # Default to 4 if not found
    
    return ds

features = Features({
    'Subject': Value('string'),
    'TextBody': Value('string'),
    'Priority': Value('string'),
    'text': Value('string'),
    'label': ClassLabel(names=['Medium', 'Low', 'Normal', 'Urgent', 'nan', 'Immediate', 'Critical'])
})

# --- Convert Pandas DataFrame to Hugging Face Dataset ---
try:
    # Create the Hugging Face Dataset from the DataFrame
    original_dataset = Dataset.from_pandas(train_dataset)
    print("Successfully converted DataFrame to Hugging Face Dataset.")
except Exception as e:
    print(f"Error creating Dataset from DataFrame: {e}")
    exit()

try:
    # Apply .map to the Hugging Face Dataset object
    processed_dataset = original_dataset.map(
        prepare_features,
        batched=False,
        remove_columns=['Subject', 'TextBody', 'Priority'],
        # features=features
    )
    print("Successfully mapped 'prepare_features' onto the dataset.")
    print("\nProcessed Dataset Example:")
    print(processed_dataset[0])
except Exception as e:
    print(f"Error during dataset mapping: {e}")
    # Print details about the dataset structure just before mapping if needed
    print("Original dataset structure:", original_dataset)
    exit()

# --- Load Tokenizer ---
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# --- Tokenize the Dataset ---
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_dataset = processed_dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])

# Split dataset
train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# --- Define LoRA Configuration ---
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_lin", "k_lin", "v_lin"],  # Target specific layers in DistilBERT Attention
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS  # For classification task
)

# --- Load Base Model and Apply LoRA ---
# Create label mapping for the model
id2label = {0: 'Medium', 1: 'Low', 2: 'Normal', 3: 'Urgent', 4: 'nan', 5: 'Immediate', 6: 'Critical'}
label2id = {'Medium': 0, 'Low': 1, 'Normal': 2, 'Urgent': 3, 'nan': 4, 'Immediate': 5, 'Critical': 6}

# Load the base model
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    device_map="cpu",
    num_labels=7,  # Updated for 7 priority classes
    id2label=id2label,
    label2id=label2id
)

# Apply LoRA configuration to the base model
model = get_peft_model(base_model, lora_config)

# Verify trainable parameters
model.print_trainable_parameters()

# --- Prepare Trainer ---
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./distilbert_lora_email_priority_classifier_1",
    learning_rate=2e-4,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    logging_dir='./logs_lora_priority',
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Define evaluation metric
accuracy_metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions = np.argmax(predictions, axis=1)
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
    plt.figure(figsize=(12, 10))  # Slightly larger for more classes
    sns.heatmap(cm_np, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Priority Classification Confusion Matrix')
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

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# --- Start Training ---
print("Starting LoRA training for Priority classification...")
trainer.train()

# --- Evaluate the model ---
cm = evaluate_with_confusion_matrix_pytorch(trainer, eval_dataset)

# --- Save the trained adapter ---
model.save_pretrained(adapter_path)
print(f"LoRA adapter for Priority classification saved to {adapter_path}")

# How to load the model later:
"""
from peft import PeftModel
config = LoraConfig.from_pretrained(adapter_path)
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=7, 
    id2label={0: 'Medium', 1: 'Low', 2: 'Normal', 3: 'Urgent', 4: 'nan', 5: 'Immediate', 6: 'Critical'},
    label2id={'Medium': 0, 'Low': 1, 'Normal': 2, 'Urgent': 3, 'nan': 4, 'Immediate': 5, 'Critical': 6}
)
loaded_model = PeftModel.from_pretrained(base_model, adapter_path)
"""