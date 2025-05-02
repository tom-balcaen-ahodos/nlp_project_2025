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

login(token = '')

csv_file_path = 'train_set_small_df_filtered.csv'
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
    print(f"Error creating Dataset from DataFrame: {e}")
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
    output_dir="./distilbert_lora_email_classifier_3",
    learning_rate=2e-4, # LoRA often benefits from slightly higher LR than full fine-tuning
    num_train_epochs=3,
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

# --- Optional: Save the trained adapter ---
# After training, you can save just the trained LoRA adapter weights
adapter_path = "./distilbert_email_classifier_adapter_3"
model.save_pretrained(adapter_path)
print(f"LoRA adapter saved to {adapter_path}")

# To load later:
# from peft import PeftModel
# config = LoraConfig.from_pretrained(adapter_path)
# base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, id2label={0: 'false', 1: 'true'}, label2id={'false': 0, 'true': 1})
# loaded_model = PeftModel.from_pretrained(base_model, adapter_path)