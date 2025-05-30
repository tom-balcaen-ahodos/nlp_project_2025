'''Okay, let's outline how to perform inference using your fine-tuned DistilBERT model with the loaded LoRA adapters.

The key steps involve:

Loading the Base Model: Start with the original distilbert-base-uncased.

Loading the LoRA Adapter: Attach your saved adapter weights to the base model using peft.

Loading the Tokenizer: Use the exact same tokenizer you used for training.

Preparing Input Data: Format your new subject/body text the same way you did for training.

Tokenizing: Convert the text to input IDs and attention masks.

Running the Model: Pass the tokenized input to the model.

Interpreting Output: Convert the model's logits (raw scores) into the final classification ("true" or "false").

Here's the code structure:'
'''

import torch
import pandas as pd # Keep pandas if you want to load test data from CSV easily
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig # Import PeftModel for loading adapters
from datasets import Dataset, Features, Value, ClassLabel, load_dataset

# --- Configuration ---
base_model_name = "distilbert-base-uncased"
# IMPORTANT: Set this to the directory where your LoRA adapter was saved
# This is the directory you might have passed to model.save_pretrained() or the output_dir in TrainingArguments if saved there
adapter_path = "./distilbert_lora_email_classifier_4/checkpoint-1014" # <--- ADJUST THIS PATH

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Load Tokenizer ---
# Must be the same tokenizer used during training
try:
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit()

# --- 2. Load the Base Model ---
# Load the original DistilBERT model configured for sequence classification
# Ensure num_labels, id2label, and label2id match your training setup
try:
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=6,
        # id2label={0: 'false', 1: 'true'}, # Crucial for interpreting output
        # label2id={'false': 0, 'true': 1}
        id2label={0: '', 1: 'packaging', 2: 'logistics', 3: 'quality', 4: 'not applicable', 5: 'pricing errors'},
        label2id={'': 0, 'packaging': 1, 'logistics': 2, 'quality': 3, 'not applicable': 4, 'pricing errors': 5}
    )
    print("Base model loaded successfully.")
except Exception as e:
    print(f"Error loading base model: {e}")
    exit()

# --- 3. Load the LoRA Adapter ---
# Load the fine-tuned LoRA weights from the specified path and merge them into the base model
try:
    # Load the PEFT model - this attaches the adapter to the base_model
    model = PeftModel.from_pretrained(base_model, adapter_path)
    print(f"LoRA adapter loaded successfully from {adapter_path}")

    # Optional: Merge adapter weights into the base model for potentially faster inference
    # This creates a standard model with the LoRA changes baked in.
    # You might not need this step if inference speed isn't critical.
    # model = model.merge_and_unload()
    # print("LoRA adapter merged and unloaded (optional step).")

except Exception as e:
    print(f"Error loading PEFT adapter: {e}")
    print(f"Check if '{adapter_path}' contains 'adapter_model.bin' and 'adapter_config.json'")
    exit()

# --- 4. Set Model to Evaluation Mode and Move to Device ---
model.eval()  # IMPORTANT: Set model to evaluation mode (disables dropout etc.)
model.to(device) # Move the complete model (base + adapter) to the GPU/CPU
print(f"Model set to evaluation mode and moved to {device}.")


# --- 5. Inference Function ---
def classify_email(subject, body, complaint):
    """
    Classifies a single email using the loaded PEFT model.

    Args:
        subject (str): The email subject.
        body (str): The email body.

    Returns:
        str: The predicted label ('true' or 'false').
        float: The confidence score for the predicted label.
    """
    # Prepare input text in the same format as training
    text = f"Subject: {str(subject)}\nBody: {str(body)}"

    # Tokenize the text
    # padding=True, truncation=True are fine for single inference
    # return_tensors="pt" gives PyTorch tensors
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Move tokenized inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Perform inference without calculating gradients
    with torch.no_grad():
        outputs = model(**inputs)

    # Get logits (raw scores)
    logits = outputs.logits

    # Calculate probabilities using softmax
    probabilities = torch.softmax(logits, dim=-1)

    # Get the predicted class index (0 or 1)
    predicted_class_id = torch.argmax(probabilities, dim=-1).item()

    # Get the confidence score for the predicted class
    confidence_score = probabilities[0, predicted_class_id].item()

    # Map the index back to the label string ('true'/'false')
    predicted_label = model.config.id2label[predicted_class_id]

    return predicted_label, confidence_score, complaint


# --- 6. Example Usage ---
print("\n--- Running Inference Examples ---")

# # Example 1: Potentially "true" (spam/urgent based on your training)
# subject1 = "Report of Damages - INV. 90577395"
# body1 = "Dear All I hope this email finds you well. As per the email subject, we are here to report some damages found at the arrival of this container. MSCU7483777 Please, sensibilize your staff so that these kind of damages may decrease. Thank you for your daily support."
# predicted_label1, score1 = classify_email(subject1, body1)
# print(f"Email 1 - Subject: {subject1}")
# print(f"Predicted Label: {predicted_label1}")
# print(f"Confidence: {score1:.4f}\n")

# # Example 2: Potentially "false"
# subject2 = "World food / WO 755 756 757 758 / SWB"
# body2 = "Hi Ximena, Please find final SWB enclosed."
# predicted_label2, score2 = classify_email(subject2, body2)
# print(f"Email 2 - Subject: {subject2}")
# print(f"Predicted Label: {predicted_label2}")
# print(f"Confidence: {score2:.4f}\n")

# # Example 3: Another potentially "false"
# subject3 = "Your Weekly Newsletter"
# body3 = "Check out the latest updates and articles from our team this week."
# predicted_label3, score3 = classify_email(subject3, body3)
# print(f"Email 3 - Subject: {subject3}")
# print(f"Predicted Label: {predicted_label3}")
# print(f"Confidence: {score3:.4f}\n")


#--- Optional: Inference on a DataFrame ---
#If you have a CSV/DataFrame of emails to test:

csv_file_path = 'output_small_first_test.csv'
train_dataset = pd.read_csv(csv_file_path, sep=';')
original_dataset = Dataset.from_pandas(train_dataset)

# test_data = [
#     {"subject": "Test Subject 1", "body": "Test body 1..."},
#     {"subject": "Test Subject 2", "body": "Test body 2..."},
# ]
test_df = pd.DataFrame(original_dataset)

predictions = []
confidences = []
true_lable = []
for index, row in test_df.iterrows():
    label, score, truelabel = classify_email(row['Subject'], row['TextBody'], row['Complaint'])
    predictions.append(label)
    confidences.append(score)
    true_lable.append(truelabel)

test_df['predicted_label'] = predictions
test_df['confidence'] = confidences
test_df['true_label'] = true_lable
print("\n--- Inference Results on DataFrame ---")
df_1 = test_df[['predicted_label','confidence','Complaint']].copy()
print(df_1)

matches = df_1['predicted_label'] == df_1['Complaint'].astype(str).str.lower()

# 2. Sum the boolean Series. True values are treated as 1, False as 0.
match_count = matches.sum()

# --- Display the Result ---
print(f"Number of rows where 'predicted_label' matches 'Complaint': {match_count}")

# --- Optional: Calculate and Display Accuracy ---
total_rows = len(df_1)
if total_rows > 0:
    accuracy = (match_count / total_rows) * 100
    print(f"Total rows: {total_rows}")
    print(f"Accuracy: {accuracy:.2f}%")
else:
    print("DataFrame is empty.")

print(df_1['Complaint'].value_counts().get(True, 0));

'''Explanation:

Configuration: Set the base model name and, crucially, the adapter_path pointing to the directory where your trained LoRA weights (adapter_model.bin or .safetensors, adapter_config.json, etc.) are saved.

Load Tokenizer/Base Model: Same as before, ensuring the id2label and label2id mappings are correct in the model config.

Load PEFT Adapter: PeftModel.from_pretrained(base_model, adapter_path) takes the loaded base model and the path to the adapter weights. It returns a PeftModel object which combines both.

model.eval(): This is critical for inference. It turns off layers like Dropout that behave differently during training and evaluation.

model.to(device): Moves the model (including adapters) to the appropriate device (GPU or CPU).

classify_email Function:

Takes subject and body as input.

Formats the text exactly as done during training ("Subject: ...\nBody: ...").

Tokenizes the text using tokenizer, ensuring return_tensors="pt".

Moves the tokenized inputs to the same device as the model.

Uses with torch.no_grad(): to disable gradient calculations, saving memory and speeding up inference.

Calls model(**inputs) to get the model's output.

Extracts logits (the raw output scores for each class).

Applies torch.softmax to convert logits into probabilities that sum to 1.

Uses torch.argmax to find the index (0 or 1) with the highest probability.

Retrieves the corresponding label string ('true' or 'false') using model.config.id2label.

Returns the label and the confidence score.

Example Usage: Demonstrates how to call the function with new email content.

Remember to replace "./distilbert_lora_email_classifier_csv" with the actual path where your adapter weights are stored.'
'''