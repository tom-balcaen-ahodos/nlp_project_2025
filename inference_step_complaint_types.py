import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch

# Path to your saved adapter
adapter_path = "./distilbert_email_classifier_adapter_4"
model_name = "distilbert-base-uncased"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the base model with the same configuration as during training
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=6,
    id2label={0: '', 1: 'packaging', 2: 'logistics', 3: 'quality', 4: 'not applicable', 5: 'pricing errors'},
    label2id={'': 0, 'packaging': 1, 'logistics': 2, 'quality': 3, 'not applicable': 4, 'pricing errors': 5}
)

# Load the trained LoRA adapter
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()  # Set to evaluation mode

# Function to prepare input in the same format as during training
def prepare_input(subject, body):
    text = f"Subject: {subject}\nBody: {body}"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    return inputs

# Function to perform inference on a single example
def predict(subject, body):
    inputs = prepare_input(subject, body)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(predictions, dim=1).item()
        confidence = predictions[0][predicted_class].item()
    
    # Map the predicted class to its label
    class_labels = {0: '', 1: 'packaging', 2: 'logistics', 3: 'quality', 4: 'not applicable', 5: 'pricing errors'}
    predicted_label = class_labels[predicted_class]
    
    return {
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "probabilities": {class_labels[i]: predictions[0][i].item() for i in range(len(class_labels))},
    }

# Example inference on a single email
example_subject = "Production site issues - 309107 DS HB Rect (P 09/02/2023) - DS order #408"
example_body = "Dear Wouter, DS has re-packed 200 cartons products. DS says all the 200 cartons are with more or less such quality issues. DS is reliant to take this item anymore. Would you check the stocks of this item, in order to make sure no more bad quality products to be shipped? Jack Lo From: Wouter Claerhout <Wouter.Claerhout@agristo.com> Sent: Friday, December 08, 2023 4:24 PM To: taiwanfood <taiwanfood@seed.net.tw>; Beau Declerck <Beau.Declerck@agristo.com> Cc: Agristo Claims <Claims@agristo.com> Subject: RE: [Ticket:125369] RE: quality issues - 309107 DS HB Rect (P 09/02/2023) - DS order #408 Hi Jack, Further investigation of our quality just in: In the order there is one composite pallet that contains a small volume of stripped product that comes from 2 different production orders (produced March 2022 and January 2022). I see that they had quite a few quality issues in the January run that may be related to this complaint. Product was blocked each time, but it is possible that a small volume was able to continue this way. However, this should only be a small volume (max 10 CAR) and not the 75 CAR that the customer mentions in his email. In one of the photos I can see that the rostis are very sticky together. When rosti are properly frozen, they are bone hard and do not stick together. Based on the photos, it appears that they have been thawed (which can also cause deformation) and then frozen again. Are temperature registrations available?"

result = predict(example_subject, example_body)
print(f"Predicted class: {result['predicted_label']} (Class {result['predicted_class']})")
print(f"Confidence: {result['confidence']:.4f}")
print("\nClass probabilities:")
for label, prob in result['probabilities'].items():
    print(f"  {label}: {prob:.4f}")

# Example of batch inference
def batch_predict(data):
    """
    Perform inference on a batch of emails
    
    Parameters:
    data: DataFrame with 'Subject' and 'TextBody' columns
    
    Returns:
    DataFrame with predictions added
    """
    results = []
    
    for _, row in data.iterrows():
        subject = str(row['Subject'])
        body = str(row['TextBody'])
        prediction = predict(subject, body)
        results.append({
            'Subject': subject,
            'Predicted_Class': prediction['predicted_label'],
            'Confidence': prediction['confidence']
        })
    
    return pd.DataFrame(results)

# Example of loading test data and predicting on it
# Uncomment and modify as needed
'''
test_data = pd.read_csv('test_emails.csv', sep=';')
predictions = batch_predict(test_data)
print(predictions.head())
predictions.to_csv('email_predictions.csv', index=False)
'''