import pandas as pd
from datasets import Dataset
import re

def clean_text(text):
    """
    Clean and preprocess text by:
    - Removing extra whitespaces
    - Removing special characters
    - Converting to lowercase
    """
    if not isinstance(text, str):
        return ""
    # Remove special characters and extra whitespaces
    text = re.sub(r'[^\w\s]', '', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def create_classification_dataset(csv_path):
    """
    Prepare a dataset for SFT training focused on root cause classification
    
    Args:
        csv_path (str): Path to the CSV file
    
    Returns:
        Dataset: Hugging Face dataset formatted for SFT training
    """
    # Read the CSV file
    df = pd.read_csv(csv_path, sep=';', encoding='utf-8')
    
    # Prepare the dataset entries
    processed_data = []
    
    for _, row in df.iterrows():
        # Combine and clean text body and subject
        input_text = clean_text(f"{row['Subject']} {row['TextBody']}")
        
        # Prepare the target information
        root_cause = clean_text(str(row['AGR_Root_Cause__c']))
        extra_info = clean_text(str(row['AGR_Extra_Information_Root_Cause__c']))
        
        # Create a structured instruction for the model
        instruction = f"""Analyze the following complaint text and identify:
1. The root cause of the complaint
2. Additional contextual information

Input Text: {input_text}

Provide your analysis focusing on:
- Precise root cause classification
- Relevant extra information"""
        
        # Combine root cause and extra information as the output
        output = f"""Root Cause: {root_cause}
Additional Information: {extra_info}"""
        
        processed_data.append({
            "instruction": instruction,
            "output": output
        })
    
    # Convert to Hugging Face Dataset
    return Dataset.from_list(processed_data)

# Usage example
try:
    dataset = create_classification_dataset('output_1.csv')
    print(f"Dataset created with {len(dataset)} entries")
    dataset.to_json('output_dataset.json')

    # Optional: Preview the first few entries
    # for i in range(min(len(dataset))):
    #     print(f"\nEntry {i+1}:")
    #     print("Instruction:", dataset[i]['instruction'])
    #     print("Output:", dataset[i]['output'])
except Exception as e:
    print(f"Error creating dataset: {e}")