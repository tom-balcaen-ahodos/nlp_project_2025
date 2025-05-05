import pandas as pd
import re
import os
from langdetect import detect

# Configuration
number_of_rows = 200000
url_suffix = '_200k'
input_file = '../salesforce_case_v2.csv'
train_file = f"train_set_80_percent_complaints_only{url_suffix}.csv"
test_file = f"test_set_20_percent_complaints_only{url_suffix}.csv"

def main():
    # Load dataset
    df = pd.read_csv(input_file, sep=';', dtype=str, on_bad_lines='skip', skiprows=range(1, 1), nrows=number_of_rows)
    print(f"Total rows loaded: {len(df)}")

    # Anonymize Name field
    df['Name'] = df['Name'].apply(lambda x: "1")

    # Filter out unwanted rows (auto replies, forwards, etc.)
    df = remove_unwanted_rows(df)

    # Keep only complaints (non-empty 'AGR_Type_of_Complaint__c')
    df = df[df['AGR_Type_of_Complaint__c'].notna() & (df['AGR_Type_of_Complaint__c'] != '')]
    df['Complaint'] = True
    df['Complaint_type'] = df['AGR_Type_of_Complaint__c']

    # Clean text body
    df['TextBody'] = df['TextBody'].apply(clean_body).apply(clean_text_of_textBody)

    # Select relevant columns
    selected_columns = ['Subject', 'TextBody', 'Priority', 'AGR_Type_of_Complaint__c', 'Complaint', 'Complaint_type']
    df = df[selected_columns]

    # 🔍 Print counts of different classes
    print("\n📊 Priority Class Distribution:")
    print(df['Priority'].value_counts(dropna=False))

    print("\n📊 Complaint Type Distribution:")
    print(df['Complaint_type'].value_counts(dropna=False))

    # Split into 80% train and 20% test
    df_train = df.sample(frac=0.8, random_state=42)
    df_test = df.drop(df_train.index)

    # Print stats
    print(f"Filtered complaints: {len(df)}")
    print(f"Train set: {len(df_train)} rows")
    print(f"Test set: {len(df_test)} rows")

    # Save
    save_to_csv(df_train, train_file)
    save_to_csv(df_test, test_file)
    print(f"✅ Train set saved: {train_file}")
    print(f"✅ Test set saved: {test_file}")

def remove_unwanted_rows(df):
    mask = ~df['Subject'].str.contains('Automatic reply|Automatisch antwoord|Fw:|RE:', case=False, na=False)
    return df[mask]

def clean_body(text):
    if isinstance(text, str):
        text = re.sub(r'ATTENTION:.*?safe\.', '', text, flags=re.DOTALL|re.IGNORECASE)
        text = re.sub(r'(Best regards.*|Kind regards.*|Regards,.*)', '', text, flags=re.DOTALL|re.IGNORECASE)
        text = re.sub(r'The information contained in this communication.*?obligation\.', '', text, flags=re.DOTALL|re.IGNORECASE)
        text = re.sub(r'(Van:.*|Verzonden:.*|Aan:.*|Onderwerp:.*)', '', text, flags=re.DOTALL|re.IGNORECASE)
        text = re.sub(r'[^\n\r]+@[^\n\r]+\n.*?\d{9,}.*', '', text, flags=re.DOTALL)
        text = re.sub(r'###\s*.*\n', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_text_of_textBody(text):
    if isinstance(text, str):
        text = re.sub(r'[\n\r]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
    return text

def save_to_csv(df, output_file):
    df.to_csv(output_file, sep=';', index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    main()
