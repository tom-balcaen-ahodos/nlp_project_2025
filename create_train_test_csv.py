# import pandas as pd
import pandas as pd
import re
from langdetect import detect

# Configuration
number_of_rows = 5000

url_sufix = '_5k'
input_file = '../salesforce_case_v2.csv'
train_file = f"train_set_80_percent{url_sufix}.csv"
test_file = f"test_set_20_percent{url_sufix}.csv"

def main():
    # Read the CSV file
    df = pd.read_csv(input_file, sep=';', dtype=str, on_bad_lines='skip', skiprows=range(1, 1), nrows=number_of_rows)
    
    print(f"Total rows: {len(df)}")
    
    # Replace name field for privacy
    df['Name'] = df['Name'].apply(lambda x: "1")
    
    # Detect language
    # df['lang'] = df['TextBody'].apply(detect_language)
    
    # Remove unwanted rows (automated replies, forwards, etc.)
    df_filtered = remove_unwanted_rows(df)
    
    # Create complaint indicator column
    df_filtered['Complaint'] = df_filtered['AGR_Type_of_Complaint__c'].notna() & (df_filtered['AGR_Type_of_Complaint__c'] != '')
    df_filtered['Complaint_type'] = df_filtered['AGR_Type_of_Complaint__c']
    
    # Clean the text fields
    df_filtered['TextBody'] = df_filtered['TextBody'].apply(clean_body)
    df_filtered['TextBody'] = df_filtered['TextBody'].apply(clean_text_of_textBody)
    
    # Select only the specified columns
    selected_columns = ['Subject', 'TextBody', 'Priority', 'AGR_Type_of_Complaint__c', 'Complaint', 'Complaint_type']
    df_selected = df_filtered[selected_columns].copy()
    
    # Split into train (80%) and test (20%) sets
    df_train = df_selected.sample(frac=0.8, random_state=42)
    df_test = df_selected.drop(df_train.index)
    
    # Print statistics
    print(f"Train set size: {len(df_train)} ({len(df_train)/len(df_selected)*100:.2f}%)")
    print(f"Test set size: {len(df_test)} ({len(df_test)/len(df_selected)*100:.2f}%)")
    print(f"Complaints in train set: {df_train['Complaint'].sum()} ({df_train['Complaint'].mean()*100:.2f}%)")
    print(f"Complaints in test set: {df_test['Complaint'].sum()} ({df_test['Complaint'].mean()*100:.2f}%)")
    
    # Save to CSV files
    save_to_csv(df_train, train_file)
    save_to_csv(df_test, test_file)
    
    print(f"✅ Train set saved as: {train_file}")
    print(f"✅ Test set saved as: {test_file}")

def detect_language(text):
    if isinstance(text, str):
        try:
            return detect(text)
        except:
            return 'unknown'
    return 'unknown'

def save_to_csv(df, output_file):
    """Saves DataFrame to a CSV file with UTF-8 BOM encoding."""
    df.to_csv(output_file, sep=';', index=False, encoding='utf-8-sig')

def remove_unwanted_rows(df):
    # Remove rows containing 'Automatic reply' or 'Automatisch antwoord'
    mask = (
        ~df['Subject'].str.contains('Automatic reply|Automatisch antwoord|Fw:|RE:', case=False, na=False)
    )
    df_filtered = df[mask]
    return df_filtered

def clean_body(text):
    if isinstance(text, str):
        # Remove external email warning
        text = re.sub(r'ATTENTION:.*?safe\.', '', text, flags=re.DOTALL|re.IGNORECASE)
        # Remove email signatures and contact information
        text = re.sub(r'(Best regards.*|Kind regards.*|Regards,.*)', '', text, flags=re.DOTALL|re.IGNORECASE)
        # Remove confidentiality notices
        text = re.sub(r'The information contained in this communication.*?obligation\.', '', text, flags=re.DOTALL|re.IGNORECASE)
        # Remove email headers and footers
        text = re.sub(r'(Van:.*|Verzonden:.*|Aan:.*|Onderwerp:.*)', '', text, flags=re.DOTALL|re.IGNORECASE)
        # Remove detailed contact information
        text = re.sub(r'[^\n\r]+@[^\n\r]+\n.*?\d{9,}.*', '', text, flags=re.DOTALL)
        # Remove form-specific headers
        text = re.sub(r'###\s*.*\n', '', text, flags=re.MULTILINE)
        # Remove extra whitespace and newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'\s+', ' ', text).strip()
    return text

# Comprehensive newline and whitespace removal function
def clean_text_of_textBody(text):
    if isinstance(text, str):
        # Remove all types of newlines (including \n, \r, \r\n)
        text = re.sub(r'[\n\r]+', ' ', text)
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)        
        # Trim leading and trailing whitespace
        text = text.strip()
    return text

if __name__ == "__main__":
    main()