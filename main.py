# from utils import read_csv_file, prefix_metadata_columns, merge_datasets, count_matching_records
import pandas as pd
from IPython.display import display
import re
from langdetect import detect
from datasets import Dataset, DatasetDict, NamedSplit

def main():
    print("test")
    df = pd.read_csv('../salesforce_case_v2.csv', sep=';', dtype=str, on_bad_lines='skip', skiprows=range(1, 50000), nrows=10000) #, nrows=10000
    #dataframe is niet geindexeert


    print(len(df))
    # Print column names with their data types
    # Remove the column
    df['Name'] = df['Name'].apply(lambda x: "1") 

    df['lang'] = df['TextBody'].apply(detect_language)

    # Print unique values in 'lang' column and amount of each
    dutchLang = df['lang'].value_counts().get('nl', 0);
    englishLang = df['lang'].value_counts().get('en', 0);
    frenchLang = df['lang'].value_counts().get('fr', 0);
    spanishLang = df['lang'].value_counts().get('es', 0);
    italianLang = df['lang'].value_counts().get('it', 0);
    
    '''
    print(df['lang'].unique())
    print("total count: ", len(df))
    print("nl: ", df['lang'].value_counts().get('nl', 0), " ", df['lang'].value_counts().get('nl', 0)/len(df)*100)
    print("en: ", df['lang'].value_counts().get('en', 0), " ", df['lang'].value_counts().get('en', 0)/len(df)*100)
    print("fr: ", df['lang'].value_counts().get('fr', 0), " ", df['lang'].value_counts().get('fr', 0)/len(df)*100)
    print("es: ", df['lang'].value_counts().get('es', 0), " ", df['lang'].value_counts().get('es', 0)/len(df)*100)
    print("it: ", df['lang'].value_counts().get('it', 0), " ", df['lang'].value_counts().get('it', 0)/len(df)*100)
    '''
    data = {
    "Language": ["nl", "en", "fr", "es", "it"],
    "Count": [
        dutchLang,
        englishLang,
        frenchLang,
        spanishLang,
        italianLang
    ],
    "Percentage": [
        dutchLang/len(df)*100,
        englishLang/len(df)*100,
        frenchLang/len(df)*100,
        spanishLang/len(df)*100,
        italianLang/len(df)*100
    ]
    }

    # Create DataFrame
    df_lang = pd.DataFrame(data)
    df_lang.to_csv('language_count.csv')
    print(df['Priority'].unique())
    print(df['AGR_Type_of_Complaint__c'].unique())
    # Remove rows containing 'Automatic reply' or 'Automatisch antwoord'
    df_filtered = remove_unwanted_rows(df)

    # Simplified version create new column and set T or F.
    df_filtered['Complaint'] = df_filtered['AGR_Type_of_Complaint__c'].notna() & (df_filtered['AGR_Type_of_Complaint__c'] != '')

    # if df_filtered['TextBody'].dtype == str:
    df_filtered['TextBody'] = df_filtered['TextBody'].apply(clean_body)
        # Apply the cleaning function to TextBody
    df_filtered['TextBody'] = df_filtered['TextBody'].apply(clean_text)

    # Remove rows with missing values in 'AGR_Type_of_Complaint__c'
    df_filtered_complaints = df_filtered[df_filtered['AGR_Type_of_Complaint__c'].notna()]
    df_filtered_no_complaints = df_filtered[df_filtered['AGR_Type_of_Complaint__c'].isna()]
    
    #print complaint amount
    complaint_amount = df_filtered['Complaint'].value_counts().get(True, 0);
    print('complaint amount: ', complaint_amount)

    small_df_complaints = df_filtered_complaints[['Subject','TextBody','Complaint']].copy()
    small_df_no_complaints = df_filtered_no_complaints[['Subject','TextBody','Complaint']].copy()
    
    small_df_complaints = small_df_complaints.head(1000)
    small_df_no_complaints = small_df_no_complaints.head(1000)

    combined_df = pd.concat([small_df_complaints, small_df_no_complaints], ignore_index=True)
    #save_to_csv(df_filtered, 'output_1.csv')

    save_to_csv(df_filtered_complaints, 'filtered_complaints.csv')
    save_to_csv(df_filtered_no_complaints, 'filtered_no_complaints.csv')
    save_to_csv(combined_df, 'combined_df.csv')


    # save_to_csv(small_df_complaints, 'output_small_validation_set_model_v2.csv')
    # save_to_csv(small_df_no_complaints, 'output_small_validation_set_model_v2.csv')

    # dataset = Dataset.from_pandas(df)
    # df_train = df_filtered.sample(frac=0.8, random_state=42)
    # df_test = df_filtered.drop(df_train.index)
    # ds_train = Dataset.from_pandas(df_train)
    # ds_test = Dataset.from_pandas(df_test)
    # ds = DatasetDict()
    # ds['train'] = ds_train
    # ds['validation'] = ds_test
    # print(dataset)

def detect_language(text):
    if(type(text) == str or type(text) == object):
        lang = detect(text)
        return lang
    
# def is_complaint(text):
#     if not text:
#         return False
#     else :
#         return True


def save_to_csv(df, output_file, record_limit=3000):
    """Saves DataFrame to a CSV file with UTF-8 BOM encoding."""
    df.head(record_limit).to_csv(output_file, sep=';', index=False, encoding='utf-8-sig')
    print(f"✅ Merged file saved as: {output_file}")

def remove_unwanted_rows(df):
    # Remove rows containing 'Automatic reply' or 'Automatisch antwoord'
    df_filtered = df[~df['Name'].str.contains('Automatic reply|Automatisch antwoord|Fw:|RE:', case=False, na=False)]
    return df_filtered


def clean_body(text):
    if(type(text) == str or type(text) == object):
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
def clean_text(text):
    if(type(text) == str or type(text) == object):
        # Remove all types of newlines (including \n, \r, \r\n)
        text = re.sub(r'[\n\r]+', ' ', text)
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)        
        # Trim leading and trailing whitespace
        text = text.strip()

    return text


if __name__ == "__main__":
    main()
