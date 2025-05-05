import pandas as pd

# Configuration
input_file = 'train_set_80_percent_complaints_only_200k.csv'  # Replace with the actual path to your CSV file

def analyze_cases(file_path):
    """
    Analyzes a CSV file to count the number of different cases for 'Priority',
    'Complaint_type', and the number of non-empty values in 'Complaint'.
    Also counts non-empty values for 'Subject' and 'TextBody'.

    Args:
        file_path (str): The path to the CSV file.
    """
    try:
        df = pd.read_csv(file_path, sep=';', dtype=str, on_bad_lines='skip')
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    print("--- Analysis of Cases ---")

    # Analyze 'Priority'
    if 'Priority' in df.columns:
        priority_counts = df['Priority'].value_counts(dropna=False)
        print("\n--- Priority Cases ---")
        print(priority_counts)
    else:
        print("\nWarning: 'Priority' column not found in the file.")

    # Analyze 'Complaint_type'
    if 'Complaint_type' in df.columns:
        complaint_type_counts = df['Complaint_type'].value_counts(dropna=False)
        print("\n--- Complaint Type Cases ---")
        print(complaint_type_counts)
    else:
        print("\nWarning: 'Complaint_type' column not found in the file.")

    # Analyze 'Complaint' (assuming True/False or similar for complaint indicator)
    if 'Complaint' in df.columns:
        complaint_counts = df['Complaint'].value_counts(dropna=False)
        print("\n--- Complaint Indicator (True/False) ---")
        print(complaint_counts)
        total_complaints = df['Complaint'].astype(bool).sum()  # Count True values
        print(f"\nTotal number of complaints (where 'Complaint' is True): {total_complaints}")
    else:
        print("\nWarning: 'Complaint' column not found in the file.")

    # Analyze non-empty values in specified columns
    print("\n--- Non-Empty Values in Specific Columns ---")
    columns_to_check = ['Subject', 'TextBody', 'Priority', 'AGR_Type_of_Complaint__c', 'Complaint', 'Complaint_type']
    for col in columns_to_check:
        if col in df.columns:
            non_empty_count = df[col].astype(bool).sum() # Count non-empty strings as True
            print(f"Number of non-empty values in '{col}': {non_empty_count}")
        else:
            print(f"Warning: Column '{col}' not found.")

if __name__ == "__main__":
    analyze_cases(input_file)