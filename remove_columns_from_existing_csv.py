import pandas as pd

def remove_lines_from_csv(file_path, condition_column, condition_value, output_file_path):
    """
    Removes lines from a CSV file where the value in a specified column matches a given condition, and saves the result to a new CSV file.

    Args:
        file_path (str): The path to the CSV file to read from.
        condition_column (str): The name of the column to check the condition against.
        condition_value (str): The value to compare against in the specified column.
        output_file_path (str): The path to the CSV file to write the filtered data to.
    """
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path, sep=';', encoding='utf-8')

        # Print the number of rows before filtering
        print(f"Number of rows before filtering: {len(df)}")

        # Filter the DataFrame to keep only the rows where the specified column does not equal the specified value
        df_filtered = df[df[condition_column] != condition_value]

        # Print the number of rows after filtering
        print(f"Number of rows after filtering: {len(df_filtered)}")

        # Save the filtered DataFrame to a new CSV file
        df_filtered.to_csv(output_file_path, sep=';', index=False, encoding='utf-8')

        print(f"Successfully removed rows where '{condition_column}' is '{condition_value}' and saved the filtered data to '{output_file_path}'.")

    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'. Please check the file path and try again.")
    except KeyError as e:
        print(f"Error: Column '{condition_column}' not found in the CSV file. Please check the column name and try again.  KeyError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("Please check the file and data, and try again.")

if __name__ == "__main__":
    file_path = "train_set_80_percent_complaints_only_200k.csv"  # Replace with the actual path to your CSV file
    condition_column = "Complaint_type"
    condition_value = "Not applicable"
    output_file_path = "train_set_80_percent_complaints_only_200k_without_not_applicable.csv" # Added output file path

    # Call the function to remove lines from the CSV file and save to a new file
    remove_lines_from_csv(file_path, condition_column, condition_value, output_file_path)
    print("Script execution completed.")
    input("Press Enter to close the script...")
