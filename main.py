from utils import read_csv_file, prefix_metadata_columns, merge_datasets, count_matching_records


def main():
    data_folder = "data"
    metadata_file = f"{data_folder}/metadata.csv"
    emails_file = f"{data_folder}/emails.csv"

    df_metadata = read_csv_file(metadata_file)
    df_emails = read_csv_file(emails_file)

    # Print column names
    print("Emails CSV Columns:", df_emails.columns.tolist())
    print("Metadata CSV Columns:", df_metadata.columns.tolist())

    if df_metadata is None or df_emails is None:
        print("❌ Process aborted due to missing files.")
        return

    # Prefix metadata columns
    df_metadata = prefix_metadata_columns(df_metadata)

    # Print column names
    print("Emails CSV Columns:", df_emails.columns.tolist())
    print("Metadata CSV Columns:", df_metadata.columns.tolist())

    count_matching_records(df_metadata, df_emails, "AGR_Id", "RelatedToId")
    df = merge_datasets(df_emails, df_metadata, "RelatedToId", "AGR_Id")
    print(df_emails.head())


if __name__ == "__main__":
    main()
