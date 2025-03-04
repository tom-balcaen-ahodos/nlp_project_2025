import chardet
import pandas as pd

from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # Zorgt voor consistente detectie

import re
# Ensure Pandas displays all columns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)  # Set a wider display
pd.set_option('display.expand_frame_repr', False)  # Prevent column wrapping


def detect_file_encoding(file_path, sample_size=100000):
    """
    Detects the encoding of a given file by reading a sample of its content.

    :param file_path: Path to the file.
    :param sample_size: Number of bytes to read for detection (default: 100,000 bytes).
    :return: Detected encoding and confidence level.
    """
    with open(file_path, "rb") as f:
        result = chardet.detect(f.read(sample_size))

    encoding = result.get("encoding", "utf-8")
    confidence = result.get("confidence", 0)

    print(f"Detected Encoding: {encoding} (Confidence: {confidence})")
    return encoding, confidence


def read_csv_file(file_path):
    """Reads a CSV file with UTF-8 encoding and returns a DataFrame."""
    try:
        return pd.read_csv(file_path, sep=';', dtype=str, encoding='utf-8', low_memory=False)
    except FileNotFoundError as fnf_error:
        print(f"❌ File not found: {fnf_error}")
        return None
    except pd.errors.ParserError as parse_error:
        print(f"❌ Error parsing CSV file: {parse_error}")
        return None


def prefix_metadata_columns(metadata_df):
    """Prefixes metadata columns with 'AGR_' if not already prefixed."""
    return metadata_df.rename(columns=lambda col: col if col.startswith("AGR_") else f"AGR_{col}")


def merge_datasets(emails_df, metadata_df, left_on="ParentId", right_on="AGR_Id"):
    """Merges emails DataFrame with metadata DataFrame on ParentId -> AGR_Id."""
    return emails_df.merge(metadata_df, left_on=left_on, right_on=right_on, how="left")


def save_to_csv(df, output_file, record_limit=100):
    """Saves DataFrame to a CSV file with UTF-8 BOM encoding."""
    df.head(record_limit).to_csv(output_file, sep=';', index=False, encoding='utf-8-sig')
    print(f"✅ Merged file saved as: {output_file}")


def detect_language(text):
    """Detecteert de taal van een tekstblok."""
    try:
        return detect(text)
    except:
        return "unknown"


def split_and_detect_languages(text):
    """Splits de e-mail per zin en detecteert de taal per sectie."""
    sentences = text.split(". ")
    language_map = {sent: detect_language(sent) for sent in sentences}
    return language_map


def filter_dutch_text(text):
    sentences = text.split(". ")
    return ". ".join([sent for sent in sentences if detect_language(sent) == "nl"])


# Bekende disclaimers en meldingen
DISCARD_PATTERNS = [
    r"ATTENTION: This is an external email.*?safe\.",  # Security waarschuwing
    r"Please consider the environment before printing this email\.",  # Milieu disclaimer
    r"This message and any attachments are intended only for the addressee.*?",  # Standaard e-mail disclaimer
    r"https?://checkpoint\.url-protection\.com/[^ ]+",  # Checkpoint URL-beveiliging
]


def clean_email_body(text):
    """Verwijdert standaard disclaimers, waarschuwingen en beveiligings-URL's uit de e-mail body."""
    if not isinstance(text, str):
        return ""

    # Verwijder de ongewenste patronen
    for pattern in DISCARD_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

    # Extra cleanup: spaties, meerdere nieuwe regels naar 1 regel
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def extract_original_url(text):
    """Zoekt en vervangt beveiligings-URL's met de originele links."""
    return re.sub(r'https?://checkpoint\.url-protection\.com/v1/url\?o=([^&]+)&.*?', r'\1', text)


def preprocess_email_body(text):
    """Schoont de e-mail body op en behoudt enkel relevante Nederlandse content."""
    text = clean_email_body(text)  # Stap 1: verwijder disclaimers
    text = extract_original_url(text)  # Stap 2: herstel beveiligings-URL's
    text = filter_dutch_text(text)  # Stap 3: behoud Nederlandse content
    return text


def count_matching_records(df_metadata, df_emails, fieldName1, fieldName2):
    """
    Counts how many records in df_metadata have a matching record in df_emails.

    Parameters:
    - df_metadata: Pandas DataFrame containing metadata records.
    - df_emails: Pandas DataFrame containing email records.
    - fieldName1: Column name in df_metadata to match.
    - fieldName2: Column name in df_emails to match.

    Returns:
    - match_count: Number of matching records
    """

    # Ensure column names exist
    if fieldName1 not in df_metadata.columns or fieldName2 not in df_emails.columns:
        print(f"❌ Error: One of the specified fields ({fieldName1}, {fieldName2}) does not exist in the dataframes.")
        return None

    # Count matching records
    match_count = df_metadata[fieldName1].isin(df_emails[fieldName2]).sum()

    # Print results
    print(
        f"🔍 Matching records between '{fieldName1}' (metadata) and '{fieldName2}' (emails): {match_count}/{len(df_metadata)}")

    return match_count
