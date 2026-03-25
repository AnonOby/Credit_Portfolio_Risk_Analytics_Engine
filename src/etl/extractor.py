import pandas as pd
import sys
import os
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import config

class DataExtractor:
    """
    Handles extractions of large datasets using chunking to optimize memory usage.
    Implements a Generator to yield pieces of data on demand.
    """

    def __init__(self, file_path):
        self.file_path = file_path

    def validate_file(self):
        """
        Check if the raw data file exists in the designated directory.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Data not found at: {self.file_path}\n"
                f"Please download the 'accepted_2007_to_2018Q4.csv' from Lending Club Kaggle,\n"
                f"rename it to 'lending_club_loan.csv' and place it in the '{config.RAW_DATA_DIR} folder.'"
            )
        print(f"File found: {self.file_path.name} ({self.file_path.stat().st_size / 1024 / 1024:.2f} MB)")

    def get_chunks(self, chunksize = 50000):
        """
        Generator that yields pandas DataFrames in chunks.

        Args:
            chunksize (int): Number of rows per chunk. 50,000 is a good balance for 1GB files.

        Yields:
            pd.DataFrame: A chunk of the data.
        """
        self.validate_file()

        print(f"Starting data extraction (Chunk size: {chunksize} rows...)")
        print(f"This may take a moment depending on your disk speed...")

        # Initialize tqdm progress bar (estimating total rows for visualization)
        # We pass a rough estimate or let tqdm just count iterations
        try:
            with tqdm(total = None, desc = "Reading Chunks", unit = "chunk") as pbar:

                # Create the chunk iterator
                chunk_iterator = pd.read_csv(
                    self.file_path,
                    chunksize=chunksize,
                    engine='python',  # 🚀 FIX: Force Python engine (slower but won't hang)
                    sep=',',  # 🚀 FIX: Explicit separator
                    encoding="utf-8",  # Try utf-8 first, if fails change to 'latin-1'
                    on_bad_lines="skip",  # 🚀 FIX: Skip bad lines silently
                    quoting=3  # 🚀 FIX: Ignore all quotes (safer for messy LC data)
                )

                for chunk in chunk_iterator:
                    yield chunk
                    pbar.update(1)

        except Exception as e:
            print(f"Error loading file: {e}")
            raise

    def get_columns(self):
        """
        Utility method to quickly read just the header to get column names
        without loading the whole file.
        """
        self.validate_file()
        # Read only the first row (header)
        df_sample = pd.read_csv(self.file_path, nrows=1)
        return df_sample.columns.tolist()

# --- Test Block ---
if __name__ == "__main__":
    # This block runs only when you run this file directly (for testing)
    print("--- Testing DataExtractor ---")

    try:
        extractor = DataExtractor(config.RAW_LOAN_DATA_FILE)

        # Test 1: Check if file exists (will fail if you haven't download data yet)
        extractor.validate_file()

        # Test 2: Try to read the first chunk (will fail if file is missing)
        for i, chunk in enumerate(extractor.get_chunks()):
            print(f"Received chunk {i}: Shape {chunk.shape}")
            print(f"Columns: {list(chunk.columns)[:]}...") # Print just the first 5 columns
            if i == 0: # Test only the first chunk to save time
                break

    except Exception as e:
        print(e)