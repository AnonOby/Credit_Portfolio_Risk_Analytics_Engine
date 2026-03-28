import gc
import pandas as pd
import sys
import os
from sqlalchemy import text

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.etl.extractor import DataExtractor
from src.etl.cleaner import DataCleaner
from src.database.connection import get_engine
import config

# ---------------------------------------------------------
# TEST MODE SETTINGS
# ---------------------------------------------------------
# Set to True for quick testing (process only first N chunks)
# Set to False for full production run
TEST_MODE = True
MAX_TEST_CHUNKS = 2  # Number of chunks to process in test mode


class PortfolioDataLoader:
    """
    Orchestrates the ETL pipeline:

    1. Loads processed Census data (Master Lookup)
    2. Extracts Loan Data in chunks
    3. Cleans Loan Data
    4. Merges Loan + Census (Enrichment)
    5. Loads into PostgreSQL
    """

    def __init__(self):
        self.cleaner = DataCleaner()
        self.census_df = None
        self.engine = None

    def run(self):
        """
        Execute the full ETL pipeline.
        """
        try:
            # -------------------------------------------------
            # STEP 0: Initialize Database Connection
            # -------------------------------------------------
            print("🔌 Step 0/5: Connecting to Database...")
            self.engine = get_engine()
            if not self.engine:
                raise ConnectionError(
                    "Failed to connect to database. Check config.py and ensure PostgreSQL is running."
                )
            print("✅ Database connection established.")

            # -------------------------------------------------
            # STEP 0.5: Hard Reset Table (Clear old data)
            # -------------------------------------------------
            print("🔥 Force-dropping table to ensure clean state...")
            with self.engine.connect() as conn:
                conn.execute(text("DROP TABLE IF EXISTS loans_master;"))
                conn.commit()
            print("✅ Old table dropped. Starting fresh.")

            # -------------------------------------------------
            # STEP 1: Load Census Data into Memory
            # -------------------------------------------------
            print("📊 Step 1/5: Loading Master Census Data...")
            self._load_census_reference()

            # -------------------------------------------------
            # STEP 2: Setup Database Table (Schema Creation)
            # -------------------------------------------------
            print("🛠️ Step 2/5: Creating Database Table Schema...")
            self._reset_database_table()

            # -------------------------------------------------
            # STEP 3: Process and Load Loan Data in Chunks
            # -------------------------------------------------
            if TEST_MODE:
                msg = f"🚀 Step 3/5: Processing Loan Data (TEST MODE - {MAX_TEST_CHUNKS} chunks)..."
            else:
                msg = "🚀 Step 3/5: Processing Loan Data (PRODUCTION MODE - Full dataset)..."
            print(msg)

            self._process_loan_chunks(limit=MAX_TEST_CHUNKS)

            # -------------------------------------------------
            # STEP 4: Verify Data
            # -------------------------------------------------
            print("✅ Step 4/5: Verifying Data...")
            self._verify_upload()

            # -------------------------------------------------
            # COMPLETION SUMMARY
            # -------------------------------------------------
            print("\n" + "=" * 60)
            if TEST_MODE:
                print(f"⚡ TEST MODE COMPLETE: Processed {MAX_TEST_CHUNKS} chunks.")
                print("   Set TEST_MODE = False in loader.py for full production run.")
            else:
                print("🏆 PRODUCTION MODE COMPLETE: Full dataset processed.")
            print("=" * 60)

        except Exception as e:
            print("\n" + "=" * 60)
            print(f"❌ Pipeline Failed: {e}")
            print("=" * 60)
            print("🐛 Detailed Error Traceback:")
            import traceback
            traceback.print_exc()

    def _load_census_reference(self):
        """
        Load and aggregate Census data by 3-digit ZIP prefix.

        The Census data contains 5-digit ZIP codes, but Lending Club only provides
        the first 3 digits (e.g., '006xx'). We truncate and aggregate Census data
        to match this format, preventing one-to-many merge explosion.
        """
        if not config.PROCESSED_CENSUS_FILE.exists():
            raise FileNotFoundError(
                f"Census data not found at {config.PROCESSED_CENSUS_FILE}. "
                f"Please run census_processor.py first."
            )

        self.census_df = pd.read_parquet(config.PROCESSED_CENSUS_FILE)

        # Ensure zip_code is string
        self.census_df['zip_code'] = self.census_df['zip_code'].astype(str)

        # Truncate to 3-digit prefix
        self.census_df['zip_code'] = self.census_df['zip_code'].str[:3]

        # ---------------------------------------------------------
        # KEY FIX: Aggregate by 3-digit prefix
        # Multiple 5-digit ZIPs map to the same 3-digit prefix,
        # so we take the mean to avoid one-to-many merge explosion.
        # ---------------------------------------------------------
        # Define columns to aggregate (all numeric feature columns)
        agg_cols = [col for col in self.census_df.columns if col != 'zip_code']

        self.census_df = self.census_df.groupby('zip_code', as_index=False)[agg_cols].mean()

        # Optimize memory: downcast float64 to float32
        for col in self.census_df.select_dtypes(include=['float64']).columns:
            self.census_df[col] = pd.to_numeric(self.census_df[col], downcast='float')

        print(f"   -> Loaded {len(self.census_df)} ZIP code regions (3-digit prefix, aggregated)")

    def _reset_database_table(self):
        """
        Create the loans_master table with correct schema.
        Drops existing table if present.
        """
        # Get sample schema by processing one row
        sample_df = self._get_sample_schema()

        # Write empty DataFrame with schema to create table
        sample_df.head(0).to_sql(
            'loans_master',
            self.engine,
            if_exists='replace',
            index=False
        )
        print("   -> Table 'loans_master' schema created successfully.")

    def _get_sample_schema(self):
        """
        Create a sample DataFrame to establish SQL schema.
        Merges loan sample with census data to get all columns.
        """
        sample_df = self._get_cleaned_sample()

        # Truncate loan zip_code to 3 digits for merge
        if 'zip_code' in sample_df.columns:
            sample_df['zip_code'] = sample_df['zip_code'].astype(str).str[:3]

        # Merge with census to get final schema
        merged_sample = sample_df.merge(
            self.census_df,
            on='zip_code',
            how='left'
        )
        return merged_sample

    def _get_cleaned_sample(self):
        """
        Helper method: read and clean 1 row to establish schema.
        """
        extractor = DataExtractor(config.RAW_LOAN_DATA_FILE)

        # Read just 1 row
        sample_chunk = next(extractor.get_chunks(chunksize=1))

        # Clean using DataCleaner
        return self.cleaner.execute_pipeline(sample_chunk)

    def _process_loan_chunks(self, limit=None):
        """
        Main processing loop: Extract -> Clean -> Merge -> Load.

        Args:
            limit: Maximum number of chunks to process (for testing)
        """
        extractor = DataExtractor(config.RAW_LOAN_DATA_FILE)
        chunksize = 10000
        chunk_counter = 0

        for chunk in extractor.get_chunks(chunksize=chunksize):
            chunk_counter += 1

            # -----------------------------------------
            # 1. Clean the chunk
            # -----------------------------------------
            chunk_clean = self.cleaner.execute_pipeline(chunk)

            # -----------------------------------------
            # 2. Prepare Zip Code for Merge
            # -----------------------------------------
            if 'zip_code' in chunk_clean.columns:
                chunk_clean['zip_code'] = chunk_clean['zip_code'].astype(str).str[:3]
            else:
                # Fallback if zip_code missing
                chunk_clean['zip_code'] = '000'

            # -----------------------------------------
            # 3. Merge with Census (Enrichment)
            # -----------------------------------------
            chunk_enriched = chunk_clean.merge(
                self.census_df,
                on='zip_code',
                how='left'
            )

            # -----------------------------------------
            # 4. Load to Database
            # -----------------------------------------
            chunk_enriched.to_sql(
                'loans_master',
                self.engine,
                if_exists='append',
                index=False
            )

            print(f"   -> Chunk {chunk_counter}: Uploaded {len(chunk_enriched):,} rows")

            # -----------------------------------------
            # 5. Test Mode Break Condition
            # -----------------------------------------
            if TEST_MODE and limit and chunk_counter >= limit:
                print(f"   ✅ TEST MODE: Reached limit of {limit} chunks.")
                break

            # Garbage Collection to free memory
            gc.collect()

    def _verify_upload(self):
        """
        Query database to verify data was loaded successfully.
        """
        df_count = pd.read_sql(
            "SELECT COUNT(*) as total_rows FROM loans_master",
            self.engine
        )
        count = df_count['total_rows'].iloc[0]
        print(f"   -> Total rows in database: {count:,}")

        # Also show sample of data
        df_sample = pd.read_sql(
            "SELECT id, loan_amnt, grade, zip_code FROM loans_master LIMIT 5",
            self.engine
        )
        print("\n   📋 Sample records:")
        print(df_sample.to_string(index=False))


# ---------------------------------------------------------
# Main Execution Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    loader = PortfolioDataLoader()
    loader.run()