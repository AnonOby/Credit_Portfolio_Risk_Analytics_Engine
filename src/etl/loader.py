import pandas as pd
import sys
import os
import gc
from sqlalchemy import text

# Add project root to path (Safety net if Sources Root isn't set)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import modules
# Note: If you marked 'src' as Sources Root, these imports will work perfectly
from src.etl.extractor import DataExtractor
from src.etl.cleaner import DataCleaner
from src.database.connection import get_engine
import config


class PortfolioDataLoader:
    """
    Orchestrates the ETL pipeline:
    1. Loads processed Census data (Master Lookup).
    2. Extracts Loan Data in chunks.
    3. Cleans Loan Data.
    4. Merges Loan + Census.
    5. Loads into PostgreSQL.
    """

    def __init__(self):
        self.cleaner = DataCleaner()
        self.census_df = None
        self.engine = None

    def run(self):
        """
        Execute the full pipeline.
        """
        try:
            # ---------------------------------------------------------
            # STEP 0: INITIALIZE DATABASE CONNECTION (Do this FIRST!)
            # ---------------------------------------------------------
            print("🔌 Step 0/5: Connecting to Database...")
            self.engine = get_engine()
            if not self.engine:
                raise ConnectionError(
                    "Failed to connect to database. Check config.py and ensure PostgreSQL is running.")
            print("✅ Database connection established.")

            # ---------------------------------------------------------
            # 🔍 DIAGNOSTIC: PRINT THE URL WE ARE CONNECTING TO
            # ---------------------------------------------------------
            print(f"🔍 DIAGNOSTIC: We are connected to URL -> {self.engine.url}")

            # ---------------------------------------------------------
            # STEP 0.5: HARD RESET TABLE (Clear old data)
            # ---------------------------------------------------------
            print("🔥 Force-dropping table to ensure clean state...")
            with self.engine.connect() as conn:
                conn.execute(text("DROP TABLE IF EXISTS loans_master;"))
                conn.commit()
            print("✅ Old table dropped. Starting fresh.")

            # ---------------------------------------------------------
            # 🔍 DIAGNOSTIC: CHECK WHAT TABLES EXIST NOW
            # ---------------------------------------------------------
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT table name FROM pg_tables WHERE schema name = 'public';"))
                tables = result.fetchall()
                print(f"🔍 DIAGNOSTIC: Tables currently in 'public' schema: {[t[0] for t in tables]}")

            # ---------------------------------------------------------
            # STEP 1: Load Census Data into Memory
            # ---------------------------------------------------------
            print("📊 Step 2/5: Loading Master Census Data...")
            self._load_census_reference()

            # ---------------------------------------------------------
            # STEP 2: Setup Database Table (Schema Creation)
            # ---------------------------------------------------------
            print("🛠️ Step 3/5: Resetting Database Table Schema...")
            self._reset_database_table()

            # ---------------------------------------------------------
            # STEP 3: Process and Load Loan Data in Chunks
            # ---------------------------------------------------------
            print("🚀 Step 4/5: Processing Loan Data (This takes time)...")
            self._process_loan_chunks()

            # ---------------------------------------------------------
            # STEP 4: Verify Data
            # ---------------------------------------------------------
            print("✅ Step 5/5: Verifying Data...")
            self._verify_upload()

            print("\n" + "=" * 50)
            print("🏆 ETL PIPELINE COMPLETED SUCCESSFULLY! 🏆")
            print("=" * 50)

        except Exception as e:
            print(f"\n❌ Pipeline Failed: {e}")
            import traceback
            traceback.print_exc()

    def _load_census_reference(self):
        """
        Load the processed parquet file containing economic features.
        """
        if not config.PROCESSED_CENSUS_FILE.exists():
            raise FileNotFoundError(
                f"Census data not found at {config.PROCESSED_CENSUS_FILE}. Please run census_processor.py first.")

        self.census_df = pd.read_parquet(config.PROCESSED_CENSUS_FILE)

        # Ensure zip_code is string
        self.census_df['zip_code'] = self.census_df['zip_code'].astype(str)

        # 🚀 FIX: Truncate Census Zip to first 3 digits to match Lending Club masked format (e.g., '00601' -> '006')
        self.census_df['zip_code'] = self.census_df['zip_code'].str[:3]

        # Optimize memory
        for col in self.census_df.select_dtypes(include=['float64']).columns:
            self.census_df[col] = pd.to_numeric(self.census_df[col], downcast='float')

        print(f"   -> Loaded {len(self.census_df)} Zip Code regions (Truncated to 3-digit prefix).")

    def _reset_database_table(self):
        """
        Drop the table if it exists to ensure a clean run.
        """
        # We write an empty dataframe with the correct schema to create the table
        sample_df = self._get_sample_schema()

        # if_exists='replace' will drop and recreate
        sample_df.head(0).to_sql(
            'loans_master',
            self.engine,
            if_exists='replace',
            index=False
        )
        print("   -> Table 'loans_master' is ready.")

    def _get_sample_schema(self):
        """
        Create a dummy DataFrame to establish the SQL Schema.
        """
        extractor = DataExtractor(config.RAW_LOAN_DATA_FILE)
        # Just read first 10 rows to get columns
        sample_chunk = next(extractor.get_chunks(chunksize=10))
        cleaned_sample = self.cleaner.execute_pipeline(sample_chunk)

        # Merge with census to get final schema
        if 'zip_code' in cleaned_sample.columns:
            cleaned_sample['zip_code'] = cleaned_sample['zip_code'].astype(str)

        merged_sample = cleaned_sample.merge(
            self.census_df,
            on='zip_code',
            how='left'
        )
        return merged_sample

    def _process_loan_chunks(self):
        """
        Main loop: Extract -> Clean -> Merge -> Load.
        """
        extractor = DataExtractor(config.RAW_LOAN_DATA_FILE)
        chunksize = 10000

        chunk_counter = 0

        for chunk in extractor.get_chunks(chunksize=chunksize):
            chunk_counter += 1

            # 1. Clean
            chunk_clean = self.cleaner.execute_pipeline(chunk)

            # 2. Prepare Zip Code for Merge
            if 'zip_code' in chunk_clean.columns:
                # 🚀 FIX: Truncate to first 3 digits to match Census (e.g., '967xx' -> '967')
                chunk_clean['zip_code'] = chunk_clean['zip_code'].astype(str).str[:3]
            else:
                chunk_clean['zip_code'] = '00000'

            # 3. Merge with Census (Enrichment)
            chunk_enriched = chunk_clean.merge(
                self.census_df,
                on='zip_code',
                how='left'
            )

            # 4. Load to Database
            # 🚀 OPTIMIZATION: Re-enabled 'method=multi' because we reduced chunksize to 10000.
            # This is much faster (bulk insert) and safe now.
            chunk_enriched.to_sql(
                'loans_master',
                self.engine,
                if_exists='append',
                index=False,
            )

            print(f"   -> Processed and uploaded chunk {chunk_counter} ({chunksize} rows)")

            # This prevents memory buildup and freezing
            gc.collect()

    def _verify_upload(self):
        """
        Quick query to check if data landed in DB.
        """
        # Using pandas read_sql to execute a count query
        df_count = pd.read_sql("SELECT COUNT(*) as total_rows FROM loans_master", self.engine)
        count = df_count['total_rows'].iloc[0]
        print(f"   -> Total rows in Database: {count}")


# --- Test Block ---
if __name__ == "__main__":
    loader = PortfolioDataLoader()
    loader.run()  # ⚡️ Start the engine