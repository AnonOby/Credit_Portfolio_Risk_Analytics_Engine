import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import config


class CensusProcessor:
    """
    Processes US Census Bureau S2503 (Financial Characteristics) data.
    """

    def __init__(self):
        # Key Column IDs from Census S2503 Table
        self.COL_INCOME = 'S2503_C01_013E'
        self.COL_HOUSING = 'S2503_C01_024E'

    def process_pipeline(self):
        print("🏛️ Starting Census Data Processing Pipeline...")

        # 1. Load and Clean Individual Years
        print("📂 Loading and cleaning 2022 data...")
        df_2022 = self._load_single_year(config.CENSUS_FILE_2022, suffix='_2022')

        print("📂 Loading and cleaning 2023 data...")
        df_2023 = self._load_single_year(config.CENSUS_FILE_2023, suffix='_2023')

        print("📂 Loading and cleaning 2024 data...")
        df_2024 = self._load_single_year(config.CENSUS_FILE_2024, suffix='_2024')

        # 2. Merge DataFrames on Zip Code
        print("🔗 Merging datasets on Zip Code...")
        df_merged = df_2022.merge(df_2023, on='zip_code', how='outer', suffixes=('', '_drop'))
        df_merged = df_merged.merge(df_2024, on='zip_code', how='outer', suffixes=('', '_drop'))

        # Clean up duplicate columns from merge
        df_merged = df_merged.loc[:, ~df_merged.columns.str.endswith('_drop')]

        # 3. Feature Engineering: Economic Growth Rates
        print("📈 Calculating economic growth trends...")

        # Growth 2022 -> 2023
        df_merged['income_growth_22_23'] = self._calculate_growth(
            df_merged['median_income_2023'],
            df_merged['median_income_2022']
        )

        # Growth 2023 -> 2024
        df_merged['income_growth_23_24'] = self._calculate_growth(
            df_merged['median_income_2024'],
            df_merged['median_income_2023']
        )

        # 4. Final Selection
        final_cols = [
            'zip_code',
            'median_income_2024',
            'housing_cost_2024',
            'income_growth_22_23',
            'income_growth_23_24'
        ]

        df_final = df_merged[final_cols].copy()

        # Handle NaNs
        df_final.fillna(0, inplace=True)

        print(f"✅ Census processing complete. Final shape: {df_final.shape}")
        return df_final

    def _load_single_year(self, file_path, suffix):
        """
        Loads a single year CSV, filters for Zip Codes, and selects key columns.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Census file not found: {file_path}")

        # Load CSV
        # Use low_memory=False for mixed types
        # Only load necessary columns to save RAM
        usecols = ['GEO_ID', self.COL_INCOME, self.COL_HOUSING]

        df = pd.read_csv(file_path, usecols=usecols, dtype={'GEO_ID': str})

        print(f"   -> Raw rows loaded: {len(df)}")

        # FIX: Corrected Prefix to '8600000US' (7 zeros) for Zip Code Tabulation Areas
        # Standard Census Geo ID for ZCTA is 8600000US<ZIP>
        zcta_mask = df['GEO_ID'].str.contains(r'860.*US\d{5}', regex=True)

        # Check if filter found anything
        if zcta_mask.sum() == 0:
            print("   ⚠️ WARNING: No rows matched the '8600000US' prefix.")
            print("   🐞 Debug: Showing first 5 GEO_IDs to check format:")
            print(df['GEO_ID'].head())
            # Fallback: try to find anything with 'US' just in case format varies wildly
            # (This is a safety net, usually not needed)
            zcta_mask = df['GEO_ID'].str.contains('US')
            print(f"   -> Fallback mask found {zcta_mask.sum()} rows containing 'US'.")

        df = df[zcta_mask].copy()

        print(f"   -> Rows after ZCTA filter: {len(df)}")

        # Extract Zip Code
        # Logic: split string by 'US' and take the second part
        df['zip_code'] = df['GEO_ID'].apply(lambda x: x.split('US')[1])

        # Clean numeric values
        for col in [self.COL_INCOME, self.COL_HOUSING]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Rename columns with year suffix
        rename_map = {
            self.COL_INCOME: f'median_income{suffix}',
            self.COL_HOUSING: f'housing_cost{suffix}'
        }
        df.rename(columns=rename_map, inplace=True)

        return df[['zip_code', f'median_income{suffix}', f'housing_cost{suffix}']]

    def _calculate_growth(self, current, previous):
        """Calculates percentage growth safely."""
        return np.where(
            previous != 0,
            (current - previous) / previous,
            0.0
        )

    def save_processed_data(self, df):
        """Save the final clean census data to parquet."""
        df.to_parquet(config.PROCESSED_CENSUS_FILE, index=False)
        print(f"💾 Processed Census data saved to: {config.PROCESSED_CENSUS_FILE}")


# --- Test Block ---
if __name__ == "__main__":
    processor = CensusProcessor()
    try:
        df_final = processor.process_pipeline()

        print("\n" + "=" * 40)
        print("📊 Sample of Processed Data (First 10 Rows):")
        print("=" * 40)
        print(df_final.head(10).to_string())

        print("\n" + "=" * 40)
        print("📈 Statistics for Income Growth (23-24):")
        print("=" * 40)
        print(df_final['income_growth_23_24'].describe())

        # Save automatically
        processor.save_processed_data(df_final)

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback

        traceback.print_exc()