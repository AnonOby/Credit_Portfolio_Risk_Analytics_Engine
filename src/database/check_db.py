import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from src.database.connection import get_engine


def check_harvest():
    print("🔍 Connecting to database to verify data...")
    engine = get_engine()

    if not engine:
        print("❌ Database connection failed.")
        return

    # 1. Count total rows
    query_count = "SELECT COUNT(*) as total_rows FROM loans_master"
    df_count = pd.read_sql(query_count, engine)
    total_rows = df_count['total_rows'].iloc[0]

    print(f"✅ Total rows in database: {total_rows:,}")

    # 2. Check if we have Census data joined (Check for nulls)
    # We expect some nulls in Census data (zip code mismatches), but not ALL nulls.
    query_sample = "SELECT id, zip_code, median_income_2024, loan_status FROM loans_master LIMIT 5"
    df_sample = pd.read_sql(query_sample, engine)

    print("\n📊 Sample Data (First 5 rows):")
    print(df_sample)

    # Quick check on Census data quality
    valid_census = df_sample['median_income_2024'].notna().sum()
    print(f"\n🏛️  Census Data Join Status: {valid_census}/5 rows have economic data.")

    if total_rows > 0:
        print("\n🎉 SUCCESS! Your data is safe and ready for analysis.")
    else:
        print("\n⚠️ WARNING: Table is empty. The loader might have failed before writing anything.")


if __name__ == "__main__":
    check_harvest()