import pandas as pd
import sys
import os
from sqlalchemy import text

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.database.connection import get_engine


def check_harvest():
    print("🔍 Connecting to database to verify data...")
    engine = get_engine()

    # 🔍 NEW: DIAGNOSTIC
    if engine:
        print(f"🔍 DIAGNOSIS (Check DB): Connected to -> {engine.url}")
    else:
        print("❌ Connection is None! This is the problem.")
        return

    # 🔍 NEW: DIAGNOSTIC: List tables
    print("🔍 DIAGNOSIS: Checking existing tables in this connection...")
    with engine.connect() as conn:
        result = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public';"))
        tables = [row[0] for row in result]
        print(f"🔍 DIAGNOSIS: Tables found: {tables}")

        # 🔥 EMERGENCY FIX: Try to force drop the table HERE as well
        if 'loans_master' in tables:
            print("🔨 Check_DB: Trying to force drop the table now...")
            conn.execute(text("DROP TABLE IF EXISTS loans_master;"))
            conn.commit()  # Make sure it's persisted
            print("✅ Check_DB: Table dropped by check_db.py.")

        # Re-check
        result2 = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'"))
        tables_after = [row[0] for row in result2]
        print(f"🔍 DIAGNOSIS: Tables after dropping: {tables_after}")

    # 1. Count total rows
    query_count = "SELECT COUNT(*) as total_rows FROM loans_master"
    df_count = pd.read_sql(query_count, engine)
    total_rows = df_count['total_rows'].iloc[0]

    print(f"✅ Total rows in database: {total_rows:,}")

    if total_rows > 0:
        print(
            "⚠️ WARNING: Data still exists! This confirms we are looking at a different DB or Loader didn't write anything.")
    else:
        print("✅ Table is confirmed empty. Loader didn't write anything.")


if __name__ == "__main__":
    check_harvest()