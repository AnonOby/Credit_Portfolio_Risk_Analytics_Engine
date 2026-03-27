import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.database.connection import get_engine
from sqlalchemy import text  # <--- 🚀 新增这行导入


def hard_reset():
    print("🔨 Performing HARD RESET on database...")
    engine = get_engine()

    if not engine:
        print("❌ Connection failed.")
        return

    try:
        with engine.connect() as conn:
            # 🚀 修改这里：用 text() 包裹 SQL 语句
            conn.execute(text("DROP TABLE IF EXISTS loans_master;"))
            conn.commit()
        print("✅ Table 'loans_master' has been DROPPED successfully.")
        print("🧹 The database is now clean.")

    except Exception as e:
        print(f"❌ Error dropping table: {e}")


if __name__ == "__main__":
    hard_reset()