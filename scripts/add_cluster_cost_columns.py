"""
One-time migration: add predicted_failure_days, estimated_repair_cost,
delayed_repair_cost, cost_savings to clusters table if missing.
Run from backend dir: python scripts/add_cluster_cost_columns.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import engine
from sqlalchemy import text

COLUMNS = [
    ("predicted_failure_days", "INTEGER"),
    ("estimated_repair_cost", "FLOAT"),
    ("delayed_repair_cost", "FLOAT"),
    ("cost_savings", "FLOAT"),
]

def main():
    with engine.connect() as conn:
        for col_name, col_type in COLUMNS:
            conn.execute(text(f"""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'clusters' AND column_name = '{col_name}'
                    ) THEN
                        ALTER TABLE clusters ADD COLUMN {col_name} {col_type};
                    END IF;
                END $$;
            """))
        conn.commit()
    print("Cluster cost columns ensured.")

if __name__ == "__main__":
    main()
