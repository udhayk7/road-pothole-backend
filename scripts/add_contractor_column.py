"""
One-time migration: add contractor_name to clusters table if missing.
Run from backend dir: python scripts/add_contractor_column.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import engine
from sqlalchemy import text

def main():
    with engine.connect() as conn:
        # Works on PostgreSQL 9.5+: add column only if it doesn't exist
        conn.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'clusters' AND column_name = 'contractor_name'
                ) THEN
                    ALTER TABLE clusters ADD COLUMN contractor_name VARCHAR;
                END IF;
            END $$;
        """))
        conn.commit()
    print("Column clusters.contractor_name ensured.")

if __name__ == "__main__":
    main()
