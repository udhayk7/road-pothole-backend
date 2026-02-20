"""
One-time migration: add confidence_score to reports table if missing.
Run from backend dir: python scripts/add_confidence_score_column.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import engine
from sqlalchemy import text

def main():
    with engine.connect() as conn:
        conn.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'reports' AND column_name = 'confidence_score'
                ) THEN
                    ALTER TABLE reports ADD COLUMN confidence_score FLOAT;
                END IF;
            END $$;
        """))
        conn.commit()
    print("Column reports.confidence_score ensured.")

if __name__ == "__main__":
    main()
