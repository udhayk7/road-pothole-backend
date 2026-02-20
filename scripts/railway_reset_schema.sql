-- Railway schema reset (hackathon/demo)
-- Run this against Railway Postgres when the table schema is out of date
-- (e.g. column clusters.risk_category does not exist).
-- After running, restart/redeploy the backend so Base.metadata.create_all(bind=engine) recreates tables.

-- Drop in order: reports has FK to clusters
DROP TABLE IF EXISTS reports CASCADE;
DROP TABLE IF EXISTS clusters CASCADE;
