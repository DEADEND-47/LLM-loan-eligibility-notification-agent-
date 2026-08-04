import sys
import os
import logging
from dotenv import load_dotenv

# Load env variables from .env
load_dotenv()

# Enable verbose console logging to trace where it is executing
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Ensure root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.scheduler import AgentScheduler

def main():
    print("Initializing AgentScheduler...")
    scheduler = AgentScheduler("config.yaml")
    
    print("\nRunning a single pipeline sweep...")
    metrics = scheduler.run_pipeline()
    print(f"\nPipeline Sweep Results: {metrics}")
    
    # Check if anything was written to SQLite notifications database
    import sqlite3
    db_path = scheduler.db_path
    if os.path.exists(db_path):
        print(f"\nChecking SQLite database: {db_path}")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM notifications ORDER BY sent_at DESC LIMIT 5")
        rows = cursor.fetchall()
        print(f"Most recent rows in notifications table (max 5):")
        for r in rows:
            print(dict(r))
        conn.close()
    else:
        print(f"\nDatabase not found at: {db_path}")

if __name__ == "__main__":
    main()
