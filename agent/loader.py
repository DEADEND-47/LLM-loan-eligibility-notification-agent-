import os
import logging
import sqlite3
from typing import Optional
import pandas as pd

class DataLoader:
    """Handles Excel data loading, schema validation, and incremental record tracking using SQLite watermarks."""

    def __init__(self, config: dict) -> None:
        """Initialize the DataLoader.

        Args:
            config (dict): Global configuration settings dictionary.
        """
        self.config = config
        self.dataset_path = config["dataset"]["path"]
        self.sheet_name = config["dataset"]["sheet_name"]
        self.db_path = config["output"]["sqlite_db"]
        self.logger = logging.getLogger(__name__)

    def load_data(self) -> pd.DataFrame:
        """Load the customer loan dataset from Excel and validate the required schema.

        Raises:
            FileNotFoundError: If the configured Excel dataset is missing.
            ValueError: If the Excel file is missing required columns.

        Returns:
            pd.DataFrame: Sorted DataFrame of customer records.
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at path: {self.dataset_path}")

        df = pd.read_excel(self.dataset_path, sheet_name=self.sheet_name)

        required_columns = [
            "PROSPECTID",
            "Approved_Flag",
            "Risk_Tier",
            "Recommended_Loan_Amount",
            "Interest_Rate_Pct",
            "Tenure_Years",
            "Repayment_Method",
            "Total_Interest_Payable",
            "Total_Amount_Payable",
            "Monthly_EMI",
            "Reason_For_Approval",
            "Credit_Health_Score",
            "Income_TL_Ratio"
        ]

        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in dataset: {', '.join(missing_cols)}")

        # Sort by PROSPECTID to ensure sequential watermarking
        df = df.sort_values(by="PROSPECTID", ascending=True)
        self.logger.info(f"Successfully loaded {len(df)} rows from Excel dataset.")
        return df

    def _get_connection(self) -> sqlite3.Connection:
        """Create a thread-safe connection to the SQLite database.

        Returns:
            sqlite3.Connection: SQLite connection object.
        """
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _ensure_watermark_table(self, conn: sqlite3.Connection) -> None:
        """Ensure the watermark table exists and has a default record.

        Args:
            conn (sqlite3.Connection): Active SQLite connection.
        """
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS watermark (
                id INTEGER PRIMARY KEY,
                last_prospect_id INTEGER
            )
            """
        )
        
        # Check if record exists
        cursor.execute("SELECT COUNT(1) FROM watermark WHERE id = 1")
        if cursor.fetchone()[0] == 0:
            cursor.execute("INSERT INTO watermark (id, last_prospect_id) VALUES (1, 0)")
        
        conn.commit()

    def get_new_records(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter the input DataFrame to return only records newer than the watermarked prospect ID.

        Args:
            df (pd.DataFrame): The source DataFrame of customers.

        Returns:
            pd.DataFrame: Unprocessed customer records.
        """
        conn = self._get_connection()
        try:
            self._ensure_watermark_table(conn)
            cursor = conn.cursor()
            cursor.execute("SELECT last_prospect_id FROM watermark WHERE id = 1")
            last_prospect_id = cursor.fetchone()[0]

            filtered_df = df[df["PROSPECTID"] > last_prospect_id]
            count = len(filtered_df)
            
            if count > 0:
                self.logger.info(f"Found {count} new records to process (Watermark: {last_prospect_id}).")
            else:
                self.logger.info("No new records found. All records have been processed.")
                
            return filtered_df
        finally:
            conn.close()

    def update_watermark(self, max_id: int) -> None:
        """Update the SQLite watermark to record the latest processed PROSPECTID.

        Args:
            max_id (int): Maximum PROSPECTID value processed in the current batch.
        """
        conn = self._get_connection()
        try:
            self._ensure_watermark_table(conn)
            cursor = conn.cursor()
            cursor.execute("UPDATE watermark SET last_prospect_id = ? WHERE id = 1", (max_id,))
            conn.commit()
            self.logger.info(f"Watermark updated to PROSPECTID {max_id}")
        finally:
            conn.close()

    def sync_db_to_excel(self) -> None:
        """Read the SQLite database notifications and update the Excel file with status metadata."""
        if not os.path.exists(self.dataset_path):
            return

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='notifications'")
            if not cursor.fetchone():
                return

            db_df = pd.read_sql_query("SELECT * FROM notifications", conn)
            if db_df.empty:
                return

            sheet_name_str = self.sheet_name
            if isinstance(self.sheet_name, int):
                xl = pd.ExcelFile(self.dataset_path)
                sheet_name_str = xl.sheet_names[self.sheet_name]

            df = pd.read_excel(self.dataset_path, sheet_name=sheet_name_str)

            db_df = db_df.rename(columns={
                "notification_id": "Notification_ID",
                "sent_at": "Notification_Sent_At",
                "language": "Notification_Language",
                "status": "Notification_Status"
            })

            for col in ["Notification_ID", "Notification_Sent_At", "Notification_Language", "Notification_Status"]:
                if col in df.columns:
                    df = df.drop(columns=[col])

            merged_df = pd.merge(
                df,
                db_df[["prospect_id", "Notification_ID", "Notification_Sent_At", "Notification_Language", "Notification_Status"]],
                left_on="PROSPECTID",
                right_on="prospect_id",
                how="left"
            )

            if "prospect_id" in merged_df.columns:
                merged_df = merged_df.drop(columns=["prospect_id"])

            merged_df.to_excel(self.dataset_path, sheet_name=sheet_name_str, index=False)
            self.logger.info("Successfully synchronized SQLite notification records back to Excel dataset.")
        except Exception as e:
            self.logger.error(f"Failed to synchronize database records to Excel: {str(e)}")
        finally:
            conn.close()
