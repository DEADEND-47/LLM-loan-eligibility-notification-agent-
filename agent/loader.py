import os
import sys
import logging
import sqlite3
from typing import Optional
import pandas as pd

# Add the project root to sys.path to resolve dashboard imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dashboard import pipeline as pl
except ImportError:
    pl = None

class DataLoader:
    """Handles Excel and CSV data loading, schema validation, and incremental record tracking using SQLite watermarks."""

    def __init__(self, config: dict) -> None:
        """Initialize the DataLoader.

        Args:
            config (dict): Global configuration settings dictionary.
        """
        self.config = config
        self.dataset_path = config["dataset"].get("excel_path", config["dataset"].get("path"))
        self.sheet_name = config["dataset"].get("excel_sheet_name", config["dataset"].get("sheet_name"))
        self.db_path = config["output"]["sqlite_db"]
        self.logger = logging.getLogger(__name__)

    def load_data(self) -> pd.DataFrame:
        """Load the customer loan dataset from Excel or CSV and validate the required schema.

        Raises:
            FileNotFoundError: If the configured dataset is missing.
            ValueError: If the dataset is missing required columns.

        Returns:
            pd.DataFrame: Sorted DataFrame of customer records.
        """
        source = self.config["dataset"].get("source", "excel")
        
        if source == "finrisk_pipeline":
            csv_path = self.config["dataset"].get("csv_path", "data/processed/featured_dataset.csv")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV dataset not found at path: {csv_path}")
            
            df = pd.read_csv(csv_path)
            
            required_columns = [
                "PROSPECTID",
                "Credit_Score",
                "NETMONTHLYINCOME",
                "AGE",
                "Total_TL",
                "Tot_Active_TL",
                "Tot_Missed_Pmnt",
                "num_times_delinquent",
                "num_times_30p_dpd",
                "Age_Oldest_TL",
                "enq_L3m",
                "Time_With_Curr_Empr"
            ]
            
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in CSV dataset: {', '.join(missing_cols)}")
            
            # Sort by PROSPECTID to ensure sequential watermarking
            df = df.sort_values(by="PROSPECTID", ascending=True)
            self.logger.info(f"Successfully loaded {len(df)} rows from CSV dataset for FinRisk pipeline.")
            return df
            
        else: # original excel behavior
            excel_path = self.config["dataset"].get("excel_path", self.dataset_path)
            sheet_name = self.config["dataset"].get("excel_sheet_name", self.sheet_name)
            
            if not os.path.exists(excel_path):
                raise FileNotFoundError(f"Dataset not found at path: {excel_path}")
            
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            
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
                raise ValueError(f"Missing required columns in Excel dataset: {', '.join(missing_cols)}")
            
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
            
            # Fetch last prospect ID from watermark
            cursor.execute("SELECT last_prospect_id FROM watermark WHERE id = 1")
            last_prospect_id = cursor.fetchone()[0]
            
            self.logger.info(f"Last watermark read from SQLite: {last_prospect_id}")
            
            # If notifications table exists, check the maximum prospect_id recorded.
            # If the database and Excel disagree, treat SQLite as the source of truth.
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='notifications'")
            if cursor.fetchone():
                cursor.execute("SELECT MAX(prospect_id) FROM notifications")
                max_row = cursor.fetchone()
                db_max_id = max_row[0] if max_row and max_row[0] is not None else 0
                if db_max_id > last_prospect_id:
                    self.logger.warning(
                        f"SQLite notifications max prospect_id ({db_max_id}) is ahead of watermark ({last_prospect_id}). "
                        f"Updating watermark to {db_max_id} as source of truth."
                    )
                    cursor.execute("UPDATE watermark SET last_prospect_id = ? WHERE id = 1", (db_max_id,))
                    conn.commit()
                    last_prospect_id = db_max_id

            filtered_df = df[df["PROSPECTID"] > last_prospect_id]
            count = len(filtered_df)
            
            self.logger.info(f"Number of new records selected: {count}")
            
            if count > 0:
                self.logger.info(f"Found {count} new records to process (Watermark: {last_prospect_id}).")
                
                # If source is finrisk_pipeline, run prediction models to populate underwriting columns
                source = self.config["dataset"].get("source", "excel")
                if source == "finrisk_pipeline":
                    # OPTIMIZATION: Limit processing chunk size to avoid looping over 50,000+ records
                    limit = 100
                    if len(filtered_df) > limit:
                        self.logger.info(f"Throttling ML scoring to the first {limit} prospects to optimize performance.")
                        filtered_df = filtered_df.head(limit)
                    
                    self.logger.info("Executing FinRisk ML models and underwriting logic to score new prospects...")
                    
                    if pl is None:
                        raise ImportError("FinRisk pipeline module is not available in sys.path.")
                    
                    # Load contact numbers from Excel to match on PROSPECTID
                    excel_path = self.config["dataset"].get("excel_path", "Bank_Loan_DS_Project.xlsx")
                    contact_map = {}
                    if os.path.exists(excel_path):
                        try:
                            # Only read the two columns to save memory and time
                            excel_df = pd.read_excel(excel_path, usecols=["PROSPECTID", "Contact_No"])
                            contact_map = dict(zip(excel_df["PROSPECTID"], excel_df["Contact_No"]))
                        except Exception as e:
                            self.logger.error(f"Failed to load contact numbers from Excel: {e}")
                    
                    scored_rows = []
                    for _, row in filtered_df.iterrows():
                        row_dict = row.to_dict()
                        prospect_id = int(row_dict["PROSPECTID"])
                        
                        # Fetch contact number from Excel map, fallback to default if missing
                        contact_no = contact_map.get(prospect_id, "8810612756")
                        
                        # Construct ML model feature inputs
                        inputs = {
                            "Credit_Score": float(row_dict.get("Credit_Score", 700)),
                            "NETMONTHLYINCOME": float(row_dict.get("NETMONTHLYINCOME", 35000)),
                            "AGE": float(row_dict.get("AGE", 35)),
                            "Total_TL": float(row_dict.get("Total_TL", 5)),
                            "Tot_Active_TL": float(row_dict.get("Tot_Active_TL", 2)),
                            "Tot_Missed_Pmnt": float(row_dict.get("Tot_Missed_Pmnt", 0)),
                            "num_times_delinquent": float(row_dict.get("num_times_delinquent", 0)),
                            "num_times_30p_dpd": float(row_dict.get("num_times_30p_dpd", 0)),
                            "Age_Oldest_TL": float(row_dict.get("Age_Oldest_TL", 72)),
                            "enq_L3m": float(row_dict.get("enq_L3m", 0)),
                            "Time_With_Curr_Empr": float(row_dict.get("Time_With_Curr_Empr", 48)),
                        }
                        
                        try:
                            # Runs scikit-learn models & business calculations
                            res = pl.score_applicant(inputs)
                            
                            row_dict.update({
                                "Approved_Flag": res["tier"],
                                "Risk_Tier": res["tier"],
                                "Recommended_Loan_Amount": res["loan_amount"],
                                "Interest_Rate_Pct": res["interest_rate"],
                                "Tenure_Years": res["tenure_years"],
                                "Repayment_Method": res["repayment_method"],
                                "Total_Interest_Payable": res["total_interest"],
                                "Total_Amount_Payable": res["total_payable"],
                                "Monthly_EMI": res["monthly_emi"],
                                "Reason_For_Approval": res["reason"],
                                "Credit_Health_Score": res["credit_health_score"],
                                "Income_TL_Ratio": res["income_tl_ratio"],
                                "Contact_No": contact_no,
                                "repayment_schedule": res.get("schedule", [])
                            })
                        except Exception as score_err:
                            self.logger.error(f"Error scoring PROSPECTID {prospect_id}: {score_err}")
                            row_dict.update({
                                "Approved_Flag": "P4",
                                "Risk_Tier": "P4",
                                "Recommended_Loan_Amount": 0.0,
                                "Interest_Rate_Pct": 0.0,
                                "Tenure_Years": 0,
                                "Repayment_Method": "N/A",
                                "Total_Interest_Payable": 0.0,
                                "Total_Amount_Payable": 0.0,
                                "Monthly_EMI": 0.0,
                                "Reason_For_Approval": f"Scoring error: {score_err}",
                                "Credit_Health_Score": 0.0,
                                "Income_TL_Ratio": 0.0,
                                "Contact_No": contact_no,
                                "repayment_schedule": []
                            })
                        
                        scored_rows.append(row_dict)
                    
                    filtered_df = pd.DataFrame(scored_rows)
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
                "status": "Notification_Status",
                "channel": "Notification_Channel",
                "processing_status": "Processing_Status",
                "processing_remark": "Processing_Remark"
            })

            cols_to_drop = [
                "Notification_ID", "Notification_Sent_At", "Notification_Language",
                "Notification_Status", "Notification_Channel", "Processing_Status", "Processing_Remark"
            ]
            for col in cols_to_drop:
                if col in df.columns:
                    df = df.drop(columns=[col])

            merged_df = pd.merge(
                df,
                db_df[[
                    "prospect_id", "Notification_ID", "Notification_Sent_At", 
                    "Notification_Language", "Notification_Status", "Notification_Channel",
                    "Processing_Status", "Processing_Remark"
                ]],
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
