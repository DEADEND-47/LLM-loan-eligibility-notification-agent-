import os
import sqlite3
import pytest
import pandas as pd
from agent.loader import DataLoader

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Fixture returning a valid customer dataframe for test usage."""
    data = {
        "PROSPECTID": [1, 2, 3],
        "Approved_Flag": ["P1", "P2", "P3"],
        "Risk_Tier": ["P1", "P2", "P1"],
        "Recommended_Loan_Amount": [1000000, 500000, 200000],
        "Interest_Rate_Pct": [8.5, 10.5, 8.5],
        "Tenure_Years": [5, 4, 5],
        "Repayment_Method": ["EMI Fixed Monthly"] * 3,
        "Total_Interest_Payable": [230000, 115000, 46000],
        "Total_Amount_Payable": [1230000, 615000, 246000],
        "Monthly_EMI": [20500, 12812, 4100],
        "Reason_For_Approval": [
            "Strong credit score; No missed payments",
            "Stable income; No delinquency",
            "Insufficient score"
        ],
        "Credit_Health_Score": [380.0, None, 250.0],
        "Income_TL_Ratio": [25000, 15000, 5000]
    }
    return pd.DataFrame(data)

@pytest.fixture
def temp_config(tmp_path, sample_df) -> dict:
    """Fixture creating temp Excel files and returning test configurations."""
    excel_path = tmp_path / "Bank_Loan_DS_Project.xlsx"
    db_path = tmp_path / "test_notifications.db"
    
    # Save the dataframe to the mock excel path
    sample_df.to_excel(excel_path, index=False)
    
    return {
        "dataset": {
            "path": str(excel_path),
            "sheet_name": 0
        },
        "output": {
            "sqlite_db": str(db_path),
            "log_csv": str(tmp_path / "notifications_log.csv")
        }
    }

def test_load_data_success(temp_config):
    """Verify that DataLoader loads and sorts standard Excel data successfully."""
    loader = DataLoader(temp_config)
    df = loader.load_data()
    assert len(df) == 3
    assert "PROSPECTID" in df.columns
    assert "Approved_Flag" in df.columns
    assert list(df["PROSPECTID"]) == [1, 2, 3]

def test_load_data_missing_file(temp_config):
    """Verify loading from a missing file throws FileNotFoundError."""
    temp_config["dataset"]["path"] = "nonexistent_file.xlsx"
    loader = DataLoader(temp_config)
    with pytest.raises(FileNotFoundError):
        loader.load_data()

def test_get_new_records_all_new(temp_config, sample_df):
    """Verify watermark defaults to 0 and all rows are returned as new."""
    loader = DataLoader(temp_config)
    new_records = loader.get_new_records(sample_df)
    assert len(new_records) == 3

def test_get_new_records_partial(temp_config, sample_df):
    """Verify watermark filters records accurately when partially completed."""
    loader = DataLoader(temp_config)
    loader.update_watermark(1)
    
    new_records = loader.get_new_records(sample_df)
    assert len(new_records) == 2
    assert list(new_records["PROSPECTID"]) == [2, 3]

def test_update_watermark(temp_config):
    """Verify database is updated with correct watermark values."""
    loader = DataLoader(temp_config)
    loader.update_watermark(3)
    
    # Open sqlite directly to inspect
    conn = sqlite3.connect(temp_config["output"]["sqlite_db"])
    cursor = conn.cursor()
    cursor.execute("SELECT last_prospect_id FROM watermark WHERE id = 1")
    val = cursor.fetchone()[0]
    conn.close()
    
    assert val == 3

def test_recover_db_from_excel(temp_config):
    """Verify database recovery from Excel processed columns behaves correctly."""
    loader = DataLoader(temp_config)
    
    # 1. Modify the mock excel file to simulate processed status columns
    df = pd.read_excel(loader.dataset_path)
    df["Notification_ID"] = ["NOTIF-20260727-00001", "NOTIF-20260727-00002", None]
    df["Notification_Status"] = ["sent", "sent", None]
    df["Notification_Language"] = ["English", "English", None]
    df["Notification_Sent_At"] = ["2026-07-27T12:00:00", "2026-07-27T12:00:01", None]
    df["Notification_Channel"] = ["WhatsApp", "WhatsApp", None]
    df.to_excel(loader.dataset_path, index=False)
    
    # 2. Run the recovery method
    loader.recover_db_from_excel()
    
    # 3. Verify SQLite has been populated with the records
    conn = sqlite3.connect(loader.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT prospect_id, notification_id, status, channel FROM notifications ORDER BY prospect_id")
    rows = cursor.fetchall()
    
    assert len(rows) == 2
    assert rows[0] == (1, "NOTIF-20260727-00001", "sent", "WhatsApp")
    assert rows[1] == (2, "NOTIF-20260727-00002", "sent", "WhatsApp")
    
    # Verify watermark table is synchronized to the max ID
    cursor.execute("SELECT last_prospect_id FROM watermark WHERE id = 1")
    watermark_val = cursor.fetchone()[0]
    conn.close()
    
    assert watermark_val == 2
