import pandas as pd
import os

def main():
    excel_path = "Bank_Loan_DS_Project.xlsx"
    if not os.path.exists(excel_path):
        print(f"Error: {excel_path} not found.")
        return
        
    print(f"Loading {excel_path}...")
    df = pd.read_excel(excel_path)
    print(f"Loaded {len(df)} rows.")
    
    print("Updating Contact_No column to '8810612756'...")
    df['Contact_No'] = '8810612756'
    
    print(f"Saving changes back to {excel_path} (this might take a few seconds)...")
    df.to_excel(excel_path, index=False)
    print("Successfully updated all contact numbers in Excel data!")

if __name__ == "__main__":
    main()
