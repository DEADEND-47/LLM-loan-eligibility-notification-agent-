import sys
import os

# Ensure root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.app import app

def main():
    print("Initializing Flask test client...")
    app.testing = True
    client = app.test_client()
    
    print("Requesting the Overview page (/) ...")
    res_index = client.get('/')
    print(f"Status Code: {res_index.status_code}")
    assert res_index.status_code == 200, f"Expected 200, got {res_index.status_code}"
    print("Overview page loaded successfully.")
    
    print("\nRequesting the Notifications tab (/notifications) ...")
    res_notif = client.get('/notifications')
    print(f"Status Code: {res_notif.status_code}")
    assert res_notif.status_code == 200, f"Expected 200, got {res_notif.status_code}"
    print("Notifications page loaded successfully.")
    
    # Check if we can search for a customer
    print("\nRequesting Customers page (/customers) ...")
    res_cust = client.get('/customers')
    print(f"Status Code: {res_cust.status_code}")
    assert res_cust.status_code == 200, f"Expected 200, got {res_cust.status_code}"
    print("Customers page loaded successfully.")
    
    print("\nAll integration checks passed successfully!")

if __name__ == "__main__":
    main()
