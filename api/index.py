import os
import sys

# Add project root directory to sys.path so dashboard package can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.app import app
