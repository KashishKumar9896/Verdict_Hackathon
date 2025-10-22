"""
Simple script to start the YOLO detection server
Run this from the CodeGeass directory
"""

import sys
import os

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print("Starting YOLO Detection Server...")
print(f"Current directory: {current_dir}")

# Import and run the server
import server

if __name__ == '__main__':
    # The server will run from server.py
    pass
