"""
Count Data Points Script
------------------------------
This script loads the SPY one-minute bar data from a pickle file
and counts how many data points are available for a specific date.
"""

import pandas as pd
import os
import glob
from datetime import datetime

def count_data_points_for_date(target_date_str='2024-04-04'):
    # Find the most recent pickle file in the data directory
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    pickle_files = glob.glob(os.path.join(data_dir, 'spy_one_min_data_*.pkl'))
    
    if not pickle_files:
        print("No pickle files found. Please run export_to_pickle.py first.")
        return
    
    # Get the most recent file
    latest_pickle = max(pickle_files, key=os.path.getctime)
    print(f"Loading data from {os.path.basename(latest_pickle)}")
    
    # Load the pickle file
    df = pd.read_pickle(latest_pickle)
    
    # Convert the target date string to datetime
    try:
        target_date = pd.to_datetime(target_date_str).date()
    except Exception as e:
        print(f"Error parsing date: {e}")
        print("Please use format YYYY-MM-DD")
        return
    
    # Check if the index is already a datetime
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        print("Warning: Index is not datetime type. Attempting to convert...")
        try:
            df.index = pd.to_datetime(df.index)
        except:
            print("Could not convert index to datetime. Data format may be incorrect.")
            return
    
    # Filter for the target date
    daily_data = df[df.index.date == target_date]
    
    # Count data points
    count = len(daily_data)
    
    print(f"\nResults for {target_date_str}:")
    print(f"Number of data points: {count}")
    
    if count > 0:
        print(f"First entry: {daily_data.index.min()}")
        print(f"Last entry: {daily_data.index.max()}")
        print(f"Time span: {daily_data.index.max() - daily_data.index.min()}")
        
        # Show distribution by hour
        print("\nData points by hour:")
        hourly_counts = daily_data.groupby(daily_data.index.hour).size()
        for hour, count in hourly_counts.items():
            print(f"{hour:02d}:00 - {hour+1:02d}:00: {count} points")
    else:
        print(f"No data found for {target_date_str}")

if __name__ == "__main__":
    import sys
    
    # Check if a date was provided as a command line argument
    if len(sys.argv) > 1:
        target_date = sys.argv[1]
        count_data_points_for_date(target_date)
    else:
        # Default to 2024-04-04
        count_data_points_for_date('2024-04-04')
        
    print("\nUsage: python count_daily_data.py YYYY-MM-DD")
    print("If no date is provided, defaults to 2024-04-04")