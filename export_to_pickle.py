"""
MongoDB to Pickle Export Script
------------------------------
This script connects to the MongoDB database 'market_data' and collection 'spy_one_min',
retrieves all the SPY one-minute bar data, and exports it to a pickle file.

The pickle file provides faster data loading for analysis compared to querying MongoDB each time.

Field Names in the MongoDB Collection:
- date: Timestamp for the data point
- open: Opening price for the 1-minute bar
- high: Highest price during the 1-minute bar
- low: Lowest price during the 1-minute bar
- close: Closing price for the 1-minute bar
- volume: Trading volume during the 1-minute bar
- vwap: Volume-weighted average price
- num_trades: Number of trades in the 1-minute bar

Data summary:
Date range: 2009-12-31 14:30:00 to 2025-04-04 19:59:00
Number of trading days: 3760
"""

from pymongo import MongoClient
import pandas as pd
import os
from datetime import datetime

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['market_data']
spy_collection = db['spy_one_min']

def export_to_pickle():
    # Get current timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file path
    pickle_path = os.path.join(output_dir, f'spy_one_min_data_{timestamp}.pkl')
    
    print(f"Retrieving SPY one-minute bar data from MongoDB...")
    
    # Query all documents from the collection
    cursor = spy_collection.find({})
    
    # Convert MongoDB documents to a list
    data_list = list(cursor)
    
    # Check if we have data
    if not data_list:
        print("No data found in MongoDB collection.")
        return
    
    # Convert to pandas DataFrame for easier manipulation and storage
    df = pd.DataFrame(data_list)
    
    # Ensure 'date' is set as the index for time series analysis
    if 'date' in df.columns:
        df.set_index('date', inplace=True)
        # Sort by date
        df.sort_index(inplace=True)
    
    # Remove MongoDB ObjectID as it's not needed and not pickle-compatible
    if '_id' in df.columns:
        df.drop('_id', axis=1, inplace=True)
    
    # Save to pickle file
    print(f"Saving {len(df)} records to {pickle_path}")
    df.to_pickle(pickle_path)
    
    print(f"Data successfully exported to {pickle_path}")
    print(f"DataFrame shape: {df.shape}")
    
    # Display some basic statistics
    print("\nData summary:")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    # Fix: Convert numpy array to pandas Series before calling nunique
    print(f"Number of trading days: {pd.Series(df.index.date).nunique()}")
    
    return pickle_path

if __name__ == "__main__":
    pickle_file = export_to_pickle()
    if pickle_file:
        print(f"\nYou can load this data in your analysis scripts with:")
        print(f"df = pd.read_pickle('{pickle_file}')")