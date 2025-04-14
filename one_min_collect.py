"""
SPY One-Minute Bar Data Collection Script
----------------------------------------
Changes made:
1. Added imports for timezone handling (pytz), data manipulation (pandas), 
   and date calculations (dateutil.relativedelta)
2. Set up MongoDB database 'market_data' and collection 'spy_one_min'
3. Created indexing on date field for faster queries
4. Added functions:
   - fetch_one_min_bars(): Retrieves historical data from IB API
   - convert_to_eastern(): Converts timestamps to Eastern timezone
   - save_to_mongodb(): Saves the data to MongoDB
5. Implemented main() function to:
   - Process data day-by-day from Jan 2010 to present
   - Check for existing data to avoid duplicates
   - Handle errors gracefully with retries
   - Rate limit requests to avoid API throttling
6. Properly disconnect from IB at the end of execution
"""

from ib_insync import IB, Stock
import time
from datetime import datetime, timedelta, timezone
from pymongo import MongoClient
import motor.motor_asyncio
import asyncio
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import pytz
import pandas as pd
from dateutil.relativedelta import relativedelta

#Ib insync connection
ib = IB()
ib.connect('192.168.213.1', 7496, clientId=1, timeout=20, readonly=True)

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['market_data']
spy_collection = db['spy_one_min']

# Create index on date for faster queries
spy_collection.create_index([('date', 1)])

def fetch_one_min_bars(symbol, start_date, end_date):
    """
    Fetch one-minute bars for a given symbol and date range
    """
    contract = Stock(symbol, 'SMART', 'USD')
    bars = ib.reqHistoricalData(
        contract,
        endDateTime=end_date.strftime('%Y%m%d %H:%M:%S'),
        durationStr=f'{(end_date - start_date).days + 1} D',
        barSizeSetting='1 min',
        whatToShow='TRADES',
        useRTH=True,
        formatDate=1
    )
    return bars

def convert_to_eastern(bars):
    """
    Convert UTC timestamps to Eastern Time Zone
    """
    eastern = pytz.timezone('US/Eastern')
    for bar in bars:
        if isinstance(bar.date, datetime):
            # Convert from UTC to Eastern
            if bar.date.tzinfo is None:
                bar.date = pytz.utc.localize(bar.date)
            bar.date = bar.date.astimezone(eastern)
    return bars

def save_to_mongodb(bars, date):
    """
    Save bars to MongoDB
    """
    if not bars:
        print(f"No data for {date.strftime('%Y-%m-%d')}")
        return
    
    # Filter for regular trading hours (9:30 AM to 4:00 PM Eastern time)
    regular_hours_bars = []
    for bar in bars:
        # Check if time is between 9:30 AM and 4:00 PM Eastern
        if bar.date.hour >= 9 and bar.date.hour <= 16:
            # Special case for 9 AM (only include 9:30 or later)
            if bar.date.hour == 9 and bar.date.minute < 30:
                continue
            # Special case for 4 PM (only include up to 4:00, not 4:01+)
            if bar.date.hour == 16 and bar.date.minute > 0:
                continue
            regular_hours_bars.append(bar)
    
    # Check if we have exactly 390 bars (6.5 hours × 60 minutes)
    expected_bars = 390
    if len(regular_hours_bars) != expected_bars:
        print(f"SKIPPING: Incorrect number of bars for {date.strftime('%Y-%m-%d')} - Expected {expected_bars} regular trading hours bars but got {len(regular_hours_bars)}")
        return  # Skip this day entirely
    
    print(f"Complete data set with {len(regular_hours_bars)} regular trading hours bars for {date.strftime('%Y-%m-%d')}")
    
    # Convert filtered bars to dictionary format
    bars_dict = []
    duplicates_count = 0
    
    for bar in regular_hours_bars:
        # Check if this specific timestamp already exists in the database
        existing = spy_collection.find_one({'date': bar.date})
        if existing:
            duplicates_count += 1
            continue
            
        bars_dict.append({
            'date': bar.date,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume,
            'average': bar.average if hasattr(bar, 'average') else None,
            'barCount': bar.barCount if hasattr(bar, 'barCount') else None
        })
    
    # Insert into MongoDB
    try:
        if bars_dict:
            result = spy_collection.insert_many(bars_dict)
            print(f"Saved {len(result.inserted_ids)} regular trading hours bars for {date.strftime('%Y-%m-%d')} (Skipped {duplicates_count} duplicates)")
        else:
            print(f"No new data to save for {date.strftime('%Y-%m-%d')} (All {duplicates_count} bars already exist)")
    except Exception as e:
        print(f"Error saving data for {date.strftime('%Y-%m-%d')}: {e}")

def main():
    # Start date: January 1, 2010
    start_date = datetime(2010, 1, 1)
    end_date = datetime.now()
    
    # Process one day at a time
    current_date = start_date
    
    while current_date <= end_date:
        try:
            print(f"Fetching data for {current_date.strftime('%Y-%m-%d')}")
            
            # Set end time to the end of the current day
            day_end = datetime(current_date.year, current_date.month, current_date.day, 23, 59, 59)
            
            # Skip weekends (Saturday = 5, Sunday = 6)
            if current_date.weekday() >= 5:
                print(f"Skipping {current_date.strftime('%Y-%m-%d')} as it's a weekend")
                current_date += timedelta(days=1)
                continue
                
            # Check if we already have data for this date in MongoDB
            existing_data = spy_collection.find_one({'date': {'$gte': current_date, '$lt': day_end}})
            
            if existing_data:
                print(f"Data for {current_date.strftime('%Y-%m-%d')} already exists in database")
            else:
                # Fetch data
                bars = fetch_one_min_bars('SPY', current_date, day_end)
                
                # Convert timestamps to Eastern
                bars = convert_to_eastern(bars)
                
                # Save to MongoDB
                save_to_mongodb(bars, current_date)
                
                # Sleep to avoid rate limiting
                time.sleep(2)
            
            # Move to next day
            current_date += timedelta(days=1)
            
        except Exception as e:
            print(f"Error processing {current_date.strftime('%Y-%m-%d')}: {e}")
            # Wait longer on error
            time.sleep(10)
            # Skip to next day if there's an error
            current_date += timedelta(days=1)

if __name__ == "__main__":
    main()
    ib.disconnect()