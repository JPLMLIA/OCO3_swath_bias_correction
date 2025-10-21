"""
Utility functions for working with orbit-date mappings.
This module provides functions to look up dates from orbit numbers and vice versa.
"""

import pandas as pd
import os
from pathlib import Path

def get_orbit_date_mapping_path():
    """Get the path to the orbit-date mapping CSV file"""
    # Get the project root directory
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    mapping_path = project_root / "data" / "orbit_date_mapping.csv"
    return mapping_path

def load_orbit_date_mapping():
    """Load the orbit-date mapping CSV file"""
    mapping_path = get_orbit_date_mapping_path()
    
    if not mapping_path.exists():
        raise FileNotFoundError(f"Orbit-date mapping file not found at {mapping_path}")
    
    df = pd.read_csv(mapping_path)
    return df

def orbit_to_date(orbit_number):
    """
    Convert orbit number to date string.
    
    Args:
        orbit_number (int): OCO-3 orbit number
        
    Returns:
        str: Date in YYYY-MM-DD format, or None if orbit not found
    """
    mapping_df = load_orbit_date_mapping()
    
    result = mapping_df[mapping_df['orbit'] == orbit_number]
    
    if len(result) == 0:
        print(f"Warning: Orbit {orbit_number} not found in mapping")
        return None
    
    return result['date'].iloc[0]

def date_to_orbits(date_string):
    """
    Find all orbits for a given date.
    
    Args:
        date_string (str): Date in YYYY-MM-DD format
        
    Returns:
        list: List of orbit numbers for that date
    """
    mapping_df = load_orbit_date_mapping()
    
    result = mapping_df[mapping_df['date'] == date_string]
    
    if len(result) == 0:
        print(f"Warning: No orbits found for date {date_string}")
        return []
    
    return result['orbit'].tolist()

def find_sam_date(sam_id):
    """
    Extract date from SAM ID (format: targetID_orbit).
    
    Args:
        sam_id (str): SAM identifier in format targetID_orbit
        
    Returns:
        str: Date in YYYY-MM-DD format, or None if orbit not found
        
    Example:
        >>> find_sam_date("fossil0001_19513")
        "2024-01-15"
    """
    try:
        # Extract orbit number from SAM ID
        orbit_str = sam_id.split('_')[-1]
        orbit_number = int(orbit_str)
        
        # Look up date for this orbit
        return orbit_to_date(orbit_number)
        
    except (ValueError, IndexError) as e:
        print(f"Error parsing SAM ID '{sam_id}': {e}")
        return None

def get_orbit_range_for_date_range(start_date, end_date):
    """
    Get all orbits within a date range.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        list: List of orbit numbers within the date range
    """
    mapping_df = load_orbit_date_mapping()
    
    # Filter by date range
    mask = (mapping_df['date'] >= start_date) & (mapping_df['date'] <= end_date)
    result = mapping_df[mask]
    
    return result['orbit'].tolist()

# Example usage and testing
if __name__ == "__main__":
    # Test the functions
    print("Testing orbit-date mapping functions...")
    
    # Test orbit to date conversion
    test_orbit = 19513
    date = orbit_to_date(test_orbit)
    print(f"Orbit {test_orbit} -> Date: {date}")
    
    # Test date to orbits conversion
    test_date = "2019-08-06"
    orbits = date_to_orbits(test_date)
    print(f"Date {test_date} -> Orbits: {orbits}")
    
    # Test SAM ID to date conversion
    test_sam = "fossil0001_19513"
    sam_date = find_sam_date(test_sam)
    print(f"SAM {test_sam} -> Date: {sam_date}")
    
    # Test date range
    orbits_in_range = get_orbit_range_for_date_range("2019-08-06", "2019-08-07")
    print(f"Orbits from 2019-08-06 to 2019-08-07: {orbits_in_range}") 