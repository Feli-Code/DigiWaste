import pandas as pd
import glob
import os

def load_composting_data(file_path, start_row, end_row):
    """
    Load and process CSV data for composting model.
    
    Args:
        file_path (str): Path to the CSV file
        start_row (int): Starting row number
        end_row (int): Ending row number
        
    Returns:
        pd.DataFrame: Processed DataFrame with required columns
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path, header=None, skiprows=1)
        
        all_data = []
        for i in range(start_row - 1, end_row):
            relevant_data = df.iloc[i, :9].values.tolist()  # Get first 9 columns
            all_data.append(relevant_data)
            
        columns = ['Mc', 'P', 'L', 'C', 'CE', 'H', 'LG', 'A', 'In']
        processed_df = pd.DataFrame(all_data, columns=columns).astype(float)
        
        return processed_df
        
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")