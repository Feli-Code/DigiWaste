import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path

class DataProcessor:
    """Core class for handling waste management data processing and validation."""

    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.metadata: Dict = {}

    def load_data(self, file_path: Union[str, Path], file_type: str = 'csv') -> pd.DataFrame:
        """Load data from various file formats.

        Args:
            file_path: Path to the data file
            file_type: Type of file ('csv', 'excel', 'json')

        Returns:
            Loaded DataFrame
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_type.lower() == 'csv':
            self.data = pd.read_csv(file_path)
        elif file_type.lower() == 'excel':
            self.data = pd.read_excel(file_path)
        elif file_type.lower() == 'json':
            self.data = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        return self.data

    def validate_data(self, required_columns: List[str]) -> bool:
        """Validate if the data contains required columns and formats.

        Args:
            required_columns: List of column names that must be present

        Returns:
            True if validation passes, raises error otherwise
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data first.")

        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        return True

    def preprocess_data(self) -> pd.DataFrame:
        """Perform standard preprocessing steps on the data.

        Returns:
            Preprocessed DataFrame
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data first.")

        # Remove duplicates
        self.data = self.data.drop_duplicates()

        # Handle missing values
        self.data = self.data.fillna(0)

        # Convert data types
        numeric_columns = self.data.select_dtypes(include=['float64', 'int64']).columns
        self.data[numeric_columns] = self.data[numeric_columns].astype('float32')

        return self.data

    def add_metadata(self, key: str, value: any):
        """Add metadata information about the dataset.

        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value

    def get_metadata(self, key: str) -> any:
        """Retrieve metadata information.

        Args:
            key: Metadata key

        Returns:
            Metadata value
        """
        return self.metadata.get(key)
