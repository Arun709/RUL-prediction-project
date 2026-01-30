import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

class TurbofanDataLoader:
    """
    Load and preprocess NASA C-MAPSS turbofan data.
    Dataset: 4 subsets (FD001-FD004) with different operating conditions.
    """
    
    def __init__(self, data_path='data/raw/'):
        self.data_path = data_path
        self.sensor_columns = [f'sensor_{i}' for i in range(1, 22)]
        self.setting_columns = ['setting_1', 'setting_2', 'setting_3']
        
    def load_data(self, subset='FD001'):
        """
        Load train and test data for a specific subset.
        
        Args:
            subset: FD001, FD002, FD003, or FD004
        
        Returns:
            train_df, test_df, rul_test (ground truth RUL for test)
        """
        # Column names
        columns = ['unit_id', 'time_cycle'] + self.setting_columns + self.sensor_columns
        
        # Load training data
        train_path = os.path.join(self.data_path, f'train_{subset}.txt')
        train_df = pd.read_csv(train_path, sep=' ', header=None, names=columns)
        train_df = train_df.dropna(axis=1)  # Remove empty columns
        
        # Load test data
        test_path = os.path.join(self.data_path, f'test_{subset}.txt')
        test_df = pd.read_csv(test_path, sep=' ', header=None, names=columns)
        test_df = test_df.dropna(axis=1)
        
        # Load RUL ground truth for test set
        rul_path = os.path.join(self.data_path, f'RUL_{subset}.txt')
        rul_test = pd.read_csv(rul_path, sep=' ', header=None, names=['RUL'])
        
        return train_df, test_df, rul_test
    
    def add_rul_column(self, df):
        """
        Calculate Remaining Useful Life (RUL) for training data.
        RUL = max_cycle - current_cycle for each engine.
        """
        # Get max cycle for each engine
        max_cycle = df.groupby('unit_id')['time_cycle'].max().reset_index()
        max_cycle.columns = ['unit_id', 'max_cycle']
        
        # Merge and calculate RUL
        df = df.merge(max_cycle, on='unit_id', how='left')
        df['RUL'] = df['max_cycle'] - df['time_cycle']
        df = df.drop('max_cycle', axis=1)
        
        return df
    
    def add_labels(self, df, w1=30, w0=15):
        """
        Add binary and multi-class labels for classification tasks.
        
        w1: window for class 1 (imminent failure)
        w0: window for class 0 (healthy)
        """
        df['label_binary'] = (df['RUL'] <= w1).astype(int)
        
        # Multi-class: 0=healthy, 1=warning, 2=critical
        df['label_multiclass'] = 0
        df.loc[df['RUL'] <= w1, 'label_multiclass'] = 1
        df.loc[df['RUL'] <= w0, 'label_multiclass'] = 2
        
        return df
    
    def normalize_data(self, train_df, test_df, columns):
        """
        Normalize sensor and setting columns using MinMaxScaler.
        Fit on train, transform both train and test.
        """
        scaler = MinMaxScaler()
        train_df[columns] = scaler.fit_transform(train_df[columns])
        test_df[columns] = scaler.transform(test_df[columns])
        
        return train_df, test_df, scaler


# Example usage
if __name__ == '__main__':
    loader = TurbofanDataLoader(data_path='data/raw/')
    train_df, test_df, rul_test = loader.load_data(subset='FD001')
    
    # Add RUL to training data
    train_df = loader.add_rul_column(train_df)
    train_df = loader.add_labels(train_df)
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"RUL test shape: {rul_test.shape}")
    print(f"\nFirst few rows:\n{train_df.head()}")
