import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Create advanced features for turbofan degradation prediction.
    """
    
    def __init__(self):
        self.sensor_columns = [f'sensor_{i}' for i in range(1, 22)]
    
    def create_rolling_features(self, df, windows=[5, 10, 20]):
        """
        Create rolling mean and std for sensor data.
        Captures trends and variability over time.
        """
        for window in windows:
            for sensor in self.sensor_columns:
                if sensor in df.columns:
                    # Rolling mean
                    df[f'{sensor}_roll_mean_{window}'] = df.groupby('unit_id')[sensor].transform(
                        lambda x: x.rolling(window, min_periods=1).mean()
                    )
                    # Rolling std
                    df[f'{sensor}_roll_std_{window}'] = df.groupby('unit_id')[sensor].transform(
                        lambda x: x.rolling(window, min_periods=1).std()
                    )
        return df
    
    def create_lag_features(self, df, lags=[1, 5, 10]):
        """
        Create lagged features to capture historical sensor values.
        """
        for lag in lags:
            for sensor in self.sensor_columns:
                if sensor in df.columns:
                    df[f'{sensor}_lag_{lag}'] = df.groupby('unit_id')[sensor].shift(lag)
        
        # Fill NaN with 0
        df = df.fillna(0)
        return df
    
    def create_degradation_features(self, df):
        """
        Create features that capture degradation trends.
        """
        for sensor in self.sensor_columns:
            if sensor in df.columns:
                # Cumulative mean (degradation trend)
                df[f'{sensor}_cum_mean'] = df.groupby('unit_id')[sensor].transform(
                    lambda x: x.expanding().mean()
                )
                # Difference from first cycle
                df[f'{sensor}_diff_first'] = df.groupby('unit_id')[sensor].transform(
                    lambda x: x - x.iloc[0]
                )
        return df
    
    def create_interaction_features(self, df):
        """
        Create interaction features between important sensors.
        Based on domain knowledge: temperature, pressure ratios are critical.
        """
        # Example interactions (customize based on sensor importance)
        if 'sensor_2' in df.columns and 'sensor_3' in df.columns:
            df['sensor_2_3_ratio'] = df['sensor_2'] / (df['sensor_3'] + 1e-6)
        
        if 'sensor_4' in df.columns and 'sensor_11' in df.columns:
            df['sensor_4_11_product'] = df['sensor_4'] * df['sensor_11']
        
        return df
    
    def create_all_features(self, df):
        """
        Create all engineered features.
        """
        print("Creating rolling features...")
        df = self.create_rolling_features(df, windows=[5, 10, 15])
        
        print("Creating lag features...")
        df = self.create_lag_features(df, lags=[1, 5])
        
        print("Creating degradation features...")
        df = self.create_degradation_features(df)
        
        print("Creating interaction features...")
        df = self.create_interaction_features(df)
        
        return df


# Example usage
if __name__ == '__main__':
    from data_preprocessing import TurbofanDataLoader
    
    loader = TurbofanDataLoader(data_path='data/raw/')
    train_df, test_df, rul_test = loader.load_data(subset='FD001')
    train_df = loader.add_rul_column(train_df)
    
    engineer = FeatureEngineer()
    train_df = engineer.create_all_features(train_df)
    
    print(f"Original shape: {train_df.shape}")
    print(f"After feature engineering: {train_df.shape}")
