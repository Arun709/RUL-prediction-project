import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.xgboost
import shap
import matplotlib.pyplot as plt
import joblib


class RULPredictor:
    """
    Remaining Useful Life prediction models with MLflow tracking and SHAP explanations.
    """

    def __init__(self, model_type='lightgbm'):
        """
        Initialize RUL predictor.

        Args:
            model_type: Type of model ('lightgbm', 'xgboost', 'random_forest')
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = None

    def get_model(self, **kwargs):
        """
        Initialize model based on type.

        Args:
            **kwargs: Model-specific hyperparameters

        Returns:
            Initialized model object
        """
        if self.model_type == 'lightgbm':
            return lgb.LGBMRegressor(
                n_estimators=kwargs.get('n_estimators', 500),
                learning_rate=kwargs.get('learning_rate', 0.05),
                max_depth=kwargs.get('max_depth', 7),
                num_leaves=kwargs.get('num_leaves', 31),
                random_state=42,
                verbose=-1
            )
        elif self.model_type == 'xgboost':
            return xgb.XGBRegressor(
                n_estimators=kwargs.get('n_estimators', 500),
                learning_rate=kwargs.get('learning_rate', 0.05),
                max_depth=kwargs.get('max_depth', 7),
                random_state=42,
                verbosity=0
            )
        elif self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 200),
                max_depth=kwargs.get('max_depth', 15),
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Train the model with optional validation set.

        Args:
            X_train: Training features (DataFrame)
            y_train: Training targets (array-like)
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            **kwargs: Model hyperparameters

        Returns:
            Trained model
        """
        # Input validation
        if X_train is None or y_train is None:
            raise ValueError("Training data cannot be None")
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Training data cannot be empty")
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must have same length")

        self.model = self.get_model(**kwargs)
        self.feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else None

        with mlflow.start_run(run_name=f"{self.model_type}_rul_prediction"):
            # Log parameters
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_params({k: v for k, v in kwargs.items() if v is not None})

            # Train with appropriate method based on model type
            try:
                if self.model_type == 'lightgbm' and X_val is not None and y_val is not None:
                    self.model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                    )
                elif self.model_type == 'xgboost' and X_val is not None and y_val is not None:
                    self.model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )
                else:
                    self.model.fit(X_train, y_train)

                # Log model
                if self.model_type == 'lightgbm':
                    mlflow.lightgbm.log_model(self.model, "model")
                elif self.model_type == 'xgboost':
                    mlflow.xgboost.log_model(self.model, "model")
                else:
                    mlflow.sklearn.log_model(self.model, "model")

                print(f"✓ {self.model_type} training complete!")

            except Exception as e:
                print(f"✗ Training failed: {str(e)}")
                raise

        return self.model

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of metrics and predictions
        """
        # Input validation
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        if X_test is None or y_test is None:
            raise ValueError("Test data cannot be None")
        if len(X_test) == 0 or len(y_test) == 0:
            raise ValueError("Test data cannot be empty")
        if len(X_test) != len(y_test):
            raise ValueError("X_test and y_test must have same length")

        try:
            y_pred = self.model.predict(X_test)

            metrics = {
                'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                'MAE': float(mean_absolute_error(y_test, y_pred)),
                'R2': float(r2_score(y_test, y_pred))
            }

            # Log metrics to MLflow only if run is active
            if mlflow.active_run():
                mlflow.log_metrics(metrics)
            else:
                with mlflow.start_run(run_name=f"{self.model_type}_evaluation"):
                    mlflow.log_metrics(metrics)

            print(f"\n{self.model_type} Performance:")
            print("-" * 40)
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")

            return metrics, y_pred

        except Exception as e:
            print(f"✗ Evaluation failed: {str(e)}")
            raise

    def explain_predictions(self, X_sample, save_path='models/shap_plots/'):
        """
        Generate SHAP explanations for model predictions.

        Args:
            X_sample: Sample data for SHAP analysis
            save_path: Directory to save SHAP plots

        Returns:
            SHAP explainer and values
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        import os
        os.makedirs(save_path, exist_ok=True)

        try:
            # Create SHAP explainer based on model type
            if self.model_type in ['lightgbm', 'xgboost', 'random_forest']:
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_sample)
            else:
                # For non-tree models, use KernelExplainer with proper sampling
                background = shap.sample(X_sample, min(100, len(X_sample)))
                explainer = shap.KernelExplainer(self.model.predict, background)
                shap_values = explainer.shap_values(X_sample)

            # Global feature importance summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.tight_layout()
            plt.savefig(f'{save_path}{self.model_type}_shap_summary.png', dpi=300, bbox_inches='tight')
            plt.close()

            # Feature importance bar plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, plot_type='bar', show=False)
            plt.tight_layout()
            plt.savefig(f'{save_path}{self.model_type}_shap_importance.png', dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✓ SHAP plots saved to {save_path}")

            return explainer, shap_values

        except Exception as e:
            print(f"✗ SHAP explanation failed: {str(e)}")
            raise

    def save_model(self, path='models/saved_models/'):
        """
        Save trained model and metadata to disk.

        Args:
            path: Directory path to save model
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")

        import os
        os.makedirs(path, exist_ok=True)

        try:
            # Save model
            model_path = f'{path}{self.model_type}_rul_model.pkl'
            joblib.dump(self.model, model_path)

            # Save metadata (feature names, model type)
            metadata = {
                'feature_names': self.feature_names,
                'model_type': self.model_type
            }
            metadata_path = f'{path}{self.model_type}_metadata.pkl'
            joblib.dump(metadata, metadata_path)

            print(f"✓ Model saved to {model_path}")
            print(f"✓ Metadata saved to {metadata_path}")

        except Exception as e:
            print(f"✗ Save failed: {str(e)}")
            raise

    def load_model(self, path, metadata_path=None):
        """
        Load trained model and metadata from disk.

        Args:
            path: Path to model file
            metadata_path: Path to metadata file (optional)

        Returns:
            Loaded model
        """
        try:
            self.model = joblib.load(path)

            # Load metadata if path provided
            if metadata_path:
                metadata = joblib.load(metadata_path)
                self.feature_names = metadata.get('feature_names')
                self.model_type = metadata.get('model_type', self.model_type)
                print(f"✓ Model and metadata loaded from {path}")
            else:
                # Try to auto-detect metadata file
                auto_metadata_path = path.replace('_rul_model.pkl', '_metadata.pkl')
                if os.path.exists(auto_metadata_path):
                    metadata = joblib.load(auto_metadata_path)
                    self.feature_names = metadata.get('feature_names')
                    self.model_type = metadata.get('model_type', self.model_type)
                    print(f"✓ Model and metadata loaded from {path}")
                else:
                    print(f"✓ Model loaded from {path} (metadata not found)")

            return self.model

        except Exception as e:
            print(f"✗ Load failed: {str(e)}")
            raise

    def predict(self, X):
        """
        Make predictions on new data.

        Args:
            X: Features for prediction

        Returns:
            Predicted RUL values
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() or load_model() first.")

        try:
            return self.model.predict(X)
        except Exception as e:
            print(f"✗ Prediction failed: {str(e)}")
            raise


# Scoring function for business impact
def calculate_business_impact(y_true, y_pred, cost_early_maintenance=1000, 
                               cost_unplanned_downtime=50000, threshold=15):
    """
    Calculate cost savings from predictive maintenance.

    Args:
        y_true: Actual RUL values
        y_pred: Predicted RUL values
        cost_early_maintenance: Cost of planned maintenance ($)
        cost_unplanned_downtime: Cost of unexpected failure ($)
        threshold: RUL threshold for triggering maintenance

    Returns:
        Dictionary with cost breakdown and savings analysis
    """
    # Input validation
    if y_true is None or y_pred is None:
        raise ValueError("Input arrays cannot be None")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Input arrays cannot be empty")

    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true (len={len(y_true)}) and y_pred (len={len(y_pred)}) must have same length")

    total_cost = 0
    early_maintenance_count = 0
    missed_failures = 0
    correct_predictions = 0

    for true_rul, pred_rul in zip(y_true, y_pred):
        # Predicted maintenance needed
        if pred_rul <= threshold:
            if true_rul <= threshold:
                # Correct prediction - planned maintenance
                total_cost += cost_early_maintenance
                early_maintenance_count += 1
                correct_predictions += 1
            else:
                # False alarm - early maintenance
                total_cost += cost_early_maintenance * 0.5  # Partial cost
                early_maintenance_count += 1
        else:
            if true_rul <= threshold:
                # Missed failure - unplanned downtime
                total_cost += cost_unplanned_downtime
                missed_failures += 1

    # Baseline cost (no prediction, all failures are unplanned)
    baseline_cost = np.sum(y_true <= threshold) * cost_unplanned_downtime

    # Safe division
    if baseline_cost == 0:
        cost_savings = 0.0
        savings_percentage = 0.0
    else:
        cost_savings = baseline_cost - total_cost
        savings_percentage = (cost_savings / baseline_cost) * 100

    # Calculate accuracy
    total_failures = np.sum(y_true <= threshold)
    accuracy = (correct_predictions / total_failures * 100) if total_failures > 0 else 0.0

    return {
        'total_cost': float(total_cost),
        'baseline_cost': float(baseline_cost),
        'cost_savings': float(cost_savings),
        'savings_percentage': float(savings_percentage),
        'early_maintenance_count': int(early_maintenance_count),
        'missed_failures': int(missed_failures),
        'correct_predictions': int(correct_predictions),
        'prediction_accuracy': float(accuracy)
    }


# Example usage function
def example_usage():
    """
    Example of how to use the RULPredictor class.
    """
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = 100 - (X.sum(axis=1) * 10 + np.random.randn(n_samples) * 5)
    y = np.maximum(y, 0)  # RUL cannot be negative

    # Split data
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]

    # Initialize and train model
    predictor = RULPredictor(model_type='lightgbm')
    predictor.train(X_train, y_train, X_val, y_val, n_estimators=100, learning_rate=0.1)

    # Evaluate
    metrics, y_pred = predictor.evaluate(X_test, y_test)

    # Calculate business impact
    impact = calculate_business_impact(y_test, y_pred)
    print(f"\nBusiness Impact Analysis:")
    print("-" * 40)
    for key, value in impact.items():
        print(f"{key}: {value}")

    # Save model
    predictor.save_model()

    return predictor, metrics, impact


if __name__ == "__main__":
    # Run example
    predictor, metrics, impact = example_usage()