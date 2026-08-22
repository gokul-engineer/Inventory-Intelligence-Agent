from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np


def evaluate_model(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")

    return mae, rmse