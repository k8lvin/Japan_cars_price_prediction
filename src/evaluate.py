import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(y_true, y_pred, title="Model Evaluation"):

    # Convert predictions back to original dollar values
    y_true_actual = np.expm1(y_true)
    y_pred_actual = np.expm1(y_pred)

    mae = mean_absolute_error(y_true_actual, y_pred_actual)
    rmse = np.sqrt(mean_squared_error(y_true_actual, y_pred_actual))

    # R² is typically calculated on the same scale the model was trained on
    r2 = r2_score(y_true, y_pred)

    print(title)
    print(f"MAE : ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R² Score: {r2:.4f}")

    return {"mae": mae, "rmse": rmse, "r2": r2}