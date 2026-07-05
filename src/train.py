import os
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import joblib
from evaluate import evaluate_model
from preprocess import build_preprocessor, clean_data, load_data, define_columns, TARGET

MODEL_PATH = "models/price_pipeline.pkl"


def build_pipeline(num_cols: list, onehot_cols: list, target_cols: list) -> Pipeline:
    preprocessor = build_preprocessor(num_cols, onehot_cols, target_cols)
    model = XGBRegressor(random_state=42, n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,colsample_bytree=0.8)

    pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
    ])
    
    return pipeline


def main():
    print('Japan car model  - training')
    
    # load cleaned data
    df = load_data()
    df = clean_data(df)
    num_cols, onehot_cols, target_cols = define_columns()
    
    # split features and target
    X = df[num_cols + onehot_cols + target_cols]
    y = df[TARGET]
    
    #train test split
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=42)
    
    # Build and train pipeline
    print("Building pipeline...")
    pipeline = build_pipeline(num_cols, onehot_cols, target_cols)
    
    print('Training XGBoost model')
    pipeline.fit(X_train, y_train)
    print('Training complete. \n')
    
    # Evaluate the model
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    
    train_metrics = evaluate_model(y_train, y_pred_train, title='Training set performance')
    test_metrics = evaluate_model(y_test, y_pred_test, title= 'Test set performance') 
    
    print("\nTraining Metrics")
    print(train_metrics)
    print("\nTest Metrics")
    print(test_metrics)
        
    # Save the pipeline
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n Model saved to: {MODEL_PATH}")

if __name__ == '__main__':
        main()