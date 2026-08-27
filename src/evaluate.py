from pathlib import Path
import pandas as pd
import joblib
from preprocessing import CustomDataImputer, AllPreprocessing, X_train, X_test, y_train, y_test
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models"
TEST_PATH = BASE_DIR / "data" / "raw" / "test.csv"

useTest = pd.read_csv(TEST_PATH)

models = {
    "Ridge Regression": Ridge(alpha=10.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Support Vector Regressor": SVR(kernel="rbf", C=1000.0),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42),
}

for name, model in models.items():
  pipeline_test = Pipeline([
      ("preprocessing", AllPreprocessing),
      ("regressor", model),
  ])

  # Entrenamos
  pipeline_test.fit(X_train, y_train)

  # Predecimos y evaluamos
  y_pred = pipeline_test.predict(X_test)
  r2 = r2_score(y_test, y_pred)
  rmse = np.sqrt(mean_squared_error(y_test, y_pred))

  print(
      f"--- {name} --- \n  R²: {r2:.4f} \n  RMSE: ${rmse:,.2f}\n"
  )

  predictions_current_model = pipeline_test.predict(useTest)
  submission = pd.DataFrame(
    { "Id": useTest["Id"], "SalePrice": predictions_current_model }
  )
  OUTPATH_DIR = BASE_DIR / "data" / "submission"
  OUTPATH_DIR.mkdir(parents=True, exist_ok=True)
  submission.to_csv(OUTPATH_DIR / f"submission_{name.replace(' ', '_').lower()}.csv", index=False)


  

""" 
joblib.dump(model_pipeline, MODEL_PATH / "house_price_model_svr.pkl")
print(f"Model saved to {MODEL_PATH / 'house_price_model_svr.pkl'}") 
"""