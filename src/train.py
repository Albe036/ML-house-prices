from preprocessing import X_train, X_test, y_train, y_test, AllPreprocessing
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "house_price_model.pkl"
TEST_PATH = BASE_DIR / "data" / "raw" / "test.csv"

model_pipeline = Pipeline([
  ("preprocessing", AllPreprocessing()),
  ("regression", Ridge(alpha=10.0))
])

model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R2 Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")