from pathlib import Path
import pandas as pd
import joblib
from preprocessing import CustomDataImputer

BASE_DIR = Path(__file__).resolve().parents[1]
TEST_PATH = BASE_DIR / "data" / "raw" / "test.csv"
MODEL_PATH = BASE_DIR / "models"
SUBMISSION_PATH = BASE_DIR / "data" / "submission" / "submission.csv"

loaded_model = joblib.load(MODEL_PATH / "house_price_model.pkl")
useTest = pd.read_csv(TEST_PATH)

predictions = loaded_model.predict(useTest)

submission = pd.DataFrame({"Id": useTest["Id"], "SalePrice": predictions})
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission file saved to {SUBMISSION_PATH}")
