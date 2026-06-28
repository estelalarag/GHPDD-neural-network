from preprocessing import load_data, preprocess_data


from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "house_purchase.csv"


df = load_data(DATA_PATH)


(
X_train,
X_val,
X_test,
y_train,
y_val,
y_test,
processor
)= preprocess_data(df)


print("\nListo para entrenar red neuronal")