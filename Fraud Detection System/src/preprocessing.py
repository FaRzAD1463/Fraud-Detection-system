from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from src.config import TEST_SIZE, RANDOM_STATE, SCALER_PATH


def preprocess(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, SCALER_PATH)

    return train_test_split(
        X_scaled,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )