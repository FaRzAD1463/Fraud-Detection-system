from imblearn.over_sampling import SMOTE
from src.config import RANDOM_STATE

def balance_data(X_train, y_train):
    smote = SMOTE(random_state=RANDOM_STATE)
    return smote.fit_resample(X_train, y_train)