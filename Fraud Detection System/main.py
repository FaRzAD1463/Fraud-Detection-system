from src.data_loader import load_data
from src.preprocessing import preprocess
from src.imbalance_handler import balance_data
from src.model_trainer import train_models
from src.evaluator import evaluate
from src.threshold import optimize_threshold


def main():
    print("Loading Data...")
    df = load_data()

    print("Preprocessing...")
    X_train, X_test, y_train, y_test = preprocess(df)

    print("Balancing Data...")
    X_train_bal, y_train_bal = balance_data(X_train, y_train)

    print("Training Models...")
    model = train_models(X_train_bal, y_train_bal)

    print("Evaluating...")
    evaluate(model, X_test, y_test)

    print("Optimizing Threshold...")
    optimize_threshold(model, X_test, y_test)

    print("Done.")


if __name__ == "__main__":
    main()