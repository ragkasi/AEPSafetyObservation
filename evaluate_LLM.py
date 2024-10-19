from sklearn.metrics import classification_report
import pandas as pd

# Evaluate the model
def evaluate_model(file_path):
    data = pd.read_csv(file_path)
    y_true = data['risk_level']
    y_pred = data['predicted_risk_level']
    report = classification_report(y_true, y_pred, target_names=['low','medium','high'])
    print(report)

if __name__ == "__main__":
    evaluate_model("classified_test_data.csv")