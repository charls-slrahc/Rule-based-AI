
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import numpy as np

# Credit Risk Dataset Sample (20 records)
data = [
    {"credit_amount": 5000, "age": 35, "duration": 12, "job": "skilled", "credit_history": "good", "employment": "permanent", "housing": "own", "actual": "good"},
    {"credit_amount": 12000, "age": 22, "duration": 18, "job": "unskilled", "credit_history": "bad", "employment": "temporary", "housing": "rent", "actual": "bad"},
    {"credit_amount": 8000, "age": 40, "duration": 30, "job": "skilled", "credit_history": "good", "employment": "permanent", "housing": "own", "actual": "good"},
    {"credit_amount": 15000, "age": 28, "duration": 36, "job": "unskilled", "credit_history": "none", "employment": "unemployed", "housing": "free", "actual": "bad"},
    {"credit_amount": 3000, "age": 50, "duration": 6, "job": "skilled", "credit_history": "good", "employment": "permanent", "housing": "own", "actual": "good"},
    {"credit_amount": 10000, "age": 30, "duration": 24, "job": "skilled", "credit_history": "good", "employment": "temporary", "housing": "rent", "actual": "good"},
    {"credit_amount": 20000, "age": 25, "duration": 48, "job": "unskilled", "credit_history": "bad", "employment": "unemployed", "housing": "free", "actual": "bad"},
    {"credit_amount": 6000, "age": 45, "duration": 15, "job": "skilled", "credit_history": "good", "employment": "permanent", "housing": "own", "actual": "good"},
    {"credit_amount": 18000, "age": 20, "duration": 30, "job": "unskilled", "credit_history": "none", "employment": "temporary", "housing": "rent", "actual": "bad"},
    {"credit_amount": 4000, "age": 55, "duration": 9, "job": "skilled", "credit_history": "good", "employment": "permanent", "housing": "own", "actual": "good"},
    {"credit_amount": 14000, "age": 32, "duration": 20, "job": "skilled", "credit_history": "good", "employment": "permanent", "housing": "own", "actual": "good"},
    {"credit_amount": 16000, "age": 26, "duration": 40, "job": "unskilled", "credit_history": "bad", "employment": "unemployed", "housing": "free", "actual": "bad"},
    {"credit_amount": 7000, "age": 38, "duration": 18, "job": "skilled", "credit_history": "good", "employment": "temporary", "housing": "rent", "actual": "good"},
    {"credit_amount": 22000, "age": 24, "duration": 50, "job": "unskilled", "credit_history": "none", "employment": "temporary", "housing": "rent", "actual": "bad"},
    {"credit_amount": 2500, "age": 60, "duration": 5, "job": "skilled", "credit_history": "good", "employment": "permanent", "housing": "own", "actual": "good"},
    {"credit_amount": 13000, "age": 29, "duration": 25, "job": "skilled", "credit_history": "good", "employment": "permanent", "housing": "own", "actual": "good"},
    {"credit_amount": 19000, "age": 21, "duration": 35, "job": "unskilled", "credit_history": "bad", "employment": "unemployed", "housing": "free", "actual": "bad"},
    {"credit_amount": 5500, "age": 42, "duration": 14, "job": "skilled", "credit_history": "good", "employment": "temporary", "housing": "rent", "actual": "good"},
    {"credit_amount": 17000, "age": 27, "duration": 45, "job": "unskilled", "credit_history": "none", "employment": "temporary", "housing": "rent", "actual": "bad"},
    {"credit_amount": 3500, "age": 48, "duration": 8, "job": "skilled", "credit_history": "good", "employment": "permanent", "housing": "own", "actual": "good"}
]

# Function for System 1: Simple Rule-Based Classifier
def classify_risk_system1(record):
    if record["credit_history"] == "good" and record["employment"] == "permanent":
        return "good"
    if record["credit_amount"] > 10000 and record["age"] < 25:
        return "bad"
    if record["duration"] > 24 and record["job"] == "unskilled":
        return "bad"
    if record["housing"] == "own" and record["credit_history"] == "good":
        return "good"
    return "bad"

# Function for System 2: Rule-Based Scoring System
def classify_risk_system2(record):
    score = 0
    if record["credit_history"] == "good":
        score += 5
    if record["employment"] == "permanent":
        score += 3
    if record["housing"] == "own":
        score += 2
    if record["age"] < 30:
        score -= 1
    if record["duration"] > 24:
        score -= 2
    if record["credit_amount"] > 15000:
        score -= 3
    if record["job"] == "unskilled":
        score -= 2
    return "good" if score >= 5 else "bad"

# Function to calculate metrics
def calculate_metrics(predictions, actuals):
    tp = sum(1 for p, a in zip(predictions, actuals) if p == "good" and a == "good")
    fp = sum(1 for p, a in zip(predictions, actuals) if p == "good" and a == "bad")
    fn = sum(1 for p, a in zip(predictions, actuals) if p == "bad" and a == "good")
    tn = sum(1 for p, a in zip(predictions, actuals) if p == "bad" and a == "bad")
    
    accuracy = (tp + tn) / len(actuals) if len(actuals) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1

# Run System 1
print("System 1: Simple Rule-Based Classifier")
predictions1 = []
for record in data:
    pred = classify_risk_system1(record)
    predictions1.append(pred)
    print(f"Record {data.index(record)+1}: Prediction: {pred}, Actual: {record['actual']}")

actuals = [record["actual"] for record in data]
accuracy1, precision1, recall1, f11 = calculate_metrics(predictions1, actuals)
print(f"Accuracy: {accuracy1:.2f}, Precision: {precision1:.2f}, Recall: {recall1:.2f}, F1-Score: {f11:.2f}")

print("\n" + "="*50 + "\n")

# Run System 2
print("System 2: Rule-Based Scoring System")
predictions2 = []
for record in data:
    pred = classify_risk_system2(record)
    predictions2.append(pred)
    print(f"Record {data.index(record)+1}: Prediction: {pred}, Actual: {record['actual']}")

accuracy2, precision2, recall2, f12 = calculate_metrics(predictions2, actuals)
print(f"Accuracy: {accuracy2:.2f}, Precision: {precision2:.2f}, Recall: {recall2:.2f}, F1-Score: {f12:.2f}")

# --- CONFUSION MATRIX and ROC-AUC ---
y_true = np.array([1 if a == "good" else 0 for a in actuals])
y_pred1 = np.array([1 if p == "good" else 0 for p in predictions1])
y_pred2 = np.array([1 if p == "good" else 0 for p in predictions2])

# --- CONFUSION MATRIX for System 1 ---
cm1 = confusion_matrix(y_true, y_pred1)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=["Bad", "Good"])
disp1.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - System 1: Simple Rule-Based Classifier")
plt.show()

# --- CONFUSION MATRIX for System 2 ---
cm2 = confusion_matrix(y_true, y_pred2)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=["Bad", "Good"])
disp2.plot(cmap=plt.cm.Oranges)
plt.title("Confusion Matrix - System 2: Rule-Based Scoring System")
plt.show()

# --- ROC-AUC CURVE ---
# For simplicity, use predicted label probabilities (since rules are binary, use 1 for "good" and 0 for "bad")
fpr1, tpr1, _ = roc_curve(y_true, y_pred1)
fpr2, tpr2, _ = roc_curve(y_true, y_pred2)
roc_auc1 = auc(fpr1, tpr1)
roc_auc2 = auc(fpr2, tpr2)

plt.figure(figsize=(7, 6))
plt.plot(fpr1, tpr1, color='blue', lw=2, label=f'System 1 (AUC = {roc_auc1:.2f})')
plt.plot(fpr2, tpr2, color='darkorange', lw=2, label=f'System 2 (AUC = {roc_auc2:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('ROC-AUC Curve Comparison')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
  