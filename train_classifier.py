import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# ✅ Load data from the pickle file
with open('/data1.pickle', 'rb') as f:
    data_dict = pickle.load(f)

raw_data = data_dict['data']
labels = np.array(data_dict['labels'])

# ✅ Ensure all samples are of equal length
expected_length = len(raw_data[0])
valid_data = []
valid_labels = []
rejected_count = 0

for d, label in zip(raw_data, labels):
    if len(d) == expected_length:
        valid_data.append(np.array(d).flatten())
        valid_labels.append(label)
    else:
        rejected_count += 1

X = np.array(valid_data)
y = np.array(valid_labels)

# ✅ Split the dataset
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ✅ GridSearchCV for best hyperparameters
param_grid = {
    'n_estimators': [100, 150, 200, 250, 300],
    'max_depth': [10, 15, 20, 25, 30],
    'min_samples_split': [2, 4, 5, 6, 7],
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, verbose=2)

print("🔍 Training model with GridSearchCV...")
grid_search.fit(x_train, y_train)

# ✅ Best model and predictions
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
y_pred = best_model.predict(x_test)
final_accuracy = accuracy_score(y_test, y_pred)

# ✅ Compute training score (accuracy on training set)
train_score = best_model.score(x_train, y_train)

# ✅ Compute loss (1 - accuracy for simplicity, adjust if you have a loss function)
train_loss = 1 - train_score
val_loss = 1 - final_accuracy

# ✅ Compute precision, recall, and f1
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# ✅ Save best model and metrics
with open('model7.p', 'wb') as f:
    pickle.dump({
        'model': best_model,
        'train_score': train_score,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }, f)

# ✅ Print metrics
print("\n🎯 Best Hyperparameters:")
print(best_params)
print(f"🏆 Final Accuracy: {final_accuracy * 100:.2f}%")
print(f"🎯 Train Score: {train_score * 100:.2f}%")
print(f"🔴 Train Loss: {train_loss * 100:.2f}%")
print(f"🔴 Validation Loss: {val_loss * 100:.2f}%")
print(f"🎯 Precision: {precision * 100:.2f}%")
print(f"🔁 Recall:    {recall * 100:.2f}%")
print(f"📐 F1 Score:  {f1 * 100:.2f}%")


# ✅ Summary
print("\n📊 Original samples:", len(raw_data))
print("✅ Used for training:", len(valid_data))
print("❌ Rejected due to inconsistent length:", rejected_count)

# ✅ استخراج نتائج GridSearchCV
cv_results = grid_search.cv_results_

n_estimators_list = cv_results['param_n_estimators'].data
mean_train_scores = cv_results['mean_train_score'] if 'mean_train_score' in cv_results else None
mean_val_scores = cv_results['mean_test_score']
val_losses = 1 - np.array(mean_val_scores)





# ✅ Classification report
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

# ✅ Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("🔍 Confusion Matrix")
plt.show()

from google.colab import files
files.download('model7.p')