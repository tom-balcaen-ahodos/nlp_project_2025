import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

filepath_suffix = '_23k'
version_suffix = '_v10'
# File paths
# csv_file_path = f'train_set_small_df_filtered{filepath_suffix}.csv'
csv_file_path = f'train_set_md_df_filtered{filepath_suffix}.csv'
model_path = f"./email_classifier_xgboost{version_suffix}{filepath_suffix}.json"
confusion_matrix_fileName = f"./confusion_matrix_xgboost_bin_class{version_suffix}{filepath_suffix}.png"
roc_filename = f"./roc_curve_xgboost_bin_class{version_suffix}{filepath_suffix}.png"
pr_filename = f"./pr_curve_xgboost_bin_class{version_suffix}{filepath_suffix}.png"
feature_importance_fileName = f"./feature_importance{version_suffix}{filepath_suffix}.png"
# Load data
try:
    train_dataset = pd.read_csv(csv_file_path, sep=';')
    print(f"Successfully loaded {csv_file_path} into DataFrame")
    print("DataFrame Columns:", train_dataset.columns)
except FileNotFoundError:
    print(f"Error: CSV file not found at {csv_file_path}")
    exit()
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# Prepare features
train_dataset['text'] = "Subject: " + train_dataset['Subject'].astype(str) + "\nBody: " + train_dataset['TextBody'].astype(str)
train_dataset['label'] = train_dataset['Complaint'].apply(lambda x: 1 if str(x).lower() == 'true' else 0)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    train_dataset['text'], 
    train_dataset['label'], 
    test_size=0.1, 
    random_state=42
)

# Text feature extraction using TF-IDF
print("Vectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=3000,  # Limit features to avoid memory issues
    min_df=2,            # Ignore terms that appear in less than 2 documents
    max_df=0.95,         # Ignore terms that appear in more than 95% of documents
    ngram_range=(1, 2)   # Include both unigrams and bigrams
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"Training data shape: {X_train_vec.shape}")
print(f"Testing data shape: {X_test_vec.shape}")

# Train an XGBoost model
print("Training XGBoost model...")
model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    early_stopping_rounds=20,
    random_state=42
)

model.fit(
    X_train_vec,
    y_train,
    eval_set=[(X_test_vec, y_test)],
    verbose=True
)

# Save the model
model.save_model(model_path)
print(f"Model saved to {model_path}")

# Evaluate the model
print("Evaluating model...")
y_pred = model.predict(X_test_vec)
y_prob = model.predict_proba(X_test_vec)[:, 1]  # Probabilities for positive class

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
class_names = ['false', 'true']  # Class names

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(confusion_matrix_fileName, dpi=300, bbox_inches='tight')
plt.close()

# Per-class metrics
print("\nPer-class metrics:")
print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1':<10}")
print("-" * 45)

for i in range(len(class_names)):
    tp = cm[i, i]
    all_pred_as_i = cm[:, i].sum()
    all_actual_i = cm[i, :].sum()
    
    precision = tp / all_pred_as_i if all_pred_as_i > 0 else 0
    recall = tp / all_actual_i if all_actual_i > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"{class_names[i]:<15} {precision:.4f}     {recall:.4f}     {f1:.4f}")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig(roc_filename, dpi=300, bbox_inches='tight')
plt.close()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)
average_precision = average_precision_score(y_test, y_prob)

plt.figure(figsize=(10, 8))
plt.plot(recall, precision, color='blue', lw=2, 
         label=f'Precision-Recall curve (AP = {average_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)
plt.savefig(pr_filename, dpi=300, bbox_inches='tight')
plt.close()

print(f"\nROC AUC: {roc_auc:.4f}")
print(f"Average Precision: {average_precision:.4f}")

# Feature importance (only for top features)
if hasattr(model, 'feature_importances_'):
    # Get feature names and importance scores
    feature_names = np.array(vectorizer.get_feature_names_out())
    importances = model.feature_importances_
    
    # Sort and get top 20 features
    indices = np.argsort(importances)[::-1][:20]
    
    plt.figure(figsize=(12, 8))
    plt.title('Top 20 Feature Importances')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), feature_names[indices])
    plt.xlabel('Relative Importance')
    plt.gca().invert_yaxis()  # Display highest importance at the top
    plt.tight_layout()
    plt.savefig(feature_importance_fileName, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nTop 10 most important features:")
    for i in range(10):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")