import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
import os
import argparse


filepath_suffix = '_23k'
version_suffix = '_v10'
# File paths
csv_file_path = f'test_set_20_percent_20k.csv'
model_path = f"./email_classifier_xgboost{version_suffix}{filepath_suffix}.json"
vectorizer_path = "./tfidf_vectorizer.joblib"
confusion_matrix_fileName = f"./confusion_matrix_bin_class{version_suffix}{filepath_suffix}.png"
roc_filename = f"./roc_curve_inf_bin_class{version_suffix}_150_train{filepath_suffix}.png"
pr_filename = f"./pr_curve_inf_bin_class{version_suffix}_150_train{filepath_suffix}.png"

def main():
    parser = argparse.ArgumentParser(description='Email Classifier Inference')
    parser.add_argument('--test_file', type=str, default=csv_file_path, 
                        help='Path to the test CSV file')
    parser.add_argument('--model_path', type=str, default=model_path, 
                        help='Path to the trained XGBoost model')
    parser.add_argument('--output_dir', type=str, default=f'./evaluation_results{version_suffix}{filepath_suffix}', 
                        help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = xgb.XGBClassifier()
    model.load_model(args.model_path)
    
    # Load test data
    print(f"Loading test data from {args.test_file}...")
    try:
        test_data = pd.read_csv(args.test_file, sep=';')
        print(f"Loaded {len(test_data)} records from test file.")
    except Exception as e:
        print(f"Error loading test file: {e}")
        return
    
    # Prepare features
    print("Preparing features...")
    test_data['text'] = "Subject: " + test_data['Subject'].astype(str) + "\nBody: " + test_data['TextBody'].astype(str)
    test_data['label'] = test_data['Complaint'].apply(lambda x: 1 if str(x).lower() == 'true' else 0)
    
    # Create and fit vectorizer
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(
        max_features=3000,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2)
    )
    X_test = vectorizer.fit_transform(test_data['text'])
    y_test = test_data['label']
    
    # Inference
    print("Running inference...")
    start_time = time.time()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for positive class
    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time:.2f} seconds")
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    class_names = ['Not Complaint', 'Complaint']
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/confusion_matrix.png", dpi=300)
    plt.close()
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{args.output_dir}/roc_curve.png", dpi=300)
    plt.close()
    
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)
    
    # Plot precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{args.output_dir}/precision_recall_curve.png", dpi=300)
    plt.close()
    
    # Print additional metrics
    print("\nClassification Report:")
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 45)
    
    for i in range(len(class_names)):
        tp = cm[i, i]
        all_pred_as_i = cm[:, i].sum() if cm[:, i].sum() > 0 else 1
        all_actual_i = cm[i, :].sum() if cm[i, :].sum() > 0 else 1
        
        precision = tp / all_pred_as_i
        recall = tp / all_actual_i
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{class_names[i]:<15} {precision:.4f}     {recall:.4f}     {f1:.4f}")
    
    print(f"\nROC AUC: {roc_auc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    
    # Save predictions to CSV
    results_df = pd.DataFrame({
        'Subject': test_data['Subject'],
        'TextBody': test_data['TextBody'],
        'True_Label': y_test,
        'Predicted_Label': y_pred,
        'Complaint_Probability': y_prob
    })
    results_df.to_csv(f"{args.output_dir}/prediction_results{filepath_suffix}.csv", index=False)
    
    print(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    main()