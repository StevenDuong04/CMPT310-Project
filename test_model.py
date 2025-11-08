# Authors: Steven Duong, Harry Lee, Anthony Trieu, Tony Wu
# Project: CMPT 310 Final Project - Career Path Prediction
# Date: Oct 11, 2025
# Description: This file contains the code for the data processing and model training.

"""
LightGBM Model Performance Evaluation
Evaluates LightGBM model with 10-fold cross-validation and test set metrics
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("LIGHTGBM MODEL PERFORMANCE EVALUATION")
print("=" * 80)
print()

# ============================================================================
# Data Preparation
# ============================================================================

print("[1/6] Loading and preparing dataset...")

student = pd.read_csv('student-scores.csv')
student = student[student['career_aspiration'] != 'Unknown']
student = student.iloc[:, 4:]

# Feature engineering
student['total_score'] = student[['math_score','history_score','physics_score','chemistry_score','biology_score','english_score','geography_score']].sum(axis=1)
student['average_score'] = student['total_score'] / 7
student['best_subject_score'] = student[['math_score','history_score','physics_score','chemistry_score','biology_score','english_score','geography_score']].max(axis=1)
student['worst_subject_score'] = student[['math_score','history_score','physics_score','chemistry_score','biology_score','english_score','geography_score']].min(axis=1)
student['weekly_self_study_hours'].replace(0, 1, inplace=True)
student['study_efficiency'] = student['average_score'] / student['weekly_self_study_hours']

rf_data = student.copy()

label_encoder_gender = LabelEncoder()
rf_data['gender'] = label_encoder_gender.fit_transform(rf_data['gender'])
rf_data['part_time_job'] = rf_data['part_time_job'].astype(int)
rf_data['extracurricular_activities'] = rf_data['extracurricular_activities'].astype(int)

# Domain-specific features
rf_data['science_avg'] = rf_data[['physics_score', 'chemistry_score', 'biology_score']].mean(axis=1)
rf_data['science_strength'] = rf_data[['physics_score', 'chemistry_score', 'biology_score']].max(axis=1)
rf_data['humanities_avg'] = rf_data[['history_score', 'geography_score', 'english_score']].mean(axis=1)
rf_data['humanities_strength'] = rf_data[['history_score', 'geography_score', 'english_score']].max(axis=1)
rf_data['stem_score'] = (rf_data['math_score'] + rf_data['science_avg']) / 2

score_columns = ['math_score', 'history_score', 'physics_score', 'chemistry_score',
                 'biology_score', 'english_score', 'geography_score']
rf_data['score_std'] = rf_data[score_columns].std(axis=1)
rf_data['score_range'] = rf_data[score_columns].max(axis=1) - rf_data[score_columns].min(axis=1)

rf_data['engagement_score'] = (
    rf_data['weekly_self_study_hours'] * 0.4 +
    (10 - rf_data['absence_days']) * 2 +
    rf_data['extracurricular_activities'] * 5
)

rf_data['balanced_student'] = (
    (rf_data['weekly_self_study_hours'] >= 10) &
    (rf_data['weekly_self_study_hours'] <= 30) &
    (rf_data['extracurricular_activities'] == 1)
).astype(int)

def get_subject_preference(row):
    science_max = max(row['physics_score'], row['chemistry_score'], row['biology_score'])
    humanities_max = max(row['history_score'], row['geography_score'], row['english_score'])
    math_score = row['math_score']
    overall_max = max(science_max, humanities_max, math_score)

    if overall_max == science_max:
        return 2
    elif overall_max == humanities_max:
        return 1
    else:
        return 0

rf_data['subject_preference'] = rf_data.apply(get_subject_preference, axis=1)

# Advanced features
rf_data['math_dominance'] = rf_data['math_score'] / rf_data['average_score']
rf_data['science_dominance'] = rf_data['science_avg'] / rf_data['average_score']
rf_data['humanities_dominance'] = rf_data['humanities_avg'] / rf_data['average_score']
rf_data['top_performer'] = (rf_data['average_score'] > 85).astype(int)
rf_data['struggling_student'] = (rf_data['average_score'] < 70).astype(int)
rf_data['high_dedication'] = ((rf_data['weekly_self_study_hours'] > 25) & (rf_data['absence_days'] < 3)).astype(int)
rf_data['extrovert_indicator'] = (rf_data['extracurricular_activities'] == 1).astype(int)
rf_data['stem_oriented'] = ((rf_data['math_score'] > 80) & (rf_data['science_avg'] > 80)).astype(int)
rf_data['business_oriented'] = ((rf_data['math_score'] > 75) & (rf_data['english_score'] > 75)).astype(int)
rf_data['creative_oriented'] = ((rf_data['english_score'] > 80) & (rf_data['humanities_avg'] > 80)).astype(int)
rf_data['study_score_interaction'] = rf_data['weekly_self_study_hours'] * rf_data['average_score'] / 100
rf_data['absence_impact'] = rf_data['absence_days'] * (100 - rf_data['average_score']) / 100
rf_data['math_science_gap'] = abs(rf_data['math_score'] - rf_data['science_avg'])
rf_data['science_humanities_gap'] = abs(rf_data['science_avg'] - rf_data['humanities_avg'])

selected_features = [
    'gender', 'part_time_job', 'absence_days', 'extracurricular_activities',
    'weekly_self_study_hours', 'math_score', 'history_score', 'physics_score',
    'chemistry_score', 'biology_score', 'english_score', 'geography_score',
    'science_avg', 'humanities_avg', 'stem_score', 'score_std', 'score_range',
    'engagement_score', 'balanced_student', 'subject_preference',
    'study_efficiency', 'average_score', 'math_dominance', 'science_dominance',
    'humanities_dominance', 'top_performer', 'struggling_student',
    'high_dedication', 'extrovert_indicator', 'stem_oriented',
    'business_oriented', 'creative_oriented', 'study_score_interaction',
    'absence_impact', 'math_science_gap', 'science_humanities_gap'
]

X = rf_data[selected_features]
label_encoder_career = LabelEncoder()
y = label_encoder_career.fit_transform(rf_data['career_aspiration'])

print(f"✓ Dataset loaded: {len(X)} samples, {len(selected_features)} features")

# ============================================================================
# Train/Test Split 
# ============================================================================

print("\n[2/6] Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")

# ============================================================================
# Model Training
# ============================================================================

print("\n[3/6] Training LightGBM model...")

lgb_model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=10,
    num_leaves=15,
    learning_rate=0.05,
    min_child_samples=5,
    reg_alpha=0.1,
    reg_lambda=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    subsample_freq=1,
    class_weight='balanced',
    random_state=42,
    n_jobs=1,
    verbose=-1
)

lgb_model.fit(X_train, y_train)
print("✓ Model training completed")

# ============================================================================
# Test 1: Cross-Validation Score
# ============================================================================

print("\n[4/6] Cross-Validation Score (10-Fold)")
print("-" * 80)

cv_scores = cross_val_score(lgb_model, X_train, y_train, cv=10, scoring='accuracy', n_jobs=1)
mean_cv_score = cv_scores.mean()
std_cv_score = cv_scores.std()

print(f"CV Scores (10-fold): {cv_scores}")
print(f"Mean CV Score:       {mean_cv_score:.16f}")
print(f"Std Dev:             {std_cv_score:.4f}")
print(f"Formatted:           {mean_cv_score:.4f} ± {std_cv_score:.4f}")

# ============================================================================
# Test 2: Accuracy
# ============================================================================

print("\n[5/6] Test Set Accuracy")
print("-" * 80)

y_pred = lgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Test Accuracy:      {accuracy:.16f}")
print(f"Formatted:          {accuracy:.4f} ({accuracy*100:.2f}%)")

# ============================================================================
# Test 3: Precision, Recall, F1 Score
# ============================================================================

print("\n[6/6] Precision, Recall, and F1 Score")
print("-" * 80)

precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"Precision (weighted): {precision:.16f}")
print(f"Formatted:            {precision:.4f} ({precision*100:.2f}%)")
print()
print(f"Recall (weighted):    {recall:.16f}")
print(f"Formatted:            {recall:.4f} ({recall*100:.2f}%)")
print()
print(f"F1 Score (weighted):  {f1:.16f}")
print(f"Formatted:            {f1:.4f} ({f1*100:.2f}%)")

# ============================================================================
# Visualizations
# ============================================================================

print("\n[7/7] Creating visualizations...")
print("-" * 80)

target_names = label_encoder_career.classes_

# Plot 1: Confusion Matrix
plt.figure(figsize=(14, 12))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=target_names, yticklabels=target_names,
            cbar_kws={'label': 'Count'})
plt.xlabel('Predicted Career', fontsize=12, fontweight='bold')
plt.ylabel('Actual Career', fontsize=12, fontweight='bold')
plt.title(f'LightGBM Confusion Matrix\nAccuracy: {accuracy*100:.2f}%', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('lgb_test_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: lgb_test_confusion_matrix.png")
plt.close()

# Plot 2: Feature Importance
feature_importance = pd.DataFrame({
    'Feature': selected_features,
    'Importance': lgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['Importance'], color='lightcoral', edgecolor='darkred')
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
plt.ylabel('Feature', fontsize=12, fontweight='bold')
plt.title('Top 15 Most Important Features - LightGBM', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('lgb_test_feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: lgb_test_feature_importance.png")
plt.close()

# Plot 3: Per-Class Accuracy
class_accuracies = cm.diagonal() / cm.sum(axis=1)
career_df = pd.DataFrame({
    'Career': target_names,
    'Accuracy': class_accuracies
}).sort_values('Accuracy', ascending=True)

plt.figure(figsize=(12, 8))
plt.barh(career_df['Career'], career_df['Accuracy'], color='salmon', edgecolor='darkred')
plt.xlabel('Accuracy', fontsize=12, fontweight='bold')
plt.ylabel('Career Path', fontsize=12, fontweight='bold')
plt.title('Per-Class Accuracy for Each Career Path - LightGBM', fontsize=14, fontweight='bold')
plt.xlim(0, 1)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('lgb_test_per_class_accuracy.png', dpi=300, bbox_inches='tight')
print("✓ Saved: lgb_test_per_class_accuracy.png")
plt.close()

# ============================================================================
# Final Summary
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY OF RESULTS")
print("=" * 80)
print(f"\nMean CV Score: {mean_cv_score}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

print("\n" + "=" * 80)
print("EVALUATION COMPLETED")
print("=" * 80)
