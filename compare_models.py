"""
Random Forest vs LightGBM Comparison
Compares both models on the augmented dataset with 10-fold cross-validation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("RANDOM FOREST vs LIGHTGBM vs KNN vs DECISION TREE COMPARISON")
print("=" * 80)
print()

# ============================================================================
# Data Preparation
# ============================================================================

print("[1/10] Loading and preparing dataset...")

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

print(f"✓ Dataset loaded: {len(X)} samples, {len(selected_features)} features, {len(label_encoder_career.classes_)} classes")

# ============================================================================
# Train/Test Split
# ============================================================================

print("\n[2/10] Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")

# ============================================================================
# Model 1: Random Forest
# ============================================================================

print("\n[3/10] Training Random Forest...")
print("-" * 80)

rf_model = RandomForestClassifier(
    n_estimators=200,        # Same as LightGBM for fair comparison
    max_depth=10,            # Match LightGBM depth
    min_samples_split=20,    # Prevent overfitting
    min_samples_leaf=10,     # Prevent overfitting
    max_features='sqrt',     # Reduce overfitting
    class_weight='balanced',
    random_state=42,
    n_jobs=1,
    verbose=0
)

print("Random Forest Configuration:")
print(f"  n_estimators: 200")
print(f"  max_depth: 10")
print(f"  min_samples_split: 20")
print(f"  min_samples_leaf: 10")
print(f"  max_features: 'sqrt'")
print()

rf_start = time.time()
rf_model.fit(X_train, y_train)
rf_train_time = time.time() - rf_start

print(f"✓ Training completed in {rf_train_time:.2f} seconds")

# Random Forest 10-fold CV
print("\nCalculating 10-fold Cross-Validation...")
rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=10, scoring='accuracy', n_jobs=1)
rf_cv_mean = rf_cv_scores.mean()
rf_cv_std = rf_cv_scores.std()

print(f"✓ 10-Fold CV Score: {rf_cv_mean:.4f} ± {rf_cv_std:.4f}")

# Random Forest test predictions
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred, average='weighted', zero_division=0)
rf_recall = recall_score(y_test, rf_pred, average='weighted', zero_division=0)
rf_f1 = f1_score(y_test, rf_pred, average='weighted', zero_division=0)

print(f"✓ Test Accuracy: {rf_accuracy:.4f}")

# ============================================================================
# Model 2: LightGBM
# ============================================================================

print("\n[4/10] Training LightGBM...")
print("-" * 80)

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

print("LightGBM Configuration:")
print(f"  n_estimators: 200")
print(f"  max_depth: 10")
print(f"  num_leaves: 15")
print(f"  learning_rate: 0.05")
print(f"  L1/L2 regularization: 0.1")
print()

lgb_start = time.time()
lgb_model.fit(X_train, y_train)
lgb_train_time = time.time() - lgb_start

print(f"✓ Training completed in {lgb_train_time:.2f} seconds")

# LightGBM 10-fold CV
print("\nCalculating 10-fold Cross-Validation...")
lgb_cv_scores = cross_val_score(lgb_model, X_train, y_train, cv=10, scoring='accuracy', n_jobs=1)
lgb_cv_mean = lgb_cv_scores.mean()
lgb_cv_std = lgb_cv_scores.std()

print(f"✓ 10-Fold CV Score: {lgb_cv_mean:.4f} ± {lgb_cv_std:.4f}")

# LightGBM test predictions
lgb_pred = lgb_model.predict(X_test)
lgb_accuracy = accuracy_score(y_test, lgb_pred)
lgb_precision = precision_score(y_test, lgb_pred, average='weighted', zero_division=0)
lgb_recall = recall_score(y_test, lgb_pred, average='weighted', zero_division=0)
lgb_f1 = f1_score(y_test, lgb_pred, average='weighted', zero_division=0)

print(f"✓ Test Accuracy: {lgb_accuracy:.4f}")

# ============================================================================
# Model 3: KNN
# ============================================================================

print("\n[5/10] Training KNN...")
print("-" * 80)

# Scale features for KNN (important for distance-based algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_model = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',
    metric='minkowski',
    p=2,
    n_jobs=1
)

print("KNN Configuration:")
print(f"  n_neighbors: 5")
print(f"  weights: 'distance'")
print(f"  metric: 'minkowski' (p=2, Euclidean)")
print()

knn_start = time.time()
knn_model.fit(X_train_scaled, y_train)
knn_train_time = time.time() - knn_start

print(f"✓ Training completed in {knn_train_time:.2f} seconds")

# KNN 10-fold CV
print("\nCalculating 10-fold Cross-Validation...")
knn_cv_scores = cross_val_score(knn_model, X_train_scaled, y_train, cv=10, scoring='accuracy', n_jobs=1)
knn_cv_mean = knn_cv_scores.mean()
knn_cv_std = knn_cv_scores.std()

print(f"✓ 10-Fold CV Score: {knn_cv_mean:.4f} ± {knn_cv_std:.4f}")

# KNN test predictions
knn_pred = knn_model.predict(X_test_scaled)
knn_accuracy = accuracy_score(y_test, knn_pred)
knn_precision = precision_score(y_test, knn_pred, average='weighted', zero_division=0)
knn_recall = recall_score(y_test, knn_pred, average='weighted', zero_division=0)
knn_f1 = f1_score(y_test, knn_pred, average='weighted', zero_division=0)

print(f"✓ Test Accuracy: {knn_accuracy:.4f}")

# ============================================================================
# Model 4: Decision Tree
# ============================================================================

print("\n[6/10] Training Decision Tree...")
print("-" * 80)

dt_model = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42
)

print("Decision Tree Configuration:")
print(f"  max_depth: 10")
print(f"  min_samples_split: 20")
print(f"  min_samples_leaf: 10")
print(f"  class_weight: 'balanced'")
print()

dt_start = time.time()
dt_model.fit(X_train, y_train)
dt_train_time = time.time() - dt_start

print(f"✓ Training completed in {dt_train_time:.2f} seconds")

# Decision Tree 10-fold CV
print("\nCalculating 10-fold Cross-Validation...")
dt_cv_scores = cross_val_score(dt_model, X_train, y_train, cv=10, scoring='accuracy', n_jobs=1)
dt_cv_mean = dt_cv_scores.mean()
dt_cv_std = dt_cv_scores.std()

print(f"✓ 10-Fold CV Score: {dt_cv_mean:.4f} ± {dt_cv_std:.4f}")

# Decision Tree test predictions
dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
dt_precision = precision_score(y_test, dt_pred, average='weighted', zero_division=0)
dt_recall = recall_score(y_test, dt_pred, average='weighted', zero_division=0)
dt_f1 = f1_score(y_test, dt_pred, average='weighted', zero_division=0)

print(f"✓ Test Accuracy: {dt_accuracy:.4f}")

# ============================================================================
# Detailed Comparison
# ============================================================================

print("\n[7/10] Detailed Performance Comparison")
print("=" * 80)

# Helper function to determine winner among four models
def get_winner(rf_val, lgb_val, knn_val, dt_val, lower_is_better=False):
    if lower_is_better:
        best_val = min(rf_val, lgb_val, knn_val, dt_val)
    else:
        best_val = max(rf_val, lgb_val, knn_val, dt_val)

    if best_val == rf_val:
        return "Random Forest ✅"
    elif best_val == lgb_val:
        return "LightGBM ✅"
    elif best_val == knn_val:
        return "KNN ✅"
    else:
        return "Decision Tree ✅"

print(f"\n{'Metric':<22} {'Random Forest':<16} {'LightGBM':<16} {'KNN':<16} {'Decision Tree':<16} {'Winner':<18}")
print("-" * 110)

# CV Score
rf_cv_str = f"{rf_cv_mean:.4f}±{rf_cv_std:.4f}"
lgb_cv_str = f"{lgb_cv_mean:.4f}±{lgb_cv_std:.4f}"
knn_cv_str = f"{knn_cv_mean:.4f}±{knn_cv_std:.4f}"
dt_cv_str = f"{dt_cv_mean:.4f}±{dt_cv_std:.4f}"
cv_winner = get_winner(rf_cv_mean, lgb_cv_mean, knn_cv_mean, dt_cv_mean)
print(f"{'10-Fold CV Score':<22} {rf_cv_str:<16} {lgb_cv_str:<16} {knn_cv_str:<16} {dt_cv_str:<16} {cv_winner:<18}")

# Test Accuracy
acc_winner = get_winner(rf_accuracy, lgb_accuracy, knn_accuracy, dt_accuracy)
print(f"{'Test Accuracy':<22} {rf_accuracy:<16.4f} {lgb_accuracy:<16.4f} {knn_accuracy:<16.4f} {dt_accuracy:<16.4f} {acc_winner:<18}")

# Precision
prec_winner = get_winner(rf_precision, lgb_precision, knn_precision, dt_precision)
print(f"{'Precision (weighted)':<22} {rf_precision:<16.4f} {lgb_precision:<16.4f} {knn_precision:<16.4f} {dt_precision:<16.4f} {prec_winner:<18}")

# Recall
rec_winner = get_winner(rf_recall, lgb_recall, knn_recall, dt_recall)
print(f"{'Recall (weighted)':<22} {rf_recall:<16.4f} {lgb_recall:<16.4f} {knn_recall:<16.4f} {dt_recall:<16.4f} {rec_winner:<18}")

# F1 Score
f1_winner = get_winner(rf_f1, lgb_f1, knn_f1, dt_f1)
print(f"{'F1 Score (weighted)':<22} {rf_f1:<16.4f} {lgb_f1:<16.4f} {knn_f1:<16.4f} {dt_f1:<16.4f} {f1_winner:<18}")

# Training Time
time_winner = get_winner(rf_train_time, lgb_train_time, knn_train_time, dt_train_time, lower_is_better=True)
print(f"{'Training Time (sec)':<22} {rf_train_time:<16.2f} {lgb_train_time:<16.2f} {knn_train_time:<16.2f} {dt_train_time:<16.2f} {time_winner:<18}")

# Overfitting Check (CV vs Test gap)
rf_gap = rf_cv_mean - rf_accuracy
lgb_gap = lgb_cv_mean - lgb_accuracy
knn_gap = knn_cv_mean - knn_accuracy
dt_gap = dt_cv_mean - dt_accuracy

print()
print("Overfitting Analysis (CV Score - Test Accuracy):")
print(f"  Random Forest: {rf_gap:+.4f} {'⚠️ Overfitting' if rf_gap > 0.02 else '✅ Good generalization'}")
print(f"  LightGBM:      {lgb_gap:+.4f} {'⚠️ Overfitting' if lgb_gap > 0.02 else '✅ Good generalization'}")
print(f"  KNN:           {knn_gap:+.4f} {'⚠️ Overfitting' if knn_gap > 0.02 else '✅ Good generalization'}")
print(f"  Decision Tree: {dt_gap:+.4f} {'⚠️ Overfitting' if dt_gap > 0.02 else '✅ Good generalization'}")

# ============================================================================
# Classification Reports
# ============================================================================

print("\n[8/10] Detailed Classification Reports")
print("=" * 80)

target_names = label_encoder_career.classes_

print("\nRandom Forest Classification Report:")
print("-" * 80)
print(classification_report(y_test, rf_pred, target_names=target_names, zero_division=0))

print("\nLightGBM Classification Report:")
print("-" * 80)
print(classification_report(y_test, lgb_pred, target_names=target_names, zero_division=0))

print("\nKNN Classification Report:")
print("-" * 80)
print(classification_report(y_test, knn_pred, target_names=target_names, zero_division=0))

print("\nDecision Tree Classification Report:")
print("-" * 80)
print(classification_report(y_test, dt_pred, target_names=target_names, zero_division=0))

# ============================================================================
# Visualizations
# ============================================================================

print("\n[9/10] Creating comparison visualizations...")

# Side-by-side confusion matrices (2x2 for 4 models)
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# Random Forest Confusion Matrix
cm_rf = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=target_names, yticklabels=target_names, cbar_kws={'label': 'Count'})
axes[0, 0].set_xlabel('Predicted Career', fontsize=11)
axes[0, 0].set_ylabel('Actual Career', fontsize=11)
axes[0, 0].set_title(f'Random Forest Confusion Matrix\nAccuracy: {rf_accuracy*100:.2f}%', fontsize=12, fontweight='bold')
axes[0, 0].tick_params(axis='x', rotation=45)

# LightGBM Confusion Matrix
cm_lgb = confusion_matrix(y_test, lgb_pred)
sns.heatmap(cm_lgb, annot=True, fmt='d', cmap='Reds', ax=axes[0, 1],
            xticklabels=target_names, yticklabels=target_names, cbar_kws={'label': 'Count'})
axes[0, 1].set_xlabel('Predicted Career', fontsize=11)
axes[0, 1].set_ylabel('Actual Career', fontsize=11)
axes[0, 1].set_title(f'LightGBM Confusion Matrix\nAccuracy: {lgb_accuracy*100:.2f}%', fontsize=12, fontweight='bold')
axes[0, 1].tick_params(axis='x', rotation=45)

# KNN Confusion Matrix
cm_knn = confusion_matrix(y_test, knn_pred)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens', ax=axes[1, 0],
            xticklabels=target_names, yticklabels=target_names, cbar_kws={'label': 'Count'})
axes[1, 0].set_xlabel('Predicted Career', fontsize=11)
axes[1, 0].set_ylabel('Actual Career', fontsize=11)
axes[1, 0].set_title(f'KNN Confusion Matrix\nAccuracy: {knn_accuracy*100:.2f}%', fontsize=12, fontweight='bold')
axes[1, 0].tick_params(axis='x', rotation=45)

# Decision Tree Confusion Matrix
cm_dt = confusion_matrix(y_test, dt_pred)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Purples', ax=axes[1, 1],
            xticklabels=target_names, yticklabels=target_names, cbar_kws={'label': 'Count'})
axes[1, 1].set_xlabel('Predicted Career', fontsize=11)
axes[1, 1].set_ylabel('Actual Career', fontsize=11)
axes[1, 1].set_title(f'Decision Tree Confusion Matrix\nAccuracy: {dt_accuracy*100:.2f}%', fontsize=12, fontweight='bold')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('model_comparison_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✓ Saved: model_comparison_confusion_matrices.png")
plt.close()

# Metrics comparison bar chart (now 4 models)
metrics_data = {
    'Metric': ['CV Score', 'Test Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Random Forest': [rf_cv_mean, rf_accuracy, rf_precision, rf_recall, rf_f1],
    'LightGBM': [lgb_cv_mean, lgb_accuracy, lgb_precision, lgb_recall, lgb_f1],
    'KNN': [knn_cv_mean, knn_accuracy, knn_precision, knn_recall, knn_f1],
    'Decision Tree': [dt_cv_mean, dt_accuracy, dt_precision, dt_recall, dt_f1]
}

df_metrics = pd.DataFrame(metrics_data)
x = np.arange(len(df_metrics['Metric']))
width = 0.2

fig, ax = plt.subplots(figsize=(16, 6))
bars1 = ax.bar(x - 1.5*width, df_metrics['Random Forest'], width, label='Random Forest', color='steelblue', edgecolor='navy')
bars2 = ax.bar(x - 0.5*width, df_metrics['LightGBM'], width, label='LightGBM', color='coral', edgecolor='darkred')
bars3 = ax.bar(x + 0.5*width, df_metrics['KNN'], width, label='KNN', color='seagreen', edgecolor='darkgreen')
bars4 = ax.bar(x + 1.5*width, df_metrics['Decision Tree'], width, label='Decision Tree', color='mediumpurple', edgecolor='indigo')

ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Random Forest vs LightGBM vs KNN vs Decision Tree - Performance Metrics Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df_metrics['Metric'])
ax.legend()
ax.set_ylim(0.4, 1.0)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig('model_comparison_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: model_comparison_metrics.png")
plt.close()

# ============================================================================
# Final Summary
# ============================================================================

print("\n[10/10] Final Summary")
print("=" * 80)

# Determine overall winner among 4 models
rf_wins = 0
lgb_wins = 0
knn_wins = 0
dt_wins = 0

# CV Score
scores_cv = {'rf': rf_cv_mean, 'lgb': lgb_cv_mean, 'knn': knn_cv_mean, 'dt': dt_cv_mean}
best_cv = max(scores_cv, key=scores_cv.get)
if best_cv == 'rf': rf_wins += 1
elif best_cv == 'lgb': lgb_wins += 1
elif best_cv == 'knn': knn_wins += 1
else: dt_wins += 1

# Accuracy
scores_acc = {'rf': rf_accuracy, 'lgb': lgb_accuracy, 'knn': knn_accuracy, 'dt': dt_accuracy}
best_acc = max(scores_acc, key=scores_acc.get)
if best_acc == 'rf': rf_wins += 1
elif best_acc == 'lgb': lgb_wins += 1
elif best_acc == 'knn': knn_wins += 1
else: dt_wins += 1

# F1 Score
scores_f1 = {'rf': rf_f1, 'lgb': lgb_f1, 'knn': knn_f1, 'dt': dt_f1}
best_f1 = max(scores_f1, key=scores_f1.get)
if best_f1 == 'rf': rf_wins += 1
elif best_f1 == 'lgb': lgb_wins += 1
elif best_f1 == 'knn': knn_wins += 1
else: dt_wins += 1

# Training Time (lower is better)
scores_time = {'rf': rf_train_time, 'lgb': lgb_train_time, 'knn': knn_train_time, 'dt': dt_train_time}
best_time = min(scores_time, key=scores_time.get)
if best_time == 'rf': rf_wins += 1
elif best_time == 'lgb': lgb_wins += 1
elif best_time == 'knn': knn_wins += 1
else: dt_wins += 1

print(f"\nScore Summary:")
print(f"  Random Forest wins: {rf_wins}/4 metrics")
print(f"  LightGBM wins:      {lgb_wins}/4 metrics")
print(f"  KNN wins:           {knn_wins}/4 metrics")
print(f"  Decision Tree wins: {dt_wins}/4 metrics")

# Determine winner
wins = {'Random Forest': rf_wins, 'LightGBM': lgb_wins, 'KNN': knn_wins, 'Decision Tree': dt_wins}
accuracies = {'Random Forest': rf_accuracy, 'LightGBM': lgb_accuracy, 'KNN': knn_accuracy, 'Decision Tree': dt_accuracy}
cv_scores = {'Random Forest': rf_cv_mean, 'LightGBM': lgb_cv_mean, 'KNN': knn_cv_mean, 'Decision Tree': dt_cv_mean}
f1_scores = {'Random Forest': rf_f1, 'LightGBM': lgb_f1, 'KNN': knn_f1, 'Decision Tree': dt_f1}
max_wins = max(wins.values())
winners = [model for model, w in wins.items() if w == max_wins]

if len(winners) == 1:
    winner = winners[0]
    print(f"\n🏆 OVERALL WINNER: {winner}")
    print(f"   - Test accuracy: {accuracies[winner]*100:.2f}%")
    print(f"   - CV Score: {cv_scores[winner]:.4f}")
    print(f"   - F1 Score: {f1_scores[winner]:.4f}")
else:
    print(f"\n🤝 TIE between: {', '.join(winners)}")

print("\n" + "-" * 80)
print("Model Performance Summary:")
print(f"  Random Forest - Accuracy: {rf_accuracy*100:.2f}%, CV: {rf_cv_mean:.4f}, F1: {rf_f1:.4f}")
print(f"  LightGBM      - Accuracy: {lgb_accuracy*100:.2f}%, CV: {lgb_cv_mean:.4f}, F1: {lgb_f1:.4f}")
print(f"  KNN           - Accuracy: {knn_accuracy*100:.2f}%, CV: {knn_cv_mean:.4f}, F1: {knn_f1:.4f}")
print(f"  Decision Tree - Accuracy: {dt_accuracy*100:.2f}%, CV: {dt_cv_mean:.4f}, F1: {dt_f1:.4f}")

print("\n" + "=" * 80)
print("COMPARISON COMPLETED SUCCESSFULLY")
print("=" * 80)
