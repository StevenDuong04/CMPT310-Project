# Authors: Steven Duong, Harry Lee, Anthony Trieu, Tony Wu
# Project: CMPT 310 Final Project - Career Path Prediction
# Date: Oct 23, 2025
# Description: This file contains the code for the data processing and model training.

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

# Load the dataset
student = pd.read_csv('student-scores.csv')
print("Dataset loaded successfully.")
print("Loading...")

# Display the first few rows of the dataset
#print(student.head())

# Check for missing values
#print(student.isnull().sum())

# Remove career_asperation with unknown values
student = student[student['career_aspiration'] != 'Unknown']

# Remove first 4 columns (ID, first_name, last_name, email) and remove columns with low importance
student = student.iloc[:, 4:]
student = student.drop(['extracurricular_activities', 'part_time_job', 'gender'], axis=1)
#print(student.head())

# Basic statistics of the dataset
# print(student.describe())

# Check the data types of each column
# print(student.dtypes)

# Check for unique values in career_aspiration
print(student['career_aspiration'].value_counts())


# Engineer new features
# Sum of all classes
student['total_score'] = student[['math_score','history_score','physics_score','chemistry_score','biology_score','english_score','geography_score']].sum(axis=1)

# Average score across all classes
student['average_score'] = student['total_score'] / 7

# best subject score
student['best_subject_score'] = student[['math_score','history_score','physics_score','chemistry_score','biology_score','english_score','geography_score']].max(axis=1)

# worst subject score
student['worst_subject_score'] = student[['math_score','history_score','physics_score','chemistry_score','biology_score','english_score','geography_score']].min(axis=1)

# study efficiency
student['study_efficiency'] = student['average_score'] / student['weekly_self_study_hours'].replace(0, 1)


# Random Forest Classifier
X = student.drop('career_aspiration', axis=1)
y = student['career_aspiration']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf = RandomForestClassifier(n_estimators=1000, random_state=42, class_weight='balanced', max_depth=20)

# Train the model
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# 10-fold Cross Validation
cv_scores = cross_val_score(rf, X, y, cv=10)
print(f"10-fold Cross Validation Scores: {cv_scores}")
print(f"Mean CV Score: {np.mean(cv_scores)}")

# Evaluate the model
print("Error Analysis:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted')}")
print(f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted')}")
print("\n\n")

# Feature Importance (Removed 3 least important features above based on this)
# importances = pd.Series(rf.feature_importances_, index=X.columns)
# print("Feature Importances:")
# print(importances.sort_values(ascending=False))


# Test prediction example with columns: absence_days, weekly_self_study_hours, 
# math_score, history_score, physics_score, chemistry_score, biology_score, english_score, geography_score,
# total_score, average_score, best_subject_score, worst_subject_score, study_efficiency

# Change in order of above columns
test_student = np.array([5, 5, 88, 53, 50, 58, 84, 53, 96, 482, 68.857, 96, 50, 13.77]).reshape(1, -1)
cols = ['absence_days','weekly_self_study_hours',
        'math_score','history_score','physics_score','chemistry_score',
        'biology_score','english_score','geography_score',
        'total_score','average_score','best_subject_score','worst_subject_score','study_efficiency']

test_student_df = pd.DataFrame(test_student, columns=cols)

# Top 3 predicted career aspirations
predicted_probs = rf.predict_proba(test_student_df)
top_3_indices = np.argsort(predicted_probs[0])[-3:][::-1]
top_3_careers = rf.classes_[top_3_indices]
top_3_probs = predicted_probs[0][top_3_indices]
print("Top 3 predicted career aspirations:")
for career, prob in zip(top_3_careers, top_3_probs):
    print(f"{career}: {prob:.4f}")

print("Done.")