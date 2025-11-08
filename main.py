# Authors: Steven Duong, Harry Lee, Anthony Trieu, Tony Wu
# Project: CMPT 310 Final Project - Career Path Prediction
# Date: Oct 11, 2025
# Description: This file contains the code for the data processing and model training.

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
student = pd.read_csv('student-scores.csv')
print("Dataset loaded successfully.")
print("Loading...")

# Remove career_aspiration with unknown values
student = student[student['career_aspiration'] != 'Unknown']

# Remove first 4 columns (ID, first_name, last_name, email) and remove columns with low importance
student = student.iloc[:, 4:]
student = student.drop(['extracurricular_activities', 'part_time_job', 'gender'], axis=1)

# Check for unique values in career_aspiration
print(student['career_aspiration'].value_counts())

# Feature engineering
student['total_score'] = student[['math_score','history_score','physics_score','chemistry_score',
                                  'biology_score','english_score','geography_score']].sum(axis=1)

student['average_score'] = student['total_score'] / 7
student['best_subject_score'] = student[['math_score','history_score','physics_score','chemistry_score',
                                         'biology_score','english_score','geography_score']].max(axis=1)
student['worst_subject_score'] = student[['math_score','history_score','physics_score','chemistry_score',
                                          'biology_score','english_score','geography_score']].min(axis=1)
student['study_efficiency'] = student['average_score'] / student['weekly_self_study_hours'].replace(0, 1)

# Define features and target
X = student.drop('career_aspiration', axis=1)
y = student['career_aspiration']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Classifier
dt = DecisionTreeClassifier(
    criterion='entropy',   # you can also use 'gini'
    max_depth=10,          # limit tree depth to prevent overfitting
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)

# Train the model
dt.fit(X_train, y_train)

# Make predictions
y_pred = dt.predict(X_test)

# 10-fold Cross Validation
cv_scores = cross_val_score(dt, X, y, cv=10)
print(f"10-fold Cross Validation Scores: {cv_scores}")
print(f"Mean CV Score: {np.mean(cv_scores)}")

# Evaluate the model
print("Error Analysis:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted')}")
print(f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted')}")