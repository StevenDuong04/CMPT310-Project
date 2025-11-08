# Authors: Steven Duong, Harry Lee, Anthony Trieu, Tony Wu
# Project: CMPT 310 Final Project - Career Path Prediction
# Date: Oct 11, 2025
# Description: This file contains the code for the data processing and model training.

# Import necessary libraries
import pandas as pd
import numpy as np

# Load the dataset
student = pd.read_csv('student-scores.csv')
print("Dataset loaded successfully.")

# Display the first few rows of the dataset
print(student.head())

# Check for missing values
print(student.isnull().sum())

# Remove career_asperation with unknown values
student = student[student['career_aspiration'] != 'Unknown']

# Remove first 4 columns (ID, first_name, last_name, email)
student = student.iloc[:, 4:]
print(student.head())

# Basic statistics of the dataset
print(student.describe())

# Check the data types of each column
print(student.dtypes)

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
