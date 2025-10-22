# Authors: Steven Duong, Harry Lee, Anthony Trieu, Tony Wu
# Project: CMPT 310 Final Project - Career Path Prediction
# Date: Oct 22, 2025
# Description: This file contains the code for the data processing and model training.

# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors

# Load dataset
unis = pd.read_csv('canadian_universities_careers.csv')

# Encode Career column
le = LabelEncoder()
unis['Career_Code'] = le.fit_transform(unis['Career'])

# Features to use for matching
features = ['General_GPA','Career_Code']
X = unis[features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit KNN
knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
knn.fit(X_scaled)

def recommend_universities(student_gpa, student_career):
    career_code = le.transform([student_career])[0]
    student_input = pd.DataFrame({'General_GPA':[student_gpa], 'Career_Code':[career_code]})
    student_scaled = scaler.transform(student_input)
    
    distances, indices = knn.kneighbors(student_scaled)
    recommendations = unis.iloc[indices[0]][['University', 'Career']]
    
    output = []
    for i, row in recommendations.iterrows():
        output.append(f"{row['University']} - {row['Career']}")
    return output

# Example usage (General_GPA: put in a value between 0-100, Career: choose from available careers in dataset)
top3 = recommend_universities(80, 'Software Engineer')
print("Top 3 recommended universities and programs:")
for i, rec in enumerate(top3, 1):
    print(f"{i}. {rec}")
