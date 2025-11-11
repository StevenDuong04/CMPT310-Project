# Authors: Steven Duong, Harry Lee, Anthony Trieu, Tony Wu
# Project: CMPT 310 Final Project - Career Path Prediction
# Date: Nov 10, 2025
# Description: This file contains the code for the data processing and model training.

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load the pre-trained model and label encoders
lgb_model = joblib.load('saved_model/lgb_model.pkl')
label_encoder_career = joblib.load('saved_model/label_encoder_career.pkl')
label_encoder_gender = joblib.load('saved_model/label_encoder_gender.pkl')

# Preprocess input data
def preprocess_input(uploaded_file: str) -> pd.DataFrame:
    student = pd.read_csv(uploaded_file)
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

    return rf_data[selected_features]

# Predict career paths
def predict_career_path(uploaded_file) ->list[tuple[str, float]]:
    data = preprocess_input(uploaded_file)
    predicted_probs = lgb_model.predict_proba(data)
    avg_probs = np.mean(predicted_probs, axis=0)
    top3_indices = np.argsort(avg_probs)[-3:][::-1]
    
    top3_careers = label_encoder_career.inverse_transform(top3_indices)
    top3_probs = avg_probs[top3_indices]
    return list(zip(top3_careers, top3_probs))