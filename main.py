# Authors: Steven Duong, Harry Lee, Anthony Trieu, Tony Wu
# Project: CMPT 310 Final Project - Career Path Prediction
# Date: Oct 22, 2025
# Description: This file contains the code for the User Interface (UI) of the Career Path Prediction application.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Career Path Predictor", page_icon="🎓", layout="wide")

# Set the title of the app
st.title("Career Path Prediction")
st.markdown(
    """
    Welcome to the **Career Path Prediction** application!
    Please enter your academic details below - once you submit, we'll analyze your strengths and suggest potential career paths.
    """
)

st.divider()

st.subheader("Academic Performance")

col1, col2, col3 = st.columns(3)

with col1:
    math_score = st.number_input("Enter your Math score (0-100):", min_value=0, max_value=100)
    history_score = st.number_input("Enter your History score (0-100):", min_value=0, max_value=100)
    physics_score = st.number_input("Enter your Physics score (0-100):", min_value=0, max_value=100)
with col2:
    chemistry_score = st.number_input("Enter your Chemistry score (0-100):", min_value=0, max_value=100)
    biology_score = st.number_input("Enter your Biology score (0-100):", min_value=0, max_value=100)
    english_score = st.number_input("Enter your English score (0-100):", min_value=0, max_value=100)
with col3:
    geography_score = st.number_input("Enter your Geography score (0-100):", min_value=0, max_value=100)
    study_hours = st.number_input("Enter the number of hours you study each week:", min_value=0, max_value=168)
    absent_days = st.number_input("Enter the number of days you were absent from school this year:", min_value=0, max_value=365)

# Calculate Total score, avg score, max score, min score, and avg score/study hours
total_score = (math_score + history_score + physics_score + chemistry_score + biology_score + english_score + geography_score)
avg_score = total_score / 7
max_score = max(math_score, history_score, physics_score, chemistry_score, biology_score, english_score, geography_score)
min_score = min(math_score, history_score, physics_score, chemistry_score, biology_score, english_score, geography_score)
avg_score_per_hour = avg_score / study_hours if study_hours > 0 else 0

# Data input to be used for the prediction model
input_data = pd.DataFrame({
        'math_score': [math_score],
        'history_score': [history_score],
        'physics_score': [physics_score],
        'chemistry_score': [chemistry_score],
        'biology_score': [biology_score],
        'english_score': [english_score],
        'geography_score': [geography_score],
        'weekly_self_study_hours': [study_hours],
        'absence_days': [absent_days],
        'total_score': [total_score],
        'average_score': [avg_score],
        'best_subject_score': [max_score],
        'worst_subject_score': [min_score],
        'study_efficiency': [avg_score_per_hour]
    })

# Submit button
if st.button("Analyze My Data"):
    st.success("Data submitted successfully!")
    st.write("Below you will find your personalized career path suggestions.")

    st.divider()
    
