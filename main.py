# Authors: Steven Duong, Harry Lee, Anthony Trieu, Tony Wu
# Project: CMPT 310 Final Project - Career Path Prediction
# Date: Oct 22, 2025
# Description: This file contains the code for the User Interface (UI) of the Career Path Prediction application.

import streamlit as st
import pandas as pd
import numpy as np
import watchdog.events
import watchdog.observers

# Set the title of the app
st.title("Career Path Prediction")
st.write("Welcome to the Career Path Prediction application! Please enter your details below to get started.")

# Ask for numbered data
math_score = st.number_input("Enter your Math score (0-100):", min_value=0, max_value=100)
history_score = st.number_input("Enter your History score (0-100):", min_value=0, max_value=100)
physics_score = st.number_input("Enter your Physics score (0-100):", min_value=0, max_value=100)
chemistry_score = st.number_input("Enter your Chemistry score (0-100):", min_value=0, max_value=100)
biology_score = st.number_input("Enter your Biology score (0-100):", min_value=0, max_value=100)
english_score = st.number_input("Enter your English score (0-100):", min_value=0, max_value=100)
geography_score = st.number_input("Enter your Geography score (0-100):", min_value=0, max_value=100)

# Ask how many hours spent on studying each week
study_hours = st.number_input("Enter the number of hours you study each week:", min_value=0, max_value=168)

# Calculate Total score, avg score, max score, min score, and avg score/study hours
total_score = (math_score + history_score + physics_score + chemistry_score + biology_score + english_score + geography_score)
avg_score = total_score / 7
max_score = max(math_score, history_score, physics_score, chemistry_score, biology_score, english_score, geography_score)
min_score = min(math_score, history_score, physics_score, chemistry_score, biology_score, english_score, geography_score)
avg_score_per_hour = avg_score / study_hours if study_hours > 0 else 0

# Submit button
if st.button("Submit"):
    st.write(f"Thank you for submitting your details!")
    st.write("Based on the information provided, we will analyze and suggest potential career paths for you.")
    
    
