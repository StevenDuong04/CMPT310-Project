import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')
import os
import sys


# ============================================================================
# THe features
# ============================================================================

# "Calculate from user inputed data"
def calculate_features(user_data):  
    # studenty scores
    scores = [
        user_data['biology_score'],
        user_data['chemistry_score'],
        user_data['english_score'], 
        user_data['geography_score'],
        user_data['history_score'], 
        user_data['physics_score'],
        user_data['math_score']]     
    
    #get average score
    avg_score = np.mean(scores)
    
    # career field averages
    science_avg = np.mean([user_data['biology_score'],
                           user_data['chemistry_score'],
                           user_data['physics_score']])
    
    humanities_avg = np.mean([user_data['english_score'], 
                              user_data['geography_score'],
                              user_data['history_score']])
    
    stem_score = (user_data['math_score'] + science_avg) / 2
    
    # Score statistics
    score_std = np.std(scores)
    score_range = max(scores) - min(scores)
    
    # Engagement score( chose heuristic weights)
    engagement_score = (
        user_data['weekly_self_study_hours']* 0.3 +
        (10 - user_data['absence_days']) * 2 +
        user_data['extracurricular_activities'] * 3
    )
    
    # model student stats would most likely be balance in studies and activities
    balanced_student = int(
        (user_data['weekly_self_study_hours'] >= 10) and (user_data['weekly_self_study_hours'] <= 30) and (user_data['extracurricular_activities'] == 1)
    )
    
    # The studens preference for subject area then groups it into the categories for averaging
    science_max = max(user_data['physics_score'], user_data['chemistry_score'], user_data['biology_score'])
    humanities_max = max(user_data['history_score'], user_data['geography_score'], user_data['english_score'])
    math_score = user_data['math_score']
    overall_max = max(science_max, humanities_max, math_score)
    
    if overall_max == science_max:
        subject_preference = 2
    elif overall_max == humanities_max:
        subject_preference = 1
    elif overall_max == math_score:
        subject_preference = 0
    
    # student study hours
    study_hours = max(user_data['weekly_self_study_hours'], 1)
    # efficiency makes sure that their grades reflect their study time
    study_efficiency = avg_score / study_hours
    
 # ADVANCE FEATURES (CHATGPT V5 GENERATED)
    math_dominance = user_data['math_score'] / avg_score if avg_score > 0 else 0
    science_dominance = science_avg / avg_score if avg_score > 0 else 0
    humanities_dominance = humanities_avg / avg_score if avg_score > 0 else 0
    
    top_performer = int(avg_score > 85)
    struggling_student = int(avg_score < 70)
    high_dedication = int((user_data['weekly_self_study_hours'] > 25) and (user_data['absence_days'] < 3))
    extrovert_indicator = user_data['extracurricular_activities']
    
    stem_oriented = int((user_data['math_score'] > 80) and (science_avg > 80))
    business_oriented = int((user_data['math_score'] > 75) and (user_data['english_score'] > 75))
    creative_oriented = int((user_data['english_score'] > 80) and (humanities_avg > 80))
    
    study_score_interaction = user_data['weekly_self_study_hours'] * avg_score / 100
    absence_impact = user_data['absence_days'] * (100 - avg_score) / 100
    math_science_gap = abs(user_data['math_score'] - science_avg)
    science_humanities_gap = abs(science_avg - humanities_avg)
    
    # Compile all features in correct order (AI help chatgptv5generate)
    features = [
        user_data['gender'], user_data['part_time_job'], user_data['absence_days'],
        user_data['extracurricular_activities'], user_data['weekly_self_study_hours'],
        user_data['math_score'], user_data['history_score'], user_data['physics_score'],
        user_data['chemistry_score'], user_data['biology_score'], user_data['english_score'],
        user_data['geography_score'], science_avg, humanities_avg, stem_score,
        score_std, score_range, engagement_score, balanced_student, subject_preference,
        study_efficiency, avg_score, math_dominance, science_dominance,
        humanities_dominance, top_performer, struggling_student, high_dedication,
        extrovert_indicator, stem_oriented, business_oriented, creative_oriented,
        study_score_interaction, absence_impact, math_science_gap, science_humanities_gap
    ]
    
    return features

# ============================================================================
# Get the users inputs here
# ============================================================================

def user_input():
    print("\n" + "=" * 80)
    print("Enter your statistics below:")
    print("=" * 80)
    print()
    
    user_data = {}
    
    # Gender
    gender_input = input("Gender (male or female?): ").strip().lower()
    
    if gender_input == 'male':
        user_data['gender'] = 1
    else:
        user_data['gender'] = 0
    
    # Part-time job
    job_input = input("Do you have a part-time job? (yes/no): ").strip().lower()
    
    if job_input == 'yes':
        user_data['part_time_job'] = 1
    else:
        user_data['part_time_job'] = 0
   
    
    
    
    # Extracurricular activities
    extra_input = input("Do you participate in extracurricular activities? (yes/no): ").strip().lower()

    if extra_input == 'yes':
        user_data['extracurricular_activities'] = 1
    else:
        user_data['extracurricular_activities'] = 0
    
    
    # Study hours
    study_hours_input = input("Weekly self-study hours: ")
    user_data['weekly_self_study_hours'] = float(study_hours_input)
    
    # Absence days
    user_data['absence_days'] = int(input("Number of absence days (0-30): "))


    print("\nEnter your grades (0-100 %):")
    user_data['math_score'] = float(input("  Mathematics: "))
    user_data['history_score'] = float(input("  History: "))
    user_data['physics_score'] = float(input("  Physics: "))
    user_data['chemistry_score'] = float(input("  Chemistry: "))
    user_data['biology_score'] = float(input("  Biology: "))
    user_data['english_score'] = float(input("  English: "))
    user_data['geography_score'] = float(input("  Geography: "))
    
    return user_data

# ============================================================================
# Load Models
# ============================================================================


def load_model(model_choice):
    """Load selected model"""
    model_dir = f"{os.path.dirname(__file__)}/trained_models"
    
    try:
        if model_choice == '1':
            model_path = f"{model_dir}/knn_model.pkl"
            model = joblib.load(model_path)
            print(f"✓ Loaded KNN model from {model_path}\n")
        elif model_choice == '2':
            model_path = f"{model_dir}/random_forest_model.pkl"
            model = joblib.load(model_path)
            print(f"✓ Loaded Random Forest model from {model_path}\n")
        elif model_choice == '3':
            model_path = f"{model_dir}/lightgbm_model.pkl"
            model = joblib.load(model_path)
            print(f"✓ Loaded LightGBM model from {model_path}\n")
        else:
            print("Invalid model selection")
            return None, None
        
        # Load label encoder
        encoder_path = f"{model_dir}/label_encoder.pkl"
        label_encoder = joblib.load(encoder_path)
        
        return model, label_encoder
    
    except FileNotFoundError:
        print(f"\n[ERROR] Model file not found!")
        print(f"Please train the model first using train_model.py")
        return None, None


# ============================================================================
# Prediction (CHATGPT V5 GENERATED)
# ============================================================================

def predict_career(model, label_encoder, user_data):
    """Make career prediction"""
    
    # Calculate features
    features = calculate_features(user_data)
    
    # Make prediction
    prediction = model.predict([features])[0]
    probabilities = model.predict_proba([features])[0]
    
    # Get career name
    career_name = label_encoder.inverse_transform([prediction])[0]
    
    # Get top 3 predictions
    top_3_indices = probabilities.argsort()[-3:][::-1]
    top_3_careers = [(label_encoder.inverse_transform([idx])[0], probabilities[idx]) 
                     for idx in top_3_indices]
    
    return career_name, probabilities[prediction], top_3_careers

def display_prediction(career, confidence, top_3):
    """Display prediction results"""
    print("\n" + "=" * 80)
    print("CAREER PREDICTION RESULTS")
    print("=" * 80)
    print(f"\n Recommended Career Path: {career}")
    print(f"   Confidence: {confidence*100:.2f}%\n")
    
    print("Top 3 Career Matches:")
    for i, (career_name, prob) in enumerate(top_3, 1):
        print(f"   {i}. {career_name}: {prob*100:.2f}%")
    
    print("\n" + "=" * 80)

# ============================================================================
# Main Functions
# ============================================================================

def predict_mode(model_choice):
    """Run prediction with selected model"""
    model, label_encoder = load_model(model_choice)
    
    if model is None:
        return
    
    while True:
        # Get user input
        user_data = user_input()
        
        # Make prediction
        career, confidence, top_3 = predict_career(model, label_encoder, user_data)
        
        # Display results
        display_prediction(career, confidence, top_3)
        
        # Ask if user wants to continue
        again = input("\nMake another prediction? (yes/no): ").strip().lower()
        if again != 'yes':
            break

def train_mode():
    """Train models - calls training scripts"""
    print("\n[Training Mode]")
    print("Run one of the following training scripts to train the models:")
    print("  - python train_knn.py")
    print("  - python train_random_forest.py")
    print("  - python train_lightgbm.py")
   

# Maybe we dont need this part?? copied over from my personal project
def demo_mode():
    """Demo with sample data"""
    print("\n[Demo Mode]")
    model_choice = input("[Select Model]\n\n [1] KNN \n [2] Random Forest \n [3] LightGBM \n\n Select: ")
    
    model, label_encoder = load_model(model_choice)
    
    if model is None:
        return
    
    # Sample data
    print("\nUsing sample student data...")
    sample_data = {
        'gender': 1,  # Male
        'part_time_job': 0,
        'absence_days': 5,
        'extracurricular_activities': 1,
        'weekly_self_study_hours': 20,
        'math_score': 85,
        'history_score': 70,
        'physics_score': 88,
        'chemistry_score': 82,
        'biology_score': 80,
        'english_score': 75,
        'geography_score': 72
    }
    
    # Make prediction
    career, confidence, top_3 = predict_career(model, label_encoder, sample_data)
    
    # Display results
    display_prediction(career, confidence, top_3)

# ============================================================================
# Main Program COpied from personal project
# ============================================================================

print("=" * 80)
print("CAREER PATH PREDICTION SYSTEM")
print("=" * 80)
print("""
    [1] Predict Career
    [2] Just training models
    [3] Demo 
    [4] Exit
""")

run_mode = input("Select Mode: ")

if run_mode == '1':
    model_choice = input("\n[Select Model]\n\n [1] KNN \n [2] Random Forest \n [3] LightGBM \n\n Select: ")
    predict_mode(model_choice)

elif run_mode == '2':
    train_mode()

elif run_mode == '3':
    demo_mode()

elif run_mode == '4':
    print("\nThank you for using Career Path Prediction System!")
    print("Goodbye! :-)")
    sys.exit()

else:
    print("Input error")
    sys.exit()
