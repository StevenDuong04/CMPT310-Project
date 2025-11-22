# Career Path Recommender - User Guide

Welcome to the Career Path Recommender! This guide will help you set up and use the application to predict career paths based on student academic performance and personal information.

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Running the Application](#running-the-application)
5. [Using the Web Interface](#using-the-web-interface)
6. [Preparing Your Data](#preparing-your-data)
7. [Understanding the Results](#understanding-the-results)
8. [Troubleshooting](#troubleshooting)

## Overview

The Career Path Recommender is a machine learning-powered web application that analyzes student transcripts and predicts the top 3 most suitable career paths. The model considers:

- **Academic Performance**: Scores in Math, History, Physics, Chemistry, Biology, English, and Geography
- **Study Habits**: Weekly self-study hours and absence days
- **Personal Information**: Gender, part-time job status, and extracurricular activities

The system uses a LightGBM (Light Gradient Boosting Machine) model trained on student data with 35+ engineered features to provide accurate career recommendations.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Python 3.7 or higher** - [Download Python](https://www.python.org/downloads/)
- **pip** (Python package installer) - Usually comes with Python
- **Git** (optional) - For cloning the repository

To check if Python is installed, open a terminal/command prompt and run:
```bash
python --version
# or
python3 --version
```

## Installation

### Step 1: Get the Project Files

**Option A: Clone with Git**
```bash
git clone https://github.com/StevenDuong04/CMPT310-Project.git
cd CMPT310-Project
```

**Option B: Download ZIP**
1. Download the ZIP file from the repository
2. Extract it to your desired location
3. Open terminal/command prompt and navigate to the project folder

### Step 2: Create a Virtual Environment

Creating a virtual environment keeps the project dependencies isolated from your system Python installation.

**macOS/Linux:**
```bash
python3 -m venv venv
```

**Windows:**
```bash
python -m venv venv
```

### Step 3: Activate the Virtual Environment

**macOS/Linux:**
```bash
source venv/bin/activate
```

**Windows (PowerShell):**
```bash
venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```bash
venv\Scripts\activate
```

After activation, you should see `(venv)` at the beginning of your terminal prompt.

### Step 4: Install Required Libraries

With the virtual environment activated, install all dependencies:

```bash
pip install -r requirements.txt
```

This will install:
- numpy, pandas - Data manipulation
- scikit-learn - Machine learning utilities
- lightgbm - The ML model
- streamlit - Web interface
- matplotlib, seaborn, plotly - Visualizations
- jupyter, notebook - For testing (optional)

## Running the Application

### Start the Streamlit Web App

With your virtual environment activated, run:

```bash
streamlit run app_recommender.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

If it doesn't open automatically, manually navigate to the URL shown in the terminal.

### Stopping the Application

To stop the application:
1. Go to your terminal window
2. Press `Ctrl + C` (Windows/Linux) or `Cmd + C` (macOS)

### Deactivating the Virtual Environment

When you're done working with the project:

```bash
deactivate
```

## Using the Web Interface

### First Time Setup

1. **Welcome Modal**: When you first open the app, you'll see a welcome modal with instructions
2. Click **"I understand — let's start"** to proceed

### Making Predictions

Follow these steps to get career recommendations:

#### 1. Prepare Your Dataset
You need a CSV or Excel file containing student transcript data. See [Preparing Your Data](#preparing-your-data) for the required format.

#### 2. Upload Your File
1. Click the **"Upload your dataset"** button
2. Select your CSV or XLSX file
3. Wait for the file to upload (you'll see the filename displayed)

#### 3. Submit for Prediction
1. Click the **"Submit"** button
2. The app will process your data and generate predictions
3. Results will appear below showing the **Top 3 Career Recommendations**

#### 4. View Results
The results show:
- **Career Name**: The recommended career path
- **Confidence Percentage**: How confident the model is (0-100%)
- **Progress Bar**: Visual representation of the confidence

#### 5. Try Another File
Click **"Run another file"** to reset and upload a different dataset

## Preparing Your Data

### Required CSV Format

Your CSV file must include the following columns (in any order):

| Column Name | Type | Description | Valid Values |
|-------------|------|-------------|--------------|
| `id` | Integer | Student ID | Any unique number |
| `first_name` | Text | First name | Any text |
| `last_name` | Text | Last name | Any text |
| `email` | Text | Email address | Valid email format |
| `gender` | Text | Gender | "male" or "female" |
| `part_time_job` | Boolean | Has part-time job | "True" or "False" |
| `absence_days` | Integer | Days absent | 0-30 |
| `extracurricular_activities` | Boolean | Participates in activities | "True" or "False" |
| `weekly_self_study_hours` | Integer/Float | Study hours per week | 0-40+ |
| `math_score` | Integer/Float | Math grade | 0-100 |
| `history_score` | Integer/Float | History grade | 0-100 |
| `physics_score` | Integer/Float | Physics grade | 0-100 |
| `chemistry_score` | Integer/Float | Chemistry grade | 0-100 |
| `biology_score` | Integer/Float | Biology grade | 0-100 |
| `english_score` | Integer/Float | English grade | 0-100 |
| `geography_score` | Integer/Float | Geography grade | 0-100 |

### Example CSV

```csv
id,first_name,last_name,email,gender,part_time_job,absence_days,extracurricular_activities,weekly_self_study_hours,math_score,history_score,physics_score,chemistry_score,biology_score,english_score,geography_score
1,John,Doe,john.doe@email.com,male,False,3,True,25,85,78,90,88,82,79,75
2,Jane,Smith,jane.smith@email.com,female,True,5,True,20,92,88,85,90,87,91,84
```

### Sample Dataset

A sample transcript is provided in the repository:
- Location: `STUDENT_TRANSCRIPTS/user1.csv`
- Use this as a template for creating your own datasets

### Important Notes

- **File Encoding**: Use UTF-8 encoding for your CSV files to avoid parsing errors
- **Headers Required**: The first row must contain column names
- **Missing Data**: Ensure all required fields are filled
- **Score Range**: All scores should be between 0-100
- **Multiple Students**: You can include multiple students in one file; the app will predict the average career path recommendation

## Understanding the Results

### Career Predictions

The model predicts careers from the following categories:
- Lawyer
- Doctor
- Government Officer
- Artist
- Unknown (if data is ambiguous)
- Business Owner
- Scientist
- Software Engineer
- Teacher
- Engineer
- Chartered Accountant
- Designer
- Construction Engineer
- Game Developer
- Stock Investor
- Real Estate Developer
- Banker

### How Predictions Work

The model analyzes **35+ engineered features**, including:

1. **Academic Metrics**:
   - Average scores across all subjects
   - Subject-specific strengths (Science vs. Humanities vs. Math)
   - Score consistency (standard deviation and range)

2. **Domain-Specific Features**:
   - Science average (Physics, Chemistry, Biology)
   - Humanities average (History, Geography, English)
   - STEM score (Math + Science combination)

3. **Behavioral Indicators**:
   - Study efficiency (grades per study hour)
   - Engagement score (study time, attendance, activities)
   - Balanced student indicator (moderate study + activities)

4. **Career Orientation Flags**:
   - STEM-oriented (high Math + Science)
   - Business-oriented (high Math + English)
   - Creative-oriented (high English + Humanities)

### Confidence Levels

- **75-100%**: Very high confidence - Strong match
- **50-75%**: High confidence - Good match
- **25-50%**: Moderate confidence - Possible match
- **0-25%**: Low confidence - Weak match

### Top 3 Recommendations

The app shows the top 3 career paths sorted by confidence. This allows you to:
- See the best match (top recommendation)
- Explore alternative career paths (2nd and 3rd)
- Compare confidence levels between options

## Troubleshooting

### Common Issues and Solutions

#### Issue: "Couldn't read your file" Error

**Cause**: File encoding or format issue

**Solutions**:
1. Save your CSV file with UTF-8 encoding:
   - In Excel: File → Save As → Tools → Web Options → Encoding → Unicode (UTF-8)
   - In Google Sheets: File → Download → Comma-separated values (.csv)
2. Ensure all required columns are present
3. Check for special characters in the data
4. Verify the file isn't corrupted

#### Issue: Virtual Environment Won't Activate (Windows PowerShell)

**Cause**: PowerShell execution policy restriction

**Solution**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try activating again.

#### Issue: Streamlit Command Not Found

**Cause**: Package not installed or virtual environment not activated

**Solutions**:
1. Ensure virtual environment is activated (look for `(venv)` in prompt)
2. Reinstall dependencies: `pip install -r requirements.txt`
3. Try running with full path: `python -m streamlit run app_recommender.py`

#### Issue: Port Already in Use

**Cause**: Another application is using port 8501

**Solution**:
```bash
streamlit run app_recommender.py --server.port 8502
```

#### Issue: Missing Model Files

**Cause**: The pre-trained model files are missing

**Solution**:
Ensure these files exist in the `saved_model/` directory:
- `lgb_model.pkl`
- `label_encoder_career.pkl`
- `label_encoder_gender.pkl`

If missing, you may need to retrain the model (see DEVELOPER_GUIDE.md).

### Getting Additional Help

If you encounter issues not covered here:

1. **Check the terminal output** for specific error messages
2. **Review your CSV file format** against the example
3. **Ensure all dependencies are installed** correctly
4. **Try the sample dataset** (`STUDENT_TRANSCRIPTS/user1.csv`) to verify the app works

## Tips for Best Results

1. **Accurate Data**: Ensure all scores and information are accurate
2. **Complete Profiles**: Fill in all required fields for better predictions
3. **Multiple Students**: The app averages predictions across all students in the file
4. **Interpret Wisely**: Use predictions as guidance, not absolute truth
5. **Consider All Three**: Review all top 3 recommendations, not just the first

---

**Authors**: Steven Duong, Harry Lee, Anthony Trieu, Tony Wu
**Course**: CMPT 310 - Artificial Intelligence
**Date**: November 2025
