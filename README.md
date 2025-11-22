# Career Path Recommender

> A machine learning-powered career recommendation system using LightGBM to predict optimal career paths based on student academic performance and personal characteristics.

**Authors**: Steven Duong, Harry Lee, Anthony Trieu, Tony Wu
**Course**: CMPT 310 - Artificial Intelligence
**Institution**: Simon Fraser University
**Date**: November 2025

---

## Overview

The Career Path Recommender is an intelligent web application that analyzes student transcripts and provides personalized career recommendations. By examining academic performance across multiple subjects, study habits, and personal characteristics, our system predicts the top 3 most suitable career paths with confidence scores.

### Key Features

- **Intelligent Predictions**: Uses LightGBM model with 35+ engineered features
- **Web-Based Interface**: User-friendly Streamlit application
- **Multiple Career Paths**: Recommends top 3 careers with confidence percentages
- **Comprehensive Analysis**: Considers academics, study habits, and personal traits
- **Flexible Input**: Accepts CSV or Excel file uploads
- **Visual Feedback**: Progress bars and clear confidence metrics

### Supported Career Paths

The model can predict from 17 different career categories:
- Engineer
- Software Engineer
- Construction Engineer
- Game Developer
- Doctor
- Scientist
- Teacher
- Lawyer
- Chartered Accountant
- Designer
- Artist
- Business Owner
- Banker
- Stock Investor
- Real Estate Developer
- Government Officer
- Unknown

---

## Quick Start

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/StevenDuong04/CMPT310-Project.git
   cd CMPT310-Project
   ```

2. **Create virtual environment**

   macOS/Linux:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

   Windows:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

```bash
streamlit run app_recommender.py
```

The application will open in your browser at `http://localhost:8501`

To stop the application: Press `Ctrl + C` (or `Cmd + C` on macOS)

### Deactivate Virtual Environment

When finished:
```bash
deactivate
```

---

## Usage

1. **Launch the app** using the command above
2. **Upload a CSV/Excel file** containing student transcript data
3. **Click Submit** to generate predictions
4. **View the top 3 career recommendations** with confidence scores
5. **Click "Run another file"** to try a different dataset

For detailed instructions, see the [**USER_GUIDE.md**](USER_GUIDE.md)

### Sample Data

A sample transcript is provided for testing:
- Location: `STUDENT_TRANSCRIPTS/user1.csv`
- Use this to understand the required data format

---

## Technology Stack

### Machine Learning
- **LightGBM**: Gradient boosting framework for the prediction model
- **scikit-learn**: Data preprocessing and model evaluation
- **NumPy & Pandas**: Data manipulation and analysis

### Web Interface
- **Streamlit**: Interactive web application framework

### Visualization
- **Matplotlib**: Model performance visualizations
- **Seaborn**: Statistical data visualization
- **Plotly**: Interactive plots

### Development Tools
- **Jupyter Notebook**: Exploratory data analysis and prototyping

---

## Project Structure

```
CMPT310-Project/
├── app_recommender.py          # Main Streamlit web application
├── model.py                    # ML model and preprocessing logic
├── requirements.txt            # Python dependencies
├── student-scores.csv          # Training dataset (489 KB)
│
├── saved_model/                # Pre-trained model files
│   ├── lgb_model.pkl          # Trained LightGBM model
│   ├── label_encoder_career.pkl   # Career label encoder
│   └── label_encoder_gender.pkl   # Gender label encoder
│
├── STUDENT_TRANSCRIPTS/        # Sample student data
│   └── user1.csv              # Example transcript file
│
├── testing_files/              # Model testing and evaluation
│   ├── test_model.py          # Model performance evaluation script
│   ├── run.py                 # Interactive prediction script
│   └── Testing.ipynb          # Jupyter notebook for testing
│
├── plots/                      # Model performance visualizations
│   ├── lgb_test_confusion_matrix.png
│   ├── lgb_test_feature_importance.png
│   └── lgb_test_per_class_accuracy.png
│
├── README.md                   # This file
├── USER_GUIDE.md              # Detailed user instructions
└── DEVELOPER_GUIDE.md         # Technical documentation (optional)
```

---

## Model Information

### Algorithm: LightGBM (Light Gradient Boosting Machine)

Our model uses LightGBM, a high-performance gradient boosting framework that:
- Handles complex non-linear relationships
- Prevents overfitting through regularization
- Provides feature importance insights
- Achieves high accuracy on multi-class classification

### Feature Engineering (35+ Features)

The model analyzes:

**Academic Performance**:
- Individual subject scores (7 subjects)
- Average score, best/worst subject scores
- Science, Humanities, and STEM averages
- Score consistency (standard deviation, range)

**Study Behavior**:
- Weekly self-study hours
- Absence days
- Study efficiency (score per hour ratio)
- Engagement score

**Personal Characteristics**:
- Gender
- Part-time job status
- Extracurricular activities

**Advanced Indicators**:
- Subject dominance ratios (Math, Science, Humanities)
- Career orientation flags (STEM, Business, Creative)
- Performance tiers (Top performer, Struggling student)
- Dedication indicators

### Model Performance

The model achieves strong performance metrics:
- **Cross-validation accuracy**: ~88-92% (10-fold CV)
- **Test set accuracy**: ~90%
- **Weighted F1-score**: ~0.89

See visualizations in the `plots/` directory for detailed performance analysis.

---

## File Descriptions

| File | Purpose |
|------|---------|
| `app_recommender.py` | Streamlit web application with UI components |
| `model.py` | Data preprocessing and prediction logic |
| `test_model.py` | Model training and evaluation script |
| `run.py` | Command-line interface for predictions |
| `student-scores.csv` | Training dataset with ~6000 student records |
| `requirements.txt` | Python package dependencies |

---

## Input Data Format

Your CSV file must include these columns:

| Column | Type | Description |
|--------|------|-------------|
| `id`, `first_name`, `last_name`, `email` | Metadata | Student identification |
| `gender` | Text | "male" or "female" |
| `part_time_job` | Boolean | True/False |
| `absence_days` | Integer | 0-30 days |
| `extracurricular_activities` | Boolean | True/False |
| `weekly_self_study_hours` | Number | Hours per week |
| `math_score`, `history_score`, `physics_score`, `chemistry_score`, `biology_score`, `english_score`, `geography_score` | Integer | Scores (0-100) |

See [USER_GUIDE.md](USER_GUIDE.md) for detailed format specifications and examples.

---

## Development

### Model Training

To retrain the model with new data:

```bash
cd testing_files
python test_model.py
```

This will:
1. Load and preprocess the training data
2. Perform feature engineering
3. Train the LightGBM model with 10-fold cross-validation
4. Evaluate performance metrics
5. Generate visualization plots
6. Save the trained model to `saved_model/`

### Testing

Run the interactive prediction script:

```bash
cd testing_files
python run.py
```

This allows you to:
- Enter student data manually
- Select different model types (KNN, Random Forest, LightGBM)
- View predictions with confidence scores

---

## Documentation

- **[USER_GUIDE.md](USER_GUIDE.md)**: Comprehensive user instructions
- **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)**: Technical documentation for developers (optional)

---

## Troubleshooting

### Common Issues

**"Couldn't read your file" error**
- Ensure CSV is UTF-8 encoded
- Verify all required columns are present

**Streamlit command not found**
- Activate virtual environment first
- Reinstall requirements: `pip install -r requirements.txt`

**Port already in use**
```bash
streamlit run app_recommender.py --server.port 8502
```

For more troubleshooting help, see [USER_GUIDE.md](USER_GUIDE.md#troubleshooting)

---

## Contributing

This is an academic project for CMPT 310 at Simon Fraser University.

**Team Members**:
- Steven Duong
- Harry Lee
- Anthony Trieu
- Tony Wu

---

## Acknowledgments

- **Course**: CMPT 310 - Introduction to Artificial Intelligence
- **Institution**: Simon Fraser University
- **Dataset**: https://www.kaggle.com/datasets/markmedhat/student-scores
- **Libraries**: Streamlit, LightGBM, scikit-learn, and all open-source contributors

