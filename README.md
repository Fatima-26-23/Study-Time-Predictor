## Overview

The Study Hours Predictor is a machine learning-powered web application that recommends the minimum number of study hours a student needs to achieve a target academic performance score. Given student-specific inputs, the system uses an inverse-prediction approach over a trained regression model to suggest an optimal study plan.
## Features

Predicts required daily study hours based on a target Performance Index
Takes into account sleep schedule, previous scores, and practice paper history
Uses a Linear Regression model trained on 9,766 student records
Interactive web interface built with Streamlit
Penalizes over-prediction to ensure realistic, conservative recommendations

## Project Structure

Project/
├── Project.ipynb        # Data preprocessing & model training notebook
├── App.py               # Streamlit web application
├── model.pkl            # Trained LinearRegression model (generated)
├── scaler.pkl           # Fitted StandardScaler (generated)
├── Student_Performance.csv   # Raw dataset (source)
└── DATASET final.csv    # Cleaned dataset after preprocessing

## Dataset
Dataset Name: Student Performance
Source: https://www.kaggle.com/datasets/nikhil7280/student-performance-
multiple-linear-regression/data — 10,000 records of student performance data. After removing 234 duplicate rows, 9,766 unique records were used for training.

## Columns:
1. Hours Studied
○ Represents the number of hours a student studies per day.
○ Data Type: Numeric
2. Previous Scores
○ Indicates the student’s previous academic scores.
○ Data Type: Numeric
3. Sleep Hours
○ Average number of hours a student sleeps per day.
○ Data Type: Numeric
4. Sample Question Papers Practiced
○ Number of practice question papers completed by the student.
○ Data Type: Numeric
5. Extracurricular Activities
○ Indicates whether the student participates in extracurricular
activities.
○ Values: Yes / No
○ Data Type: Categorical
6. Performance Index
○ Represents the overall academic performance of the student.
○ Data Type: Numeric

Note: The Extracurricular Activities column was dropped before training as it is a categorical variable that was excluded from the final feature set.

## Data Preprocessing

Loaded raw CSV using pandas
Inspected shape, data types, and null values (no nulls found)
Removed 234 duplicate rows (10,000 → 9,766 records)
Dropped Extracurricular Activities column
Saved clean data to DATASET final.csv
Model Training
Algorithm: Scikit-learn Linear Regression
Features (X): Hours Studied, Previous Scores, Sleep Hours, Sample Question Papers Practiced
Target (y): Performance Index

## Training Pipeline

Features scaled using StandardScaler
80/20 train-test split with random_state=42
Model and scaler serialized to model.pkl and scaler.pkl using pickle

## Model Performance
MAE (Mean Absolute Error)
1.64
RMSE (Root Mean Squared Error)
2.05
R² Score
0.9887

The model achieves an R² of 0.9887, indicating it explains over 98.8% of the variance in student performance — a very strong fit for a linear model.
Inverse Prediction Logic
Since the trained regression model predicts Performance Index given Hours Studied (not the reverse), an optimization-based search is employed to find the required study hours:
Iterates over candidate study hour values from 1 to 12
Runs each through the scaler and model to get a predicted Performance Index
Selects the hour count whose prediction is closest to the target
Applies a penalty of +5 to the error when the predicted score exceeds the target, to avoid over-recommendation

## Installation & Usage
### Prerequisites
pip install streamlit numpy scikit-learn pandas

Step 1: Train the Model
Open and run all cells in Project.ipynb. This will generate model.pkl and scaler.pkl in the project directory.
Step 2: Launch the App
streamlit run App.py

Step 3: Enter Inputs

## Notes

model.pkl and scaler.pkl must be present in the same directory as App.py before launching the Streamlit app.
The study hours search space is bounded to 1–12 hours. Targets requiring more than 12 hours of study may not be achievable with this model.
All input fields accept integer values only.
