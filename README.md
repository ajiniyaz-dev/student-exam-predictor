# Student Exam Predictor

A machine learning application that predicts a student's exam score based on academic, personal, and environmental factors.

The project uses an **ElasticNet Regression** model with data preprocessing and hyperparameter tuning. A **Streamlit** web application provides a simple interface for entering student information and generating predictions.

## Overview

Student performance can be influenced by factors such as study time, attendance, previous scores, motivation, sleep time, access to resources, and family background.

This project uses these factors to train a regression model that estimates a student's expected exam score.

The repository contains both the model training pipeline and a ready-to-use Streamlit application.

## Features

* Predict student exam scores
* ElasticNet Regression model
* Hyperparameter tuning with `GridSearchCV`
* Numerical feature scaling
* Categorical feature encoding
* Missing-value handling
* Pre-trained model included in the repository
* Interactive Streamlit interface
* Model evaluation using R², MAE, and RMSE

## Project Structure

```text
student-exam-predictor/
│
├── app.py
├── train_and_save.py
├── final_elasticnet_model.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

### Main Files

**`app.py`**
The Streamlit application. It collects student information, loads the trained model, and displays the predicted exam score.

**`train_and_save.py`**
Contains the machine learning pipeline used to load the dataset, preprocess the data, train and tune the ElasticNet model, evaluate its performance, and save the trained model.

**`final_elasticnet_model.pkl`**
The pre-trained model and preprocessing pipeline saved with Joblib.

**`requirements.txt`**
Contains the Python packages required to run the project.

## Input Features

The model uses a variety of student-related features, including:

* Hours Studied
* Attendance
* Previous Scores
* Tutoring Sessions
* Sleep Hours
* Physical Activity
* Motivation Level
* Parental Involvement
* Access to Resources
* Teacher Quality
* Peer Influence
* Internet Access
* School Type
* Family Income
* Parental Education Level
* Distance from Home
* Extracurricular Activities
* Learning Disabilities
* Gender

## Machine Learning

The project uses **ElasticNet Regression**, which combines L1 and L2 regularization.

The preprocessing pipeline handles different types of features separately:

* Numerical features are processed with imputation and standardization.
* Ordinal categorical features are encoded according to their natural order.
* Nominal categorical features are converted using one-hot encoding.

`GridSearchCV` is used to find suitable values for the model's `alpha` and `l1_ratio` parameters.

The model is evaluated using:

* R² Score
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

## Installation

Clone the repository:

```bash
git clone https://github.com/ajiniyaz-dev/student-exam-predictor.git
cd student-exam-predictor
```

Create and activate a virtual environment if desired:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python -m venv venv
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

The trained model is already included in the repository, so you can run the application without training the model again.

```bash
streamlit run app.py
```

Streamlit will provide a local URL where you can open the application in your browser.

Enter the required student information and the application will generate an estimated exam score.

## Training the Model

To train the model again from the dataset, run:

```bash
python train_and_save.py
```

The script loads the Student Performance Factors dataset, prepares the features, performs preprocessing and hyperparameter tuning, evaluates the model, and saves the resulting pipeline as:

```text
final_elasticnet_model.pkl
```

## Dataset

The project uses the **Student Performance Factors** dataset, which contains information about different factors that may influence students' exam performance.

The dataset includes academic, social, personal, and environmental features that can be used to study relationships between student characteristics and exam scores.

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Joblib
* Matplotlib
* Seaborn

## Limitations

The predicted score is an estimate based on the patterns learned from the training dataset. It should not be considered a guaranteed prediction of a student's actual exam result.

Model performance may also vary when the model is applied to students or situations that differ significantly from the original dataset.

## Future Improvements

Some possible improvements include:

* Comparing multiple regression models
* Improving the Streamlit interface
* Adding prediction intervals
* Adding automated model retraining

## Author

**Ajiniyaz Bazarbaev**

GitHub: https://github.com/ajiniyaz-dev

If you find the project useful, feel free to star the repository.
