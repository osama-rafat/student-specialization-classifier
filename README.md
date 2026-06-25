# Student Specialization Classifier

## Overview

Student Specialization Classifier is a Machine Learning and Neural Network project designed to predict the most suitable Computer Science specialization for students based on their skills, interests, and academic background.

The system analyzes student-related features, applies data preprocessing and feature engineering techniques, and uses a Neural Network model to recommend the best specialization.

---

## Predicted Specializations

The model predicts one of the following specializations:

- Artificial Intelligence (AI)
- Cyber Security
- Frontend Development
- Backend Development
- Software Testing

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Neural Networks (MLPClassifier)

---

## Machine Learning Pipeline

### 1. Data Cleaning
- Handle missing values
- Remove inconsistencies
- Prepare data for analysis

### 2. Exploratory Data Analysis (EDA)
- Data distribution analysis
- Feature relationships
- Class distribution analysis

### 3. Feature Engineering
- Create useful features
- Improve model performance

### 4. Feature Selection
- Select the most important features
- Reduce noise and dimensionality

### 5. Data Encoding
- Label Encoding
- One-Hot Encoding

### 6. Feature Scaling
- StandardScaler

### 7. Model Training
- Neural Network (MLPClassifier)

### 8. Model Evaluation
- Accuracy Score
- Confusion Matrix
- Classification Report

### 9. Prediction
- Predict the most suitable specialization for new students

---

## Project Structure

```text
student-specialization-classifier
│
├── images/
│   ├── confusion_matrix.png
│   ├── heatmap.png
│   ├── histogram.png
│   ├── class_distribution.png
│   └── classification_report.png
│
├── students_raw_dirty-1.csv
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Visualizations

### Data Distribution

![Histogram](images/histogram.png)

### Correlation Heatmap

![Heatmap](images/heatmap.png)

### Class Distribution

![Class Distribution](images/class_distribution.png)

### Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

### Classification Report

![Classification Report](images/classification_report.png)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/osama-rafat/student-specialization-classifier.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

---

## Evaluation Metrics

The model is evaluated using:

- Accuracy Score
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## Future Improvements

- Hyperparameter Tuning
- Deep Learning Models
- Model Deployment using Flask
- Web Application Interface
- Larger Dataset
- Advanced Feature Engineering

---

## Author

Osama Rafat

Machine Learning and Data Science Enthusiast

GitHub: https://github.com/osama-rafat
