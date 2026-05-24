# Stellar Classification using Machine Learning 

This project focuses on classifying stars into different stellar types using Machine Learning techniques. The model predicts the type of a star based on its physical and spectral properties such as temperature, luminosity, radius, color, and spectral class.

---

# Project Objective

The goal of this project is to build a Machine Learning classification pipeline capable of predicting stellar classes from astrophysical properties.

The project combines:
- Machine Learning
- Data Preprocessing
- Exploratory Data Analysis (EDA)
- Stellar Astrophysics



# Dataset Information

The dataset contains physical properties of stars such as:

| Feature | Description |
|---|---|
| Temperature | Surface temperature of the star (Kelvin) |
| L | Luminosity relative to the Sun |
| R | Radius relative to the Sun |
| A_M | Absolute Magnitude |
| Color | Stellar color |
| Spectral_Class | Spectral classification |
| Type | Target stellar class |



# Stellar Classes

The target variable (`Type`) contains the following stellar classes:

| Type | Stellar Class |
|---|---|
| 0 | Brown Dwarf |
| 1 | Red Dwarf |
| 2 | White Dwarf |
| 3 | Main Sequence |
| 4 | Supergiant |
| 5 | Hypergiant |



# Project Workflow

## 1. Data Loading

The dataset was loaded using Pandas.

```python
import pandas as pd

df = pd.read_csv('star_data.csv')
```



## 2. Initial Data Exploration

Basic dataset understanding was performed using:

```python
df.head()
df.shape
df.info()
df.describe()
```

This step helped identify:
- Data types
- Feature distributions
- Presence of categorical columns
- Dataset structure



## 3. Exploratory Data Analysis (EDA)

### Univariate Analysis
Performed using:
- Histograms
- Countplots
- Distribution plots

Purpose:
- Understand individual feature distributions
- Detect skewness and outliers

### Bivariate Analysis
Performed using:
- Boxplots
- Scatterplots
- Pairwise relationships

Purpose:
- Analyze relationship between features and target classes
- Understand stellar property variations across star types

### Correlation Analysis
Performed using a heatmap:

```python
sns.heatmap(df.corr(numeric_only=True), annot=True)
```

Purpose:
- Identify strongly correlated features
- Understand physical relationships between stellar properties



# Data Preprocessing

## 1. Categorical Data Cleaning

The `Color` column contained inconsistent naming patterns such as:

- Blue White
- Blue-white
- blue white

These categories were standardized using:

```python
df['Color'] = df['Color'].str.lower()
df['Color'] = df['Color'].str.replace('-', ' ')
```



## 2. Feature and Target Separation

```python
X = df.drop(columns=['Type'])
y = df['Type']
```



## 3. Train-Test Split

The dataset was split into training and testing sets.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### Why `stratify=y`?
To maintain equal class distribution in both training and testing datasets.



# Feature Engineering & Transformation

## ColumnTransformer

A `ColumnTransformer` was used to apply different preprocessing operations to different columns.

### One-Hot Encoding
Applied on:
- Color
- Spectral_Class

### Scaling
Applied on numerical features:
- Temperature
- L
- R
- A_M

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

trf = ColumnTransformer([

    ('color_ohe',
     OneHotEncoder(
         drop='first',
         sparse_output=False,
         handle_unknown='ignore'
     ),
     ['Color']),

    ('spectral_ohe',
     OneHotEncoder(
         drop='first',
         sparse_output=False,
         handle_unknown='ignore'
     ),
     ['Spectral_Class']),

    ('scale',
     MinMaxScaler(),
     ['Temperature', 'L', 'R', 'A_M'])

], remainder='passthrough')
```



# Machine Learning Pipeline

A professional Scikit-learn `Pipeline` was built to combine:
- preprocessing
- model training

```python
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

pipe = Pipeline([
    ('preprocess', trf),
    ('model', DecisionTreeClassifier(random_state=42))
])
```



# Model Training

```python
pipe.fit(X_train, y_train)
```



# Model Prediction

```python
y_pred = pipe.predict(X_test)
```



# Model Evaluation

The model was evaluated using:

- Accuracy Score
- Confusion Matrix
- Classification Report
- Cross Validation

## Accuracy

```python
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
```


## Confusion Matrix

```python
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)
```



## Classification Report

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```



# Cross Validation

To verify model robustness:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(pipe, X, y, cv=10)

print(scores.mean())
```

The model achieved near-perfect classification performance across folds.



# Feature Importance Analysis

Feature importance was extracted from the Decision Tree model to identify which stellar properties contribute most to classification.

```python
model = pipe.named_steps['model']

importance = model.feature_importances_
```

This helps scientifically interpret:
- which stellar properties best distinguish stellar classes
- how astrophysical parameters influence classification



# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn



# Key Machine Learning Concepts Used

- Exploratory Data Analysis (EDA)
- One-Hot Encoding
- Feature Scaling
- ColumnTransformer
- Pipeline
- Decision Tree Classification
- Cross Validation
- Feature Importance



# Scientific Relevance

This project demonstrates how Machine Learning can be applied to astrophysical datasets for stellar classification and scientific interpretation.

The model learns relationships between:
- stellar temperature
- luminosity
- radius
- spectral properties

to classify stars into different evolutionary categories.



# Future Improvements

Possible future extensions:
- Random Forest Classifier
- XGBoost
- Hyperparameter Tuning
- PCA for dimensionality reduction
- Deployment using Flask or Streamlit
- Interactive HR Diagram visualization

---

# Conclusion

This project successfully demonstrates an end-to-end Machine Learning workflow for stellar classification.

The project combines:
- Data Analysis
- Feature Engineering
- Machine Learning
- Astrophysical Interpretation

to build a scientifically meaningful classification system for stars.


