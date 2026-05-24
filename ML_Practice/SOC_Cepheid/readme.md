# Stellar Object Classification using Machine Learning

This project focuses on classifying celestial objects into **Stars**, **Galaxies**, and **Quasars (QSOs)** using photometric data from the Sloan Digital Sky Survey (SDSS).

## Objective

Build a machine learning pipeline that can accurately classify stellar objects using only photometric features and astrophysical color indices.

---

## Dataset Features

The dataset contains:

- Photometric magnitudes:
  - `u`, `g`, `r`, `i`, `z`

- Positional features:
  - `ra`, `dec`

- Engineered color-index features:
  - `u_g`
  - `g_r`
  - `r_i`
  - `i_z`

- Target classes:
  - `STAR`
  - `GALAXY`
  - `QSO`



## Workflow

1. Data Understanding and Cleaning
2. Exploratory Data Analysis (EDA)
3. Outlier and Class Imbalance Analysis
4. Feature Engineering using Color Indices
5. Data Preprocessing with `ColumnTransformer`
6. Model Training using:
   - Logistic Regression
   - Random Forest
7. Cross Validation
8. Feature Importance Analysis
9. Hidden Dataset Evaluation Pipeline



## Models and Performance

| Model | Accuracy |
|---|---|
| Logistic Regression | ~93% |
| Random Forest | ~96% |

Random Forest achieved the best overall performance and handled nonlinear relationships more effectively.



## Key Insights

- Color-index features were highly important for classification.
- Photometric information alone can effectively distinguish celestial objects.
- Rare objects such as quasars were handled successfully despite class imbalance.


## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn



## Hidden Dataset Evaluation

A reusable pipeline function was implemented to:
- load a new hidden CSV dataset
- generate color-index features
- apply preprocessing
- predict object classes
- display evaluation metrics and confusion matrix



## Author
SK
