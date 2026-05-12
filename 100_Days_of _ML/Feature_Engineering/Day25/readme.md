topic covered in this video:

Feature Scaling - Normalization | MinMaxScaling | MaxAbsScaling | RobustScaling

Video Link: https://youtu.be/eBrGyuA2MIg

---

# **Feature Scaling: Normalization Techniques**

This document provides a comprehensive summary of **Normalization**, a critical feature engineering technique used to change the values of numeric columns in a dataset to a common scale without distorting differences in the ranges of values.



### **1. Core Concept of Normalization**
Normalization is applied during data preparation to ensure that the **magnitude** of features does not negatively impact machine learning models. It effectively eliminates the influence of different units (e.g., weight in kg vs. height in cm) so the model can process values on a common scale.



### **2. Min-Max Scaling (The Most Common Technique)**
Min-Max Scaling is the most widely used normalization method and is often what people mean when they say "Normalization".

*   **Formula:** $x' = \frac{x - x_{min}}{x_{max} - x_{min}}$
*   **Resulting Range:** Always scales data between **0 and 1**.
*   **Geometric Intuition:** It "squashes" or compresses the data into a **unit hypercube** (a 1x1 square in 2D or a 1x1x1 cube in 3D).
*   **Key Characteristic:** While it changes the scale of the axes, it generally **preserves the original shape** of the data distribution.
*   **Major Drawback:** It is **highly sensitive to outliers**. An extreme outlier will compress all other "normal" data points into a very tiny range, making them hard for the model to distinguish.



### **3. Other Normalization Techniques**

#### **Mean Normalization**
*   **Formula:** $x' = \frac{x - \text{mean}}{x_{max} - x_{min}}$
*   **Range:** Resulting values fall between **-1 and 1**.
*   **Usage:** It is used when centered data is required, but it is rarely used in practice. Notably, **Scikit-Learn does not have a built-in class** for this; it must be implemented manually.

#### **Max Absolute Scaling (`MaxAbsScaler`)**
*   **Formula:** $x' = \frac{x}{|x_{max}|}$
*   **Primary Use Case:** Specifically recommended for **Sparse Data** (datasets containing many zeros).

#### **Robust Scaling (`RobustScaler`)**
*   **Formula:** $x' = \frac{x - \text{median}}{75^{th} \text{ percentile} - 25^{th} \text{ percentile}}$ (Interquartile Range).
*   **Key Advantage:** It is **robust to outliers**. If your dataset has many extreme values that you cannot remove, this is the best scaling choice.



### **4. Implementation Workflow**
When using Scikit-Learn (`MinMaxScaler`, `RobustScaler`, etc.), follow these rules:
1.  **Train-Test Split:** Always split your data before scaling to prevent data leakage.
2.  **Fit Only on Training Data:** Use `.fit()` on the training set to learn the parameters (min, max, etc.).
3.  **Transform Both:** Apply `.transform()` to both the training and testing sets using the learned parameters.



### **5. Normalization vs. Standardization: When to Use What?**

| **Technique** | **Best Use Case** |
| :--- | :--- |
| **Standardization** | The **default choice** for most problems; usually gives better results in most algorithms. |
| **Min-Max Scaling** | Use when you **know the range** of your data in advance (e.g., **Image Processing** where pixels are 0–255). |
| **Robust Scaling** | Use when your data has **significant outliers**. |
| **Max Abs Scaling** | Use for **Sparse Matrices** (data with many zeros). |
| **Tree-based Models** | **No scaling required** (Decision Trees, Random Forests, XGBoost, etc.). |



### **Final Pro-Tip**
Machine learning is experimental. If you are unsure which scaler to use, it is often beneficial to **try multiple techniques** and compare the resulting model performance.