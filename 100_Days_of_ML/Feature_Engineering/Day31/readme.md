Video Link : https://youtu.be/lV_Z4HbNAx0

---

# Power Transformer: Box-Cox and Yeo-Johnson Transformations

In machine learning, many algorithms—especially linear models like **Linear Regression**, **Logistic Regression**, and **SVM**—perform optimally when the input features follow a **Normal (Gaussian) Distribution**. When your data is skewed, a **Power Transformer** is a sophisticated tool used to map that data into a normal distribution by applying power-based mathematical functions.


## 1. What is a Power Transformer?
A **Power Transformer** belongs to the family of parametric, monotonic transformations. Unlike simple function transformations (like log or square root) which use a fixed formula, a Power Transformer searches for an optimal exponent—known as **Lambda ($\lambda$)**—to transform your data into a "Gaussian-like" shape.

### **The Intuition**
Imagine your data distribution is a lump of clay. The Power Transformer acts as a mold that "squashes" or "stretches" that clay until it resembles a symmetric bell curve.

```mermaid
graph LR
    A[Non-Normal Data <br> (Skewed)] --> B{Power Transformer}
    B --> |Finds Optimal Lambda| C[Normal Distribution <br> (Gaussian)]
```

> **Key Takeaway:** While standard function transforms use a fixed "one-size-fits-all" approach, Power Transformers are **adaptive**; they find the best mathematical "fit" specifically for your dataset.


## 2. Box-Cox vs. Yeo-Johnson
There are two primary implementations of Power Transformations available in Scikit-Learn.

### **A. Box-Cox Transformation**
Developed by George Box and David Cox, this is a highly effective transformation that searches for the best $\lambda$ value within a typical range of **-5 to 5**.
*   **Restriction:** It is strictly applicable only to **strictly positive numbers** ($x > 0$). It cannot handle zero or negative values.
*   **Workaround:** If your data contains zeros, you can add a very small constant (e.g., `0.0001`) to the column before applying Box-Cox.

### **B. Yeo-Johnson Transformation**
The Yeo-Johnson transformation is a modern variation of Box-Cox designed to overcome its limitations.
*   **Advantage:** It supports **zero and negative values**, making it more versatile for real-world datasets.
*   **Default Status:** Because of its flexibility, Yeo-Johnson is the default method in Scikit-Learn's `PowerTransformer`.

| Feature | Box-Cox | Yeo-Johnson |
| :--- | :--- | :--- |
| **Supports Positive Data** | Yes | Yes |
| **Supports Zero/Negative Data** | No | Yes |
| **Search Method** | MLE or Bayesian | MLE or Bayesian |
| **Primary Goal** | Minimize Variance/Skew | Minimize Variance/Skew |


## 3. How the Algorithm "Learns" Lambda
You don't have to manually guess which power (square, cube, etc.) will work best. The algorithm uses internal statistical techniques like **Maximum Likelihood Estimation (MLE)** or **Bayesian Statistics** to examine all potential values of $\lambda$ and select the one that results in the best approximation of a normal distribution.

> **Key Takeaway:** You can access these learned exponents after fitting the transformer using the `.lambdas_` attribute.


## 4. Technical Implementation in Scikit-Learn
Scikit-Learn provides the `PowerTransformer` class within the `preprocessing` module.

### **Key Feature: Automatic Scaling**
By default, `PowerTransformer` includes a parameter `standardize=True`. This means that after transforming the data to a normal distribution, it automatically applies **Standardization** (setting mean to 0 and standard deviation to 1). You generally don't need a separate `StandardScaler` when using this tool.

### **Basic Code Workflow**
```python
from sklearn.preprocessing import PowerTransformer

# 1. Initialize (method can be 'box-cox' or 'yeo-johnson')
pt = PowerTransformer(method='yeo-johnson')

# 2. Fit and Transform the training data
X_train_transformed = pt.fit_transform(X_train)

# 3. Transform the test data (using parameters learned from train)
X_test_transformed = pt.transform(X_test)

# 4. View the optimized lambda values for each column
print(pt.lambdas_)
```


## 5. Performance Impact
In the lecture demonstration using a **Concrete Strength dataset**, applying Power Transformations resulted in a significant boost in model performance:
*   **Without Transformation:** $R^2$ Score was approximately **0.62**.
*   **With Power Transformation:** $R^2$ Score improved to approximately **0.80+**.

### **Visual Validation (QQ Plots)**
To confirm the transformation worked, always compare the **Before** and **After** states using a **QQ Plot**. If the transformation was successful, the data points in the "After" plot will align much more closely with the 45-degree diagonal line.


## 6. Summary: When to Use Power Transformers

### **Key Takeaways**
*   **Algorithm Fit:** Use this for **Linear Regression**, **Logistic Regression**, **KNN**, and **SVM**. 
*   **Comparison:** Power Transformers generally outperform simple Log or Square Root transforms because they optimize the exponent for each specific feature.
*   **Normalization:** It is the most robust way to handle features with varying degrees of skewness simultaneously.

### **Common Mistakes**
*   **Using on Tree Models:** Do not use this for Decision Trees or Random Forests; it adds unnecessary complexity without improving accuracy for those models.
*   **Box-Cox on Zeros:** If your data has zeros and you force `method='box-cox'`, the code will crash. Use Yeo-Johnson instead.
*   **Fit-Transform Confusion:** Never `fit` your transformer on the test set; always fit on the training set and use those learned $\lambda$ values to transform the test set.
