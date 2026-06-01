Video Link: https://www.youtube.com/watch?v=NU37mF5q8VE&list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH&index=54


---
# Multiple Linear Regression: Mathematical Formulation

This guide provides a comprehensive step-by-step mathematical derivation for **Multiple Linear Regression**, from the fundamental equation to the **Normal Equation** solution.


## 1. Problem Setup

In **Multiple Linear Regression**, we model the relationship between a dependent variable $y$ and multiple independent variables (features) $x_1, x_2, ..., x_m$.

For a single prediction $\hat{y}$, the linear equation is:
$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_m x_m$$

Where:
*   $\beta_0$ is the **intercept** (bias term).
*   $\beta_1, \beta_2, ..., \beta_m$ are the **coefficients** (weights) for each feature.



## 2. Matrix Representation

To compute predictions for all $n$ samples simultaneously, we represent the data as matrices. Let $X$ be our input matrix of shape $(n, m+1)$. We prepend a column of $1$ s to the features to account for the intercept $\beta_0$.

### The Matrices

<img width="281" height="392" alt="image" src="https://github.com/user-attachments/assets/4ca44898-c3a6-4d31-8a34-945fa1a7fc04" />


### Vectorized Prediction
$$y_{pred} = X\beta$$



## 3. The Loss Function (Ordinary Least Squares)

We define the error vector $e$ as the difference between actual and predicted values: $e = y - X\beta$.

The cost function $J$ is the **sum of squared residuals**:
$$J = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

In matrix notation, this is expressed as:
$$J = e^T e = (y - X\beta)^T (y - X\beta)$$

### Step-by-Step Expansion:
1. Expand the transpose:
   $$J = (y^T - \beta^T X^T)(y - X\beta)$$
2. Distribute terms:
   $$J = y^T y - y^T X \beta - \beta^T X^T y + \beta^T X^T X \beta$$
3. Since $y^T X \beta$ is a scalar, it is equal to its transpose $(\beta^T X^T y)$. Thus:
   $$J = y^T y - 2\beta^T X^T y + \beta^T X^T X \beta$$



## 4. Derivation of the Normal Equation

To minimize $J$, we take the derivative with respect to the vector $\beta$ and set it to $0$:

$$\frac{\partial J}{\partial \beta} = 0 - 2X^T y + 2X^T X \beta = 0$$

1. Simplify the equation:
   $$2X^T X \beta = 2X^T y$$
2. Divide by 2:
   $$X^T X \beta = X^T y$$
3. Multiply by the inverse of $(X^T X)$ to solve for $\beta$:
   $$\beta = (X^T X)^{-1} X^T y$$

This is the **Normal Equation**, which provides the analytical solution for optimal weights.





### Key Takeaways
*   **Closed-Form Solution:** The Normal Equation gives exact coefficients without iteration.
*   **Computational Cost:** Calculating $(X^T X)^{-1}$ has a complexity of $O(m^3)$, making it computationally expensive for datasets with a massive number of features.
*   **Alternative:** **Gradient Descent** is often used instead for large-scale datasets to avoid the direct matrix inversion.ernative:** **Gradient Descent** is often used instead for large-scale datasets to avoid the direct matrix inversion.
> *   **OLS** is preferred for datasets where the number of features is manageable.
> *   **Gradient Descent** is used when the "Curse of Dimensionality" makes matrix inversion computationally impossible.
