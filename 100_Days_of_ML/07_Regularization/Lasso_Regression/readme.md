# Lasso Regression & L1 Regularization: Principles and Feature Selection

This repository contains comprehensive, textbook-style reference notes based on the lecture **"Lasso Regression | Intuition and Code Sample | Regularized Linear Models"**. These notes focus on the intuition, mathematical formulation, practical implications, and code implementation of Lasso Regression, emphasizing how it differs from Ridge Regression.

---

## Table of Contents
1. [Introduction to Regularization](#1-introduction-to-regularization)
2. [Lasso Regression: Intuition & Mathematical Formulation](#2-lasso-regression-intuition--mathematical-formulation)
3. [The Regularization Parameter (\$\lambda\$ / \$\alpha\$)](#3-the-regularization-parameter-lambda--alpha)
4. [Feature Selection & Sparsity (Lasso vs. Ridge)](#4-feature-selection--sparsity-lasso-vs-ridge)
5. [The Bias-Variance Trade-Off](#5-the-bias-variance-trade-off)
6. [Geometric Intuition: Impact on the Loss Function](#6-geometric-intuition-impact-on-the-loss-function)
7. [Practical Implementation in Python](#7-practical-implementation-in-python)

---

## 1. Introduction to Regularization

In machine learning, **overfitting** occurs when a model learns the noise in the training dataset too well, resulting in high variance and poor generalization on unseen data. **Regularization** is a technique used to prevent overfitting by adding a penalty term to the model's loss function, discouraging the model from learning overly complex patterns.

The two primary regularization techniques for linear models are:
*   **Ridge Regression (L2 Regularization)**: Penalizes the model using the sum of squared weights.
*   **Lasso Regression (L1 Regularization)**: Penalizes the model using the sum of the absolute values of the weights.


---

# Why Lasso Regression Creates Sparsity: Mathematical Derivation from Scratch

This reference document provides a complete, textbook-style mathematical derivation explaining why **Lasso (L1) Regression** forces model coefficients to exactly zero (creating **sparsity**), whereas **Ridge (L2) Regression** only shrinks them asymptotically. 

This is one of the most frequent machine learning interview questions because it connects optimization theory with the practical benefit of automatic feature selection.

---

## 1. The Core Intuition

In high-dimensional machine learning models, we often have redundant or irrelevant features. 

* **Sparsity** refers to a state where many of the model's feature coefficients (weights) are exactly **zero**.
* **Ridge Regression (L2)** shrinks weights close to zero, meaning you must still keep every single feature in your model to make predictions.
* **Lasso Regression (L1)** drives weights to exactly **zero**, acting as an automated **feature selection** mechanism.

### The Core Difference: Numerator vs. Denominator
In the one-feature centered derivation below, the difference can be seen from where the regularization parameter ($\lambda$) appears after optimization:
* **Lasso (L1)** produces a soft-thresholding form in this one-feature case: $\lambda$ is subtracted from the magnitude of the covariance term. When $\lambda$ is large enough, the coefficient becomes exactly **zero**.
* **Ridge (L2)** adds $\lambda$ to the denominator in this one-feature case. As $\lambda$ grows, the weight shrinks continuously toward zero.

---

## 2. Simple Linear Regression Setup (From Scratch)

To make the mathematics intuitive and explicit, we analyze a **Simple Linear Regression** model with a single input variable $x$ and a single target output $y$. 

The prediction equation is:
$$\hat{y}_i = m x_i + b$$

Where:
* $m$ is the slope coefficient (weight).
* $b$ is the intercept (bias).
* $x_i$ is the input feature for the $i$-th observation.
* $y_i$ is the actual target value for the $i$-th observation.

### Ordinary Least Squares (OLS) Baseline
Without regularization, the standard Ordinary Least Squares (OLS) objective minimizes the Mean Squared Error (MSE):
$$\text{MSE} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

By setting the partial derivative of MSE with respect to $b$ to zero, we find the optimal intercept:
$$b = \bar{y} - m\bar{x}$$

Where:
* $\bar{y} = \frac{1}{n} \sum y_i$ (the mean of $y$).
* $\bar{x} = \frac{1}{n} \sum x_i$ (the mean of $x$).

Substituting $b$ back into our prediction equation:
$$\hat{y}_i = m x_i + (\bar{y} - m\bar{x}) = \bar{y} + m(x_i - \bar{x})$$
$$y_i - \hat{y}_i = (y_i - \bar{y}) - m(x_i - \bar{x})$$

Let us define the following covariance and variance notation to simplify our terms:
* **Sum of Products (Covariance-like term)**: 
  $$S_{xy} = \sum_{i=1}^{n} (y_i - \bar{y})(x_i - \bar{x})$$
* **Sum of Squares (Variance-like term)**: 
  $$S_{xx} = \sum_{i=1}^{n} (x_i - \bar{x})^2$$

Under normal OLS, setting the derivative with respect to $m$ to zero yields the standard slope formula:
$$m_{\text{OLS}} = \frac{S_{xy}}{S_{xx}}$$

---

## 3. Mathematical Derivation of Lasso (L1) Regularization

In Lasso Regression, we add an L1 penalty term to the loss function. The L1 penalty is the sum of the absolute values of the weights:
$$\text{Loss}_{\text{Lasso}} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + 2\lambda |m|$$

*(Note: We use $2\lambda$ instead of $\lambda$ for mathematical convenience to cleanly cancel out the factor of 2 that arises during differentiation. This does not change the optimization behavior.)*

Substituting the expanded error term $(y_i - \hat{y}_i)$ into the objective function:
$$L(m) = \sum_{i=1}^{n} \left[ (y_i - \bar{y}) - m(x_i - \bar{x}) \right]^2 + 2\lambda |m|$$

Because the absolute value function $|m|$ has a sharp corner and is non-differentiable at $m = 0$, we must analyze the optimization in two separate cases based on the sign of $m$.

---

### Case 1: Assuming a Positive Slope ($m > 0$)
When $m > 0$, the absolute value $|m|$ is simply $m$. The loss function is:
$$L(m) = \sum_{i=1}^{n} \left[ (y_i - \bar{y}) - m(x_i - \bar{x}) \right]^2 + 2\lambda m$$

Differentiating with respect to $m$:
$$\frac{\partial L}{\partial m} = \sum_{i=1}^{n} 2\left[ (y_i - \bar{y}) - m(x_i - \bar{x}) \right] \cdot [-(x_i - \bar{x})] + 2\lambda$$
$$\frac{\partial L}{\partial m} = -2\sum_{i=1}^{n} (y_i - \bar{y})(x_i - \bar{x}) + 2m\sum_{i=1}^{n} (x_i - \bar{x})^2 + 2\lambda$$

Substituting our simplified notation $S_{xy}$ and $S_{xx}$:
$$\frac{\partial L}{\partial m} = -2 S_{xy} + 2m S_{xx} + 2\lambda$$

Setting this derivative to zero to find the optimal slope:
$$-2 S_{xy} + 2m S_{xx} + 2\lambda = 0$$
$$-S_{xy} + m S_{xx} + \lambda = 0$$
$$m S_{xx} = S_{xy} - \lambda$$
$$m = \frac{S_{xy} - \lambda}{S_{xx}}$$

#### The Boundary Constraint for Case 1
Since we started with the assumption that **$m > 0$**, this derived formula is only valid if:
$$\frac{S_{xy} - \lambda}{S_{xx}} > 0$$

Since the variance term $S_{xx} = \sum (x_i - \bar{x})^2$ is always strictly positive, this condition simplifies to:
$$S_{xy} - \lambda > 0 \implies \lambda < S_{xy}$$

*(This also implies that $S_{xy}$ must be positive for this case to exist.)*

---

### Case 2: Assuming a Negative Slope ($m < 0$)
When $m < 0$, the absolute value $|m|$ is $-m$. The loss function is:
$$L(m) = \sum_{i=1}^{n} \left[ (y_i - \bar{y}) - m(x_i - \bar{x}) \right]^2 - 2\lambda m$$

Differentiating with respect to $m$:
$$\frac{\partial L}{\partial m} = -2\sum_{i=1}^{n} (y_i - \bar{y})(x_i - \bar{x}) + 2m\sum_{i=1}^{n} (x_i - \bar{x})^2 - 2\lambda$$
$$\frac{\partial L}{\partial m} = -2 S_{xy} + 2m S_{xx} - 2\lambda$$

Setting the derivative to zero:
$$-2 S_{xy} + 2m S_{xx} - 2\lambda = 0$$
$$-S_{xy} + m S_{xx} - \lambda = 0$$
$$m S_{xx} = S_{xy} + \lambda$$
$$m = \frac{S_{xy} + \lambda}{S_{xx}}$$

#### The Boundary Constraint for Case 2
Since we assumed **$m < 0$**, this formula is only valid if:
$$\frac{S_{xy} + \lambda}{S_{xx}} < 0$$

Since $S_{xx} > 0$, this condition simplifies to:
$$S_{xy} + \lambda < 0 \implies \lambda < -S_{xy}$$

*(This implies that $S_{xy}$ must be negative for this case to exist.)*

---

### The Mathematical Dead Zone: When $\lambda \ge |S_{xy}|$

Let us analyze what happens when the regularization strength $\lambda$ is increased beyond the covariance magnitude $|S_{xy}|$.

#### Scenario A: Positive Covariance ($S_{xy} > 0$)
Suppose $S_{xy} = 100$ and $S_{xx} = 50$.
* **With $\lambda = 0$ (No regularization)**:
  $$m = \frac{100 - 0}{50} = 2$$
* **With $\lambda = 50$**:
  $$m = \frac{100 - 50}{50} = 1$$
* **With $\lambda = 100$**:
  $$m = \frac{100 - 100}{50} = 0$$
* **With $\lambda = 150$**:
  If we plug this into our positive-slope equation, we calculate:
  $$m = \frac{100 - 150}{50} = -1$$
  **Contradiction!** We assumed $m > 0$, but the math outputted a negative value ($-1$). This violates the assumption of Case 1, making the formula invalid.
  
  What if we try to use Case 2 ($m < 0$)?
  $$m = \frac{100 + 150}{50} = 5$$
  **Contradiction!** We assumed $m < 0$, but the math outputted a positive value ($5$). This violates Case 2, making the formula invalid.

#### Scenario B: Negative Covariance ($S_{xy} < 0$)
Suppose $S_{xy} = -100$ and $S_{xx} = 50$.
* **With $\lambda = 0$**:
  $$m = \frac{-100 + 0}{50} = -2$$
* **With $\lambda = 50$**:
  $$m = \frac{-100 + 50}{50} = -1$$
* **With $\lambda = 100$**:
  $$m = \frac{-100 + 100}{50} = 0$$
* **With $\lambda = 150$**:
  If we plug this into our negative-slope equation, we calculate:
  $$m = \frac{-100 + 150}{50} = 1$$
  **Contradiction!** We assumed $m < 0$, but the output is positive ($1$).
  
  If we try the positive-slope equation:
  $$m = \frac{-100 - 150}{50} = -5$$
  **Contradiction!** We assumed $m > 0$, but the output is negative ($-5$).

#### The Optimization Resolution
Because both mathematical paths break down once $\lambda \ge |S_{xy}|$, the optimization algorithm is physically blocked from crossing zero to the other side. The loss function's minimum point gets trapped at the non-differentiable sharp corner of the absolute value function. 

Thus, the optimizer has no choice but to stop and pin the coefficient at exactly:
$$m = 0$$

This is how Lasso Regression creates **sparsity**.

### Key Takeaways
* Due to the absolute value function $|m|$, the Lasso penalty adds a constant slope ($\pm 2\lambda$) to the loss function.
* This places $\lambda$ directly into the **numerator** of the coefficient's optimal solution ($S_{xy} \pm \lambda$).
* If $\lambda \ge |S_{xy}|$, assuming either a positive or negative slope leads to a mathematical sign contradiction.
* The optimization solver resolves this contradiction by locking the coefficient at exactly **zero**.

---

## 4. Mathematical Comparison with Ridge (L2) Regularization

To contrast this with **Ridge Regression**, let's look at its loss function with an L2 penalty:
$$\text{Loss}_{\text{Ridge}} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda m^2$$

Substituting the expanded target error:
$$L(m) = \sum_{i=1}^{n} \left[ (y_i - \bar{y}) - m(x_i - \bar{x}) \right]^2 + \lambda m^2$$

Differentiating with respect to $m$:
$$\frac{\partial L}{\partial m} = -2\sum_{i=1}^{n} (y_i - \bar{y})(x_i - \bar{x}) + 2m\sum_{i=1}^{n} (x_i - \bar{x})^2 + 2\lambda m$$
$$\frac{\partial L}{\partial m} = -2 S_{xy} + 2m S_{xx} + 2\lambda m$$

Setting the derivative to zero:
$$-2 S_{xy} + 2m S_{xx} + 2\lambda m = 0$$
$$-S_{xy} + m(S_{xx} + \lambda) = 0$$
$$m(S_{xx} + \lambda) = S_{xy}$$
$$m = \frac{S_{xy}}{S_{xx} + \lambda}$$

### Why Ridge Coefficients Never Hit Zero
In Ridge Regression's optimal solution, the regularization parameter $\lambda$ sits strictly in the **denominator**. 

$$\lim_{\lambda \to \infty} \left( \frac{S_{xy}}{S_{xx} + \lambda} \right) = 0$$

As $\lambda$ increases:
* The denominator grows larger and larger.
* The slope $m$ gets closer and closer to zero.
* For any finite value of $\lambda$, the fraction is non-zero (assuming $S_{xy} \neq 0$). 
* Thus, Ridge can shrink coefficients to tiny decimals (e.g., $0.00001$), but it **never** makes them exactly zero.
