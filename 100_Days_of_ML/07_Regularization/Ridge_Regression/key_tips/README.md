# Ridge Regression: Key Insights, Behavior, and Mathematical Intuition

**Ridge Regression**, also known as **L2 Regularization**, is a fundamental machine learning technique designed to address **overfitting** and **multicollinearity** in linear models. By adding a quadratic penalty term to the standard loss function, Ridge Regression restricts the magnitude of the model's coefficients, leading to more stable, generalizable predictions.

This guide covers the five essential takeaways of Ridge Regression, exploring their mathematical foundations, geometric properties, and practical applications.

---

## 1. Behavior of Coefficients Under Regularization ($\lambda \to \infty$)

### **Intuition**

Regularization introduces a penalty on the size of the model's weights. As the penalty strength ($\lambda$) increases, the model is forced to shrink its coefficients to minimize the total loss.

### **Mathematical Explanation**

The regularized loss function in Ridge Regression is defined as:

$$
L = \text{Original Loss} + \lambda \sum_{j=1}^{p} w_j^2
$$

Where:

- $\text{Original Loss}$ is the standard Sum of Squared Errors (SSE) or Mean Squared Error (MSE).
- $\lambda$ (Lambda/Alpha) is the regularization hyperparameter ($\lambda \ge 0$).
- $w_j$ represents the coefficient (weight) of each input feature.
- $\lambda \sum w_j^2$ is the **L2 shrinkage penalty**.

Let's analyze the limits of $\lambda$:

1. **$\lambda = 0$ (No Regularization):** The penalty term is nullified, and the model behaves exactly like standard **Ordinary Least Squares (OLS) Linear Regression**.

2. **$\lambda \to \infty$ (Maximum Regularization):** The penalty term dominates the loss function. To prevent the loss from exploding, the coefficients $w_j$ are forced to shrink toward zero.

### **The "Never Zero" Rule**

A critical property of Ridge Regression is that while coefficients shrink and get extremely close to zero, they **never become exactly zero**. This means Ridge Regression shrinks the weights but does not perform feature selection (unlike Lasso Regression).

```mermaid
graph LR
    A[Lambda = 0] -->|Standard OLS| B[Large Coefficients]
    C[Lambda increases] -->|Shrinkage| D[Coefficients approach but never reach 0]
