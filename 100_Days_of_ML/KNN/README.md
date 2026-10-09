# K-Nearest Neighbors (KNN) Algorithm

KNN is a simple, intuitive, and **instance-based learning** algorithm used primarily for classification and regression tasks. It operates on the principle that "you are the average of the people you spend the most time with."

---

## 1. Intuition

KNN functions based on **proximity**. To classify a new data point, the algorithm looks at the 'K' closest labeled data points in the feature space and assigns the new point to the majority class among those neighbors.

### How it works:
1. **Select K:** Choose the number of neighbors to consider.
2. **Distance Calculation:** Measure the distance between the query point and all training points (usually **Euclidean distance**).
3. **Sorting:** Sort the distances to find the 'K' nearest points.
4. **Majority Vote:** Assign the class that appears most frequently among the neighbors.

### Mathematical Formulation
The **Euclidean Distance** between two points $P = (p_1, p_2, ..., p_n)$ and $Q = (q_1, q_2, ..., q_n)$ is calculated as:
$$d(P, Q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$

**Key Takeaways:**
* KNN is a "lazy learner"; it does not learn a model during training but stores all training data.
* Classification is based on democratic voting (majority count).

---

## 2. Practical Implementation

When using KNN (e.g., via `scikit-learn`), **feature scaling** is mandatory. Because KNN relies on distance, features with larger scales will dominate the calculation, leading to biased results.

### Workflow:
* **Pre-processing:** Use `StandardScaler` to bring all features to a similar range (mean = 0, standard deviation = 1).
* **Fitting:** The algorithm "fits" the training data by storing it.
* **Predicting:** The algorithm iterates through stored data to compute distances for new inputs.

**Key Takeaways:**
* Always scale data using `StandardScaler` or similar methods.
* KNN is sensitive to the scale of input features.

---

## 3. Selecting the Optimal K

Choosing the right 'K' is critical to model performance:
* **Small K (e.g., 1):** Leads to **overfitting** (high variance). The model becomes sensitive to noise and individual outliers.
* **Large K:** Leads to **underfitting** (high bias). The model becomes too simple and ignores local patterns.

### Methods to find K:
* **Heuristic:** A common rule of thumb is $K = \sqrt{N}$, where $N$ is the number of training samples.
* **Cross-Validation:** Iterate through a range of K values and plot the accuracy score to find the peak performance.

**Key Takeaways:**
* Small K values capture noise; large K values simplify the boundary too much.
* Use cross-validation to find the value of K that provides the best balance.

---

## 4. Decision Surfaces

A **Decision Surface** (or boundary) is the region in the feature space where the model predicts a specific class. 
* With a small K, the boundary is jagged and complex (capturing every training point).
* As K increases, the boundary becomes smoother and more generalized.

---

## 5. Limitations of KNN

While elegant, KNN has significant drawbacks in specific scenarios:

| Limitation | Description |
| :--- | :--- |
| **Computational Cost** | High latency; distance must be calculated for every training point at prediction time. |
| **Curse of Dimensionality** | In high dimensions, the concept of "distance" becomes less meaningful and unreliable. |
| **Sensitivity to Outliers** | A single outlier can significantly shift the decision boundary. |
| **Feature Scaling** | Requires mandatory scaling to prevent dominance by large-magnitude features. |
| **Imbalanced Data** | The majority class tends to dominate predictions. |
| **Inference/Interpretability**| It is a "black box"; it does not explain which features contributed most to the prediction. |

**Key Takeaways:**
* Avoid KNN for massive datasets or very high-dimensional feature spaces.
* KNN is better suited for prediction tasks rather than inferring feature importance.
