Video Link: https://www.youtube.com/watch?v=81ymPYEtFOw&list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH&index=3

---

# Types of Machine Learning

This document provides an overview of the four primary categories of Machine Learning (ML), classified based on the level of **supervision** or guidance the algorithm receives during the training process.



## 1. Supervised Machine Learning
In **Supervised Learning**, the algorithm is trained on a labeled dataset that contains both the input features and the corresponding target output. The model learns to map inputs to outputs.

### Core Types:
*   **Regression**: Predicts continuous numerical values (e.g., house prices, salary packages).
*   **Classification**: Predicts categorical labels (e.g., "Yes/No" for student placement, "Spam/Not Spam" for emails).

### How it works:
1. Provide input data (e.g., student IQ and CGPA).
2. Provide target labels (e.g., Placement status: Yes/No).
3. The algorithm discovers the mathematical relationship to predict outcomes for future, unseen data.

> **Key Takeaways:** If you have labeled data (input + output), use Supervised Learning. Choose **Regression** for numbers and **Classification** for categories.


## 2. Unsupervised Machine Learning
In **Unsupervised Learning**, the algorithm is provided with input data *without* any corresponding output labels. The goal is to find hidden patterns or structures within the data.

### Primary Applications:
*   **Clustering**: Grouping similar data points together (e.g., customer segmentation).
*   **Dimensionality Reduction**: Simplifying data by reducing the number of variables while retaining essential information (e.g., Principal Component Analysis).
*   **Anomaly Detection**: Identifying data points that deviate significantly from the norm (e.g., fraud detection).
*   **Association Rule Learning**: Discovering relationships between variables (e.g., "People who buy diapers often buy beer").

> **Key Takeaways:** Use Unsupervised Learning when you have raw, unlabeled data and want to uncover insights, group data, or simplify complex features.



## 3. Semi-Supervised Learning
**Semi-Supervised Learning** is a hybrid approach. It uses a small amount of labeled data combined with a large volume of unlabeled data. This is particularly useful because labeling data is often expensive and time-consuming.

*   **Intuition:** Label a few key data points, and let the model "propagate" these labels to similar, unlabeled clusters.
*   **Example:** Google Photos categorizing faces. You label one photo as "Mom," and the system automatically identifies her in other photos.

> **Key Takeaways:** Best when you have massive datasets but limited resources to label every single entry.


## 4. Reinforcement Learning (RL)
**Reinforcement Learning** involves an **Agent** interacting with an **Environment**. There is no static dataset. Instead, the agent learns through a trial-and-error process by receiving **rewards** (for correct actions) or **punishments** (for mistakes).

### The Learning Loop:
mermaid
graph LR
    A[Agent] -- Action --> B(Environment)
    B -- Reward/Feedback --> A

*   **Policy**: The agent's strategy/rulebook for decision-making.
*   **Goal**: Maximize cumulative rewards over time.
*   **Famous Example**: *AlphaGo* (DeepMind), which mastered the game of Go by playing millions of games against itself.

> **Key Takeaways:** RL is ideal for scenarios like game playing, robotics, and autonomous driving where the model learns through experience rather than static examples.



## Summary Comparison

| Type | Data Requirement | Goal | Example |
| :--- | :--- | :--- | :--- |
| **Supervised** | Labeled Input/Output | Predict outputs | Price Prediction |
| **Unsupervised** | Only Input | Find hidden patterns | Clustering Customers |
| **Semi-Supervised**| Mix of both | Learn with fewer labels | Image Tagging |
| **Reinforcement** | No data (Trial/Error) | Learn optimal strategy | Game AI / Robotics |
