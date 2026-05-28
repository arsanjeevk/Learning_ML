PCA code Kaggle notebook : https://www.kaggle.com/nitsin/pca-demo-1
Video Link:https://youtu.be/tXXnxjj2wM4

Video Link: https://www.youtube.com/watch?v=iRbsBi5W0-c&list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH&index=47

---


# Principal Component Analysis (PCA): Geometric Intuition

**Principal Component Analysis (PCA)** is one of the most famous and reliable **unsupervised machine learning** techniques used for **feature extraction** and **dimensionality reduction**. Its primary goal is to transform high-dimensional data into a lower-dimensional space while preserving as much of the data's "essence" or behavior as possible.


## 1. The Core Intuition: The Photographer Analogy

Imagine a photographer taking a picture of a 3D soccer match to be printed in a 2D newspaper. 
*   The **3D environment** (stadium) is high-dimensional data.
*   The **2D photograph** is the reduced-dimensional data.
*   The **photographer's goal** is to find the best angle so that the players and the crowd are clearly visible and not overlapping in a confusing way. 

PCA acts like that photographer, finding the best "angle" (coordinate system) to project data so that the relationships between points are maintained even after dimensions are reduced.


## 2. Why Use PCA?

Using PCA in a machine learning workflow offers two primary benefits:

1.  **Faster Execution:** By reducing the number of features (e.g., from 10 columns to 3), the dataset size shrinks. This leads to faster algorithm training and lower computational costs while maintaining similar performance.
2.  **Visualization:** Humans can only visualize data in up to 3D. PCA allows us to reduce complex data with hundreds of columns (like image pixels) into 2D or 3D plots to see clusters and trends.

> [!TIP]
> **Key Takeaways**
> *   PCA is a **Feature Extraction** technique, not Feature Selection.
> *   It is used for **Dimensionality Reduction** to speed up models and enable **Visualization**.
> *   It is an **Unsupervised** technique because it doesn't require output labels.

## 3. Feature Selection vs. Feature Extraction

To understand PCA, you must distinguish it from **Feature Selection**.

| Feature Selection | Feature Extraction (PCA) |
| :--- | :--- |
| You **keep** the most important original columns and **drop** the rest. | You **transform** existing features into a completely **new set of features**. |
| Relies on keeping original data (e.g., "Number of Rooms"). | Creates new features (e.g., "Size of Flat") that represent a combination of original data. |
| Fails when multiple features are equally important or highly correlated. | Succeeds by combining correlated features into a single dimension. |

### **The Mathematical Logic of Selection**
In Feature Selection, we pick the column where the data has the most **spread** or **variance**. If you plot two features and see one has a wider range of values than the other, you pick the one with the higher variance.

## 4. Geometric Intuition: Principal Components

When simple feature selection isn't enough—because multiple features are equally important—PCA rotates the entire coordinate system.

### **The Rotation Process**
1.  **Original Axes:** Imagine two features, "Rooms" and "Washrooms," which are highly correlated.
2.  **Rotation:** PCA finds a new axis (a line) that goes through the maximum spread of the data.
3.  **Principal Component 1 (PC1):** This new axis becomes the first and most important feature. It captures the most variance.
4.  **Principal Component 2 (PC2):** This axis is perpendicular to PC1 and captures the remaining variance.

```
Original Data (X, Y)
        |
        v
Find direction of maximum variance
        |
        v
Rotate axes to align with that direction
        |
        v
PC1 (Max Variance)  +  PC2 (Remaining Variance)
```

> [!NOTE]
> The number of **Principal Components** generated will always be less than or equal to the number of original features. You then choose the top $n$ components that capture the most information.


## 5. The Importance of Variance

In PCA, **Variance** is the "Gold Standard" for preserving information.

### **Why Variance?**
Variance represents the **spread** of data. If you project data from 2D to 1D, you want to pick an axis where the data points remain as far apart as possible.
*   If variance is **maximized**, the distance between points is preserved, allowing models like **KNN** to still distinguish between different classes.
*   If variance is **low**, data points collapse onto each other, and the model loses its ability to see the difference between them.

### **Mathematical Preference**
PCA uses **Variance** (squared distances) rather than **Mean Absolute Deviation** because the square function is **differentiable at zero**. This makes it possible to solve the complex optimization problems required to find the principal components.

> [!TIP]
> **Key Takeaways**
> *   PCA aims to **maximize variance** in lower dimensions.
> *   High variance preserves the **relationships** and **distances** between data points.
> *   The new axes created are called **Principal Components**.


## 6. Summary Workflow

1.  **Analyze** the high-dimensional data.
2.  **Identify** the directions (axes) that contain the maximum variance.
3.  **Rotate** the coordinate system to these new axes (Principal Components).
4.  **Project** the original data points onto these new axes.
5.  **Select** the top components that capture the majority of the data's essence while discarding noise.
