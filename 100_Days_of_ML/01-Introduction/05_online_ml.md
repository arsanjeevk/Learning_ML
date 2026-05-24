Video Link: https://www.youtube.com/watch?v=3oOipgCbLIk&list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH&index=5

---

# Online Machine Learning

Online Machine Learning refers to a technique where a machine learning model is updated **incrementally** as new data arrives, rather than being trained once on a static, large dataset (Batch Learning). 

## 1. Batch vs. Online Learning

| Feature | Batch Learning | Online Learning |
| :--- | :--- | :--- |
| **Data Input** | Entire dataset at once | Data points/small batches sequentially |
| **Training** | Offline (one-time) | On-the-fly (incremental) |
| **Resource Usage** | High memory/CPU at training | Low, consistent resources |
| **Adaptability** | Static (requires re-training) | Dynamic (adapts to new patterns) |

## 2. How Online ML Works

In an online setting, the model exists on the server. As new data streams in, the model processes the input for a prediction and simultaneously updates its internal parameters using that specific data point. This is often referred to as **Incremental Learning**.

### Process Flow
mermaid
graph LR
    A[New Data Stream] --> B{Model Server}
    B --> C[Prediction]
    B --> D[Incremental Training]
    D --> B


### Key Takeaways
* **Efficiency:** Training occurs on small, manageable chunks of data.
* **Real-time Adaptation:** Models evolve alongside changing data trends.
* **Continuous Learning:** The model is always "live" and learning from the latest feedback.

## 3. When to Use Online ML

* **Concept Drift:** When the underlying patterns in your data change over time (e.g., stock price prediction, e-commerce trends).
* **Large-Scale Data:** When the dataset is too large to fit into memory (requires **Out-of-Core** learning).
* **Low-Latency Requirements:** When you need a system that adapts instantly without the overhead of full re-training cycles.

## 4. Implementation Strategies

### Scikit-Learn
Most standard models use the `.fit()` method, but for online learning, look for algorithms that support the `.partial_fit()` method.

* **Example (SGDRegressor):**
  python
  from sklearn.linear_model import SGDRegressor
  model = SGDRegressor()
  # Train on a small batch
  model.partial_fit(X_batch, y_batch)
  

### Specialized Libraries
* **[River](https://riverml.xyz/):** A dedicated library specifically built for online machine learning and streaming data.
* **[Vowpal Wabbit](https://vowpalwabbit.org/):** A high-performance library used for large-scale online learning and reinforcement learning tasks.

### Key Takeaways
* **Learing Rate:** Crucial in online ML; it determines how much the model adjusts to new data versus retaining old information.
* **Out-of-Core Learning:** Allows training on datasets larger than RAM by processing them in sequential streams.

## 5. Challenges and Risks

* **Catastrophic Interference:** The model might "forget" older patterns as it learns new ones.
* **Data Poisoning:** Because the model learns from live data, malicious or noisy data can bias the model, requiring robust **monitoring** and **anomaly detection**.
* **Complexity:** Setting up an infrastructure that handles continuous learning and deployment is more complex than standard batch systems.

### Key Takeaways
* **Monitoring is Mandatory:** Always implement mechanisms to detect anomalies in incoming data.
* **Version Control:** Maintain a rollback strategy in case the model begins to perform poorly due to bad streaming data.
