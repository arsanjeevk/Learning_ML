Video link: https://www.youtube.com/watch?v=A9SezQlvakw&list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH&index=14

---

# Data Science Project Lifecycle: Framing the Problem

This guide covers the first and most critical step in any data science project: **Problem Framing**. Success in a professional data environment relies not just on coding skills, but on the ability to translate business goals into actionable machine learning tasks.



## 1. Translating Business Problems to ML Tasks

The initial step is moving from a vague business objective to a concrete mathematical target. 

*   **Example (Netflix Churn):** 
    *   **Business Goal:** Increase revenue.
    *   **Strategy:** Reduce churn (the rate at which users leave the platform).
    *   **ML Objective:** Identify individual users at high risk of canceling their subscription to provide targeted interventions (e.g., discounts).

**Key Takeaways:** Always quantify your goal. Instead of "reduce churn," aim for "reduce churn rate from 4% to 3.75% within six months."



## 2. Defining the Type of Problem

Identifying the machine learning paradigm is essential before selecting an algorithm.

| Problem Type | Goal | Example |
| :--- | :--- | :--- |
| **Classification** | Predict discrete labels (Yes/No) | Will the user churn? |
| **Regression** | Predict continuous values (0-100) | Probability score of user churn |

*   **Refinement:** Moving from classification (Churn: Yes/No) to regression (Churn Probability: 0.85) allows for granular decision-making, such as tiered discount offers based on risk levels.

**Key Takeaways:** Choose the output format that best serves the business intervention strategy.



## 3. Data Identification and Engineering

You must determine what features influence the target variable. You will likely need to collaborate with **Data Engineers** to pull this from the company's data warehouse.

*   **Potential Features for Churn:**
    *   `watch_time`: Total duration of content consumed.
    *   `search_history`: Number of failed searches (inability to find content).
    *   `drop_rate`: Number of shows abandoned mid-watch.
    *   `recommendation_clicks`: User engagement with suggestions.

**Key Takeaways:** The quality of your model is bound by the relevance of your features. Engage early with those who manage the data.



## 4. Operational Strategy: Online vs. Batch

Deciding how the model learns and updates is crucial for deployment:

*   **Batch Learning:** The model is trained offline on a static dataset, updated periodically (e.g., weekly). Suitable when data is stable.
*   **Online Learning:** The model continuously updates as new data arrives. Best for highly volatile domains where user behavior changes rapidly.

**Key Takeaways:** Consider the volatility of your data; if consumer habits shift quickly, plan for frequent retraining or online updates.



## 5. Metrics for Success

Define how you will measure performance before you build the model. This acts as your "North Star."

1.  **Business Metric:** Does the actual churn rate decrease in production?
2.  **Model Metric:** Is the model accurately identifying the users who actually left?

**Key Takeaways:** Your success is measured by the impact on business KPIs, not just model accuracy scores.



## 6. Assumptions and Validation

Always list and test your assumptions early to avoid costly rework.

*   **Feature Availability:** Can we actually obtain the features we need from the live database?
*   **Geographic Variation:** Will a model trained on US user data work for international markets?
*   **Data Drift:** Is the historical data still representative of current user behavior?

**Key Takeaways:** Challenging your own assumptions prevents failures during the later, more expensive stages of development.
