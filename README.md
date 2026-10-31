# START: SHAP Explained Like a Blog

If you have ever trained a machine learning model and asked, “Why did it predict this result?” then SHAP is one of the most useful tools you can learn.

SHAP stands for Shapley Additive Explanations. It explains the output of a model by assigning a contribution to each feature. These contributions tell us how much each feature pushed the prediction up or down compared to the average prediction.

In simple words:

- SHAP answers: “Which features mattered most in this prediction?”
- It also answers: “How much did each feature contribute to the final output?”
- It is widely used in credit scoring, healthcare, marketing, and fraud detection because it makes black-box models more interpretable.

---

## Q) What is SHAP? On what basis it works

SHAP is based on a concept from game theory called the Shapley value.

The idea is simple:

- Imagine a team of players working together to win a game.
- Each player contributes a certain amount to the team’s final result.
- The Shapley value is the fair way to divide the total reward among players based on their contribution.

In machine learning, the “players” are the features, and the “game” is the prediction made by the model.

Instead of asking:

- “Did the model use this feature?”

we ask:

- “How much did this feature contribute to this specific prediction?”

The SHAP value of a feature is its average marginal contribution across all possible combinations of other features.

The main formula is:

$$
\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|! (|F|-|S|-1)!}{|F|!} \left[f(S \cup \{i\}) - f(S)\right]
$$

Where:

- $F$ = all features
- $S$ = a subset of features excluding feature $i$
- $f(S)$ = prediction using only the features in subset $S$
- $f(S \cup \{i\})$ = prediction when feature $i$ is added
- The difference tells us the marginal contribution of feature $i$
- The average over all subsets gives the SHAP value

This is why SHAP is mathematically grounded, fair, and consistent.

### Why SHAP is powerful

SHAP gives us several things:

1. Local explanations: Why did this one prediction occur?
2. Global explanations: Which features matter the most overall?
3. Fairness: The contribution of each feature is distributed based on the actual role it plays in the prediction.

---

## Q2) How is SHAP value calculated in the backend for any classification problem?

Let us understand this with a practical example.

### Sample data (3 features, 5 rows)

Suppose we have a classification problem where we want to predict whether a person will default on a loan.

| row_id | age | income | credit_score | default |
|---|---:|---:|---:|---:|
| 1 | 25 | 30 | 550 | 0 |
| 2 | 30 | 45 | 600 | 0 |
| 3 | 35 | 60 | 700 | 1 |
| 4 | 50 | 80 | 720 | 1 |
| 5 | 60 | 90 | 760 | 1 |

Here:

- Features: `age`, `income`, `credit_score`
- Target: `default` (0 = no default, 1 = default)

### The core idea

For a single sample, say a person with:

- age = 35
- income = 60
- credit_score = 700

We want to know the contribution of each feature to the model’s predicted probability of default.

The model does not just say:

- “credit_score contributed the most”

It calculates this by comparing predictions under different feature combinations.

### The backend logic

If the model is $f$, and the feature set is $F = \{age, income, credit\_score\}$, then for each feature we compare:

- prediction with feature included
- prediction without feature
- repeated over many subsets of remaining features

For a feature $i$:

$$
\text{marginal contribution} = f(S \cup \{i\}) - f(S)
$$

Then we average that difference over all possible subsets $S$ of the remaining features.

This is the exact SHAP value definition.

### What does “without considering a feature” really mean?

This is a very common confusion.

A lot of people think:

- “Without a feature means remove it from the model and rebuild the model on the remaining features.”

That is not how SHAP works in the mathematical sense.

SHAP does not mean “throw away the feature and train a new model every time.”

Instead, it means:

- check the prediction when the feature is absent from the coalition of features being considered
- compare that prediction with the one when the feature is present
- average that difference across all possible combinations of the other features

This is different from retraining a model on a reduced subset.

In practice, the “feature absent” state is approximated by using a baseline or background data distribution. For tree-based models, SHAP uses efficient algorithms to compute the exact or near-exact contribution without retraining the model repeatedly.

So the correct interpretation is:

- We are not literally rebuilding the model on only a subset of features.
- We are evaluating the model’s output under a coalition where the feature is missing and comparing it with the coalition where the feature is present.
- Then we average the contribution across all such coalitions.

This is the real meaning of SHAP.

### A simple classification example

Assume the trained model predicts a probability of default for a customer.

Suppose the base prediction (average probability over data) is:

- base value = 0.30

For a specific customer:

- predicted probability = 0.72

Now SHAP tries to explain why the final probability is 0.72 instead of 0.30.

It tests questions like:

- What if age is missing?
- What if income is missing?
- What if credit_score is missing?

For each feature, it computes:

- prediction with the feature included
- prediction with the feature removed from the coalition
- difference between the two
- average over all possible places the feature could appear in other combinations

This average difference becomes the SHAP value.

### Why the final SHAP values sum to the output difference

For one prediction:

$$
\sum \text{SHAP values} = \text{predicted value} - \text{base value}
$$

So if:

- base value = 0.30
- predicted probability = 0.72

then:

$$
\text{SHAP contributions total} = 0.72 - 0.30 = 0.42
$$

This means the features together increased the prediction by 0.42 compared to the average customer.

Example interpretation:

- `credit_score` contributed +0.20
- `income` contributed +0.15
- `age` contributed +0.07

Then:

$$
0.20 + 0.15 + 0.07 = 0.42
$$

This is the same logic used in SHAP for classification models.

---

## Q) How is SHAP value calculated at each prediction level? And what is feature importance?

At the prediction level, SHAP explains one specific sample.

For example, suppose the model predicts a default probability of 0.75 for a customer.

The model might have a baseline probability of 0.35 on average across all customers. That is the base value.

Then each feature pushes the output above or below the base value.

### Example

| Feature | SHAP value | Meaning |
|---|---:|---|
| age | +0.08 | older age increased the default probability |
| income | -0.12 | higher income reduced default risk |
| credit_score | +0.26 | low credit score strongly increased risk |

If the base value is 0.35 and the final prediction is 0.75:

$$
0.35 + 0.08 - 0.12 + 0.26 = 0.57
$$

This is a simple illustration. In a real model, the exact numbers depend on the model and training data.

### What is feature importance?

Feature importance is the average absolute SHAP value of a feature across all samples in the dataset.

The formula is:

$$
\text{Feature Importance}_i = \frac{1}{N} \sum_{n=1}^{N} |\phi_i^{(n)}|
$$

Where:

- $N$ = number of samples
- $\phi_i^{(n)}$ = SHAP value of feature $i$ for sample $n$

This tells us how much on average each feature mattered in the model’s prediction.

A feature with a large average absolute SHAP value is important.

This is the global explanation layer of SHAP.

---

## Q) Explain the SHAP feature importance plots

SHAP has several very popular visualizations.

### 1. Bar plot

This shows the average absolute SHAP value for each feature.

- Long bars = important features
- Short bars = less important features

It is a quick summary of global feature importance.

### 2. Beeswarm plot

This plot shows the distribution of SHAP values for each feature across all data points.

- Each point is one sample
- Color represents the feature value (low to high)
- X-axis shows the SHAP value

Interpretation:

- A feature with positive SHAP values pushes prediction upward
- A feature with negative SHAP values pushes prediction downward
- Wide spread means the feature impacts predictions strongly

### 3. Waterfall plot

This is the best plot for explaining one single prediction.

It starts from the base value and then adds each feature’s contribution until it reaches the final prediction.

This plot tells the story:

- “This prediction started at the average value”
- “Age increased it by 0.08”
- “Income reduced it by 0.12”
- “Credit score pushed it up by 0.26”
- “Final prediction is 0.75”

### 4. Dependence plot

This shows how a feature’s effect changes as its value changes.

For example:

- Low `credit_score` may strongly increase risk
- High `credit_score` may reduce risk

This helps detect nonlinear relationships.

---

## Q) Can SHAP be used for all models? Explain

SHAP is extremely useful, but not every model supports SHAP equally well.

### Yes, SHAP can be used on many models

It works for:

- tree models (XGBoost, Random Forest, Decision Trees)
- linear models
- logistic regression
- SVMs
- neural networks (with approximated or model-specific explainers)
- ensemble methods

### But the computational cost varies

- Linear models: SHAP is easy and fast
- Tree models: SHAP is very efficient and widely used
- Neural networks: SHAP can be computationally expensive
- Kernel SHAP: more general but slower for large datasets

### Important note

SHAP is not magic. It is only as good as the model itself.

If the model is poor, SHAP will explain a poor prediction. It does not fix the underlying model quality.

Also, SHAP explanations are local and model-specific, not universal truths about the real-world causal effect.

This means:

- SHAP tells us what the model is using
- It does not necessarily tell us the true causal relationship in the world

---

## Q) Give the same example for a regression problem statement how SHAP works in regression

Now let us look at a regression problem.

### Regression scenario

Suppose we want to predict house price using features like:

| row_id | area_sqft | bedrooms | neighborhood_score | price |
|---|---:|---:|---:|---:|
| 1 | 1200 | 2 | 5 | 250000 |
| 2 | 1500 | 3 | 6 | 310000 |
| 3 | 1800 | 3 | 7 | 380000 |
| 4 | 2200 | 4 | 8 | 440000 |
| 5 | 2600 | 5 | 9 | 520000 |

Here:

- Features: `area_sqft`, `bedrooms`, `neighborhood_score`
- Target: `price`

### How SHAP works in regression

For a regression problem, the model predicts a numeric value, for example:

- base value = 350000
- final house price prediction = 470000

Then each feature contributes to moving the prediction away from the average base value.

Example:

| Feature | SHAP value | Meaning |
|---|---:|---|
| area_sqft | +75000 | bigger house increases price |
| bedrooms | +20000 | more bedrooms increases price |
| neighborhood_score | +25000 | better neighborhood increases price |

The sum of contributions explains the difference between the final prediction and the expected value:

$$
350000 + 75000 + 20000 + 25000 = 470000
$$

This is the exact same mechanism as classification, except the model output is continuous instead of a probability.

### What is the difference between classification and regression SHAP?

The logic is the same:

- classification: model output is probability or log-odds
- regression: model output is a numeric value

The SHAP value still tells us the contribution of each feature to the model output.

---

## Q) Code snippet to build sample data for classification and regression, train an XGBoost model, and show SHAP feature importance and prediction-level contribution

Below is a complete example in Python using `scikit-learn` and `xgboost`.

```python
import numpy as np
import pandas as pd
import shap
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

# -------------------------------
# 1) Classification example
# -------------------------------
X_cls, y_cls = make_classification(
    n_samples=500,
    n_features=3,
    n_informative=3,
    n_redundant=0,
    n_classes=2,
    random_state=42
)

X_cls = pd.DataFrame(X_cls, columns=['age', 'income', 'credit_score'])
y_cls = pd.Series(y_cls, name='default')

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)

model_cls = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.1,
    objective='binary:logistic',
    random_state=42
)
model_cls.fit(X_train_cls, y_train_cls)

# SHAP explanation for classification
explainer_cls = shap.Explainer(model_cls, X_train_cls)
shap_values_cls = explainer_cls(X_test_cls)

# Global feature importance
shap.plots.bar(shap_values_cls, max_display=3)

# Prediction-level explanation for one sample
sample_index = 0
shap.plots.waterfall(shap_values_cls[sample_index], max_display=3)

# Print the model prediction and SHAP values for one sample
pred_prob = model_cls.predict_proba(X_test_cls.iloc[[sample_index]])[:, 1][0]
print('Predicted probability:', pred_prob)
print('SHAP values:', shap_values_cls.values[sample_index])
print('Base value:', shap_values_cls.base_values[sample_index])

# -------------------------------
# 2) Regression example
# -------------------------------
X_reg, y_reg = make_regression(
    n_samples=500,
    n_features=3,
    n_informative=3,
    noise=10,
    random_state=42
)

X_reg = pd.DataFrame(X_reg, columns=['feature_1', 'feature_2', 'feature_3'])
y_reg = pd.Series(y_reg, name='target')

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

model_reg = XGBRegressor(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.1,
    objective='reg:squarederror',
    random_state=42
)
model_reg.fit(X_train_reg, y_train_reg)

# SHAP explanation for regression
explainer_reg = shap.Explainer(model_reg, X_train_reg)
shap_values_reg = explainer_reg(X_test_reg)

# Global feature importance for regression
shap.plots.bar(shap_values_reg, max_display=3)

# Prediction-level explanation for one sample
sample_index_reg = 0
shap.plots.waterfall(shap_values_reg[sample_index_reg], max_display=3)

print('Regression prediction:', model_reg.predict(X_test_reg.iloc[[sample_index_reg]])[0])
print('Regression SHAP values:', shap_values_reg.values[sample_index_reg])
print('Regression base value:', shap_values_reg.base_values[sample_index_reg])
```

### Explanation of the code

1. We create synthetic classification data with three features.
2. We train an `XGBClassifier`.
3. We use `shap.Explainer` to compute SHAP values.
4. We visualize global importance with `shap.plots.bar`.
5. We inspect one prediction with `shap.plots.waterfall`.
6. We repeat the same idea for a regression problem using `XGBRegressor`.

### What you should observe

- The model will produce a predicted probability in classification and a numeric value in regression.
- SHAP will allocate this prediction across the features.
- Features with larger absolute SHAP values have greater impact on the prediction.
- The total contribution of all SHAP values explains the difference between the prediction and the base value.

---

## Final takeaway

SHAP is one of the most important tools for model explainability because it combines:

- solid mathematical theory
- practical explainability
- global and local insights
- useful visualizations for business and technical stakeholders

It does not just show what features are important; it tells us how each feature contributes to each prediction.

That is why SHAP is widely used in real-world machine learning systems.

If you understand SHAP well, you can explain black-box models in a way that is both technically correct and easy to communicate to non-technical people.

---

## Quick summary

- SHAP = Shapley Additive Explanations
- It is based on cooperative game theory
- It measures feature contribution by comparing predictions with and without each feature
- It averages contributions across all feature coalitions
- Feature importance = mean absolute SHAP value over data
- SHAP works very well for tree-based models like XGBoost
- It can be used for both classification and regression

This is the heart of SHAP: explain the model’s output in a fair and mathematically consistent way.
