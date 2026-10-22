# 📨 Message Intelligence System
### Spam Message Classification using KNN, SVM & Naive Bayes

> A machine learning project that classifies messages as **Spam (1)** or **Not Spam (0)** using three classic classifiers — **K-Nearest Neighbors**, **Support Vector Machine**, and **Gaussian Naive Bayes** — and compares them with Accuracy, Precision, Recall and F1 Score. It is backed by a theory section (Part A) covering conditional probability, Bayes' theorem and the working principle of each algorithm.

---

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Project Files](#-project-files)
3. [Tools & Technologies Used](#-tools--technologies-used)
4. [Dataset & Data Types](#-dataset--data-types)
5. [Workflow](#-workflow)
6. [Code Explanation & Conclusion (Question by Question)](#-code-explanation--conclusion-question-by-question)
7. [Final Model Comparison](#-final-model-comparison)
8. [Overall Conclusion](#-overall-conclusion)
9. [Part A – Theory File](#-part-a--theory-file)
10. [Explanation Video](#-explanation-video)
11. [How to Run](#-how-to-run)

---

## 🎯 Project Overview

| Item | Details |
|---|---|
| **Project Title** | Message Intelligence System |
| **Problem Type** | Binary Classification (Spam vs Not Spam) |
| **Target Variable** | `spam_label` (0 = Not Spam, 1 = Spam) |
| **Total Records** | 5,200 messages |
| **Total Columns** | 16 (15 input features + 1 target) |
| **Algorithms** | KNN, SVM (Linear / RBF / Polynomial), Gaussian Naive Bayes |
| **Metrics** | Accuracy, Precision, Recall, F1 Score, Confusion Matrix |

---

## 📂 Project Files

```
Message-Intelligence-System/
│
├── main.ipynb                                   # Complete code (Parts B–F)
├── Message_Intelligence_Dataset_5200.csv        # Dataset (5,200 rows × 16 columns)
├── Part_A_Message_Intelligence_System.pdf       # Part A – Theory (Probability & Concepts)
└── README.md                                    # Project documentation
```

---

## 🛠 Tools & Technologies Used

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="70" alt="Python"/>&nbsp;&nbsp;&nbsp;
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/jupyter/jupyter-original-wordmark.svg" width="70" alt="Jupyter Notebook"/>&nbsp;&nbsp;&nbsp;
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original.svg" width="70" alt="Pandas"/>&nbsp;&nbsp;&nbsp;
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" width="70" alt="NumPy"/>&nbsp;&nbsp;&nbsp;
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/scikitlearn/scikitlearn-original.svg" width="70" alt="Scikit-learn"/>&nbsp;&nbsp;&nbsp;
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/matplotlib/matplotlib-original.svg" width="70" alt="Matplotlib"/>&nbsp;&nbsp;&nbsp;
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" width="70" alt="GitHub"/>
</p>

| Tool | Purpose in this project |
|---|---|
| **Python** | Main programming language |
| **Jupyter Notebook** | Writing and running code (`main.ipynb`) |
| **Pandas** | Loading the CSV, checking missing values, building result tables |
| **NumPy** | Numerical operations |
| **Scikit-learn** | Pipeline, imputer, scaler, KNN, SVM, Naive Bayes, metrics |
| **Matplotlib** | Plots – K-value graph, accuracy bar charts, model comparison |
| **GitHub** | Hosting the project |

---

## 🗂 Dataset & Data Types

**File:** `Message_Intelligence_Dataset_5200.csv` — **5,200 rows × 16 columns**

| # | Column | Data Type | Category | Description |
|---|---|---|---|---|
| 1 | `message_id` | int64 | Identifier | Unique ID of the message |
| 2 | `message_text` | string (object) | Text | Actual message content |
| 3 | `message_length` | int64 | Numerical | Number of characters |
| 4 | `word_count` | int64 | Numerical | Number of words |
| 5 | `num_urls` | int64 | Numerical | Number of URLs in the message |
| 6 | `num_digits` | int64 | Numerical | Number of digits |
| 7 | `num_special_chars` | int64 | Numerical | Number of special characters |
| 8 | `spam_keyword_score` | int64 | Numerical | Count of spam-type keywords |
| 9 | `legit_keyword_score` | int64 | Numerical | Count of legitimate-type keywords |
| 10 | `sender_activity_score` | float64 | Numerical | Activity score of the sender *(106 missing)* |
| 11 | `sender_account_age_days` | float64 | Numerical | Age of sender account in days *(113 missing)* |
| 12 | `messages_sent_last_24h` | float64 | Numerical | Messages sent by the sender in last 24 h *(162 missing)* |
| 13 | `timestamp` | string (object) | Date/Time | Date and time of the message |
| 14 | `hour_of_day` | int64 | Numerical | Hour (0–23) |
| 15 | `day_of_week` | int64 | Numerical / Categorical | Day (0–6) |
| 16 | `spam_label` | int64 | **Target** | 0 = Not Spam, 1 = Spam |

**Summary of data types:** 11 × `int64` (including the target), 3 × `float64`, 2 × text/string (`message_text`, `timestamp`).

**Class distribution (target):**

| Class | Meaning | Count | Share |
|---|---|---|---|
| 0 | Not Spam | 4,227 | 81.3 % |
| 1 | Spam | 973 | 18.7 % |

➡️ The dataset is **imbalanced** (about 4 : 1), which is why `stratify=y` is used while splitting and `class_weight="balanced"` is used in SVM.

---

## 🔄 Workflow

```
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────────┐
│ 1. Load Dataset  │──▶│ 2. Explore Data  │──▶│ 3. Identify Features │
│ (5200 × 16)      │   │ shape, columns,  │   │ & Target             │
│                  │   │ missing values   │   │ (spam_label)         │
└──────────────────┘   └──────────────────┘   └──────────┬───────────┘
                                                         ▼
┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│ 6. Train Models      │◀──│ 5. Train/Test Split  │◀──│ 4. Preprocessing     │
│ KNN · SVM · NB       │   │ 80 % / 20 %          │   │ Median Imputer +     │
│ (inside Pipelines)   │   │ stratified           │   │ StandardScaler       │
└──────────┬───────────┘   └──────────────────────┘   └──────────────────────┘
           ▼
┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│ 7. Evaluate          │──▶│ 8. Compare Models    │──▶│ 9. Conclusion        │
│ Accuracy, Precision, │   │ Tables + Graphs      │   │ Best model selection │
│ Recall, F1, Conf.Mat │   │                      │   │                      │
└──────────────────────┘   └──────────────────────┘   └──────────────────────┘
```

**Step-by-step:**
1. **Load** the CSV into a Pandas DataFrame.
2. **Explore** shape, column names and missing values.
3. **Separate** input features (`X`) and target (`y = spam_label`).
4. **Preprocess** using a Scikit-learn `Pipeline`: median imputation → standard scaling.
5. **Split** into 80 % training (4,160) and 20 % testing (1,040) with a fixed `random_state=42`.
6. **Train** KNN, SVM and Naive Bayes.
7. **Evaluate** each model on the test set.
8. **Compare** the models using a table and bar charts.
9. **Conclude** which model is most suitable.

---

## 🧠 Code Explanation & Conclusion (Question by Question)

> Questions 1–5 are the **theory** questions answered in the Part A PDF. Questions 6–22 are implemented in `main.ipynb`.

---

### 📘 PART A — Theory (Part A PDF)

#### Q1. What is Conditional Probability?
**Explanation:** Probability of event A happening *given that* B has already happened:  
`P(A | B) = P(A ∩ B) / P(B)`, where `P(B) > 0`. Knowing B shrinks the sample space to B.  
**✅ Conclusion:** Conditional probability lets us update the chance of an outcome when new information is known — the base idea behind Bayes' theorem and Naive Bayes.

#### Q2. Explain Bayes' Theorem and its importance in classification
**Explanation:** `P(Ck | X) = P(X | Ck) · P(Ck) / P(X)` → posterior = likelihood × prior / evidence. We compute this for every class and choose the highest. Example in the PDF: with only 1 % risky transactions, a detector that flags 90 % of risky and 5 % of safe ones gives a posterior of only ≈ 0.15.  
**✅ Conclusion:** Bayes' theorem gives the minimum-error decision rule and is the foundation of Naive Bayes. The **prior matters** — a good detector can still be wrong often when the event is rare.

#### Q3. What assumptions does the Naive Bayes classifier make?
**Explanation:** (1) Features are **conditionally independent** given the class, (2) every feature contributes equally, (3) features follow an assumed distribution (Gaussian / Multinomial / Bernoulli), (4) training data is representative.  
**✅ Conclusion:** The independence assumption is rarely true in real data, but Naive Bayes is still fast, works with little data and often performs well.

#### Q4. Working principle of KNN and SVM
- **KNN:** A lazy, distance-based learner. It stores training data, finds the **k closest** points (Euclidean distance) and assigns the **majority class**. Small k → flexible but noisy; large k → smoother. Feature scaling is important.
- **SVM:** Finds the **maximum-margin hyperplane** between classes. Only the closest points (**support vectors**) define the boundary. Soft margin (`C`) tolerates errors; kernels (RBF, polynomial) handle non-linear data.

**✅ Conclusion:** KNN decides by *proximity*, SVM decides by the *widest gap* between classes. KNN needs scaling and is slow on big data; SVM is strong in high dimensions but harder to tune.

#### Q5. Compare distance-based, probabilistic and margin-based classifiers

| Aspect | KNN (Distance) | Naive Bayes (Probabilistic) | SVM (Margin) |
|---|---|---|---|
| Core idea | Similar points are close | Highest posterior probability | Maximise gap between classes |
| Training | None (lazy) | Very fast | Slower (optimisation) |
| Prediction speed | Slow on large data | Very fast | Fast (uses support vectors) |
| Output | Class vote | Class probabilities | Class (distance to boundary) |
| Key assumption | Distance meaningful; needs scaling | Features independent given class | Margin-separable / suitable kernel |
| Strength | Simple, non-linear boundaries | Small data, text & spam tasks | High dimensions, robust to overfit |
| Weakness | Noise, scale, large k | Unrealistic independence | Hard to tune, less interpretable |

**✅ Conclusion:** No single family is best everywhere. Naive Bayes is quick and suits text/spam, KNN is simple but needs scaled features, SVM is powerful but needs tuning.

---

### 💻 PART B — Data Understanding & Preprocessing (Q6–Q8)

#### Libraries & Data Loading
```python
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

df = pd.read_csv("Message_Intelligence_Dataset_5200 - Message_Intelligence_Dataset_5200.csv")
```
**What it does:** Imports all required libraries, loads the dataset, prints `df.shape` → `(5200, 16)`, lists the columns and counts missing values per column (`df.isnull().sum()`).  
**Finding:** Only 3 columns have missing values — `sender_activity_score` (106), `sender_account_age_days` (113), `messages_sent_last_24h` (162).

#### Q6. Identify input features and target variable
```python
df = pd.read_csv("Message_Intelligence_Dataset_5200 - Message_Intelligence_Dataset_5200.csv")

print("Dataset Shape:", df.shape)
print("Dataset Columns:", df.columns.tolist())

print("Missing Values in each column:")
print(df.isnull().sum())
```
**What it does:** Everything except `spam_label` becomes the input `X`; `spam_label` is the target `y`. `value_counts()` shows the class balance (0 → 4,227, 1 → 973).  
**✅ Conclusion:** There are **15 input features** and **1 target**. The data is **imbalanced** (≈ 81 % not spam, 19 % spam), so accuracy alone can be misleading — precision, recall and F1 must be checked too.

#### Q7. Perform basic data preprocessing
```python

X = df.drop(columns=["spam_label"])
y = df["spam_label"]

print("Input Features:")
print(X.columns.tolist())

print("\nTarget Variable:")
print("risk_status")

print("\nTarget Class Distribution:")
print(y.value_counts())

```
**What it does:** Builds a pipeline that (1) fills missing values with the **median** and (2) standardises every feature to mean 0 / standard deviation 1.  
**Why median:** It is not affected by outliers. **Why scaling:** KNN and SVM are distance/margin based and would otherwise be dominated by large-valued columns such as `sender_account_age_days`.  
**✅ Conclusion:** Missing values are handled and features are brought to the same scale. Using a Pipeline also prevents **data leakage**, because the imputer/scaler are learned only from the training data.

#### Q8. Split the dataset into training and testing sets
```python
print("Missing values:")
print(X.isnull().sum())

preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

print("\nPreprocessing pipeline created.")
print("Missing values will be filled using median.")
print("Features will be scaled using StandardScaler.")

```
**What it does:** Splits the data 80 % / 20 %. `stratify=y` keeps the same spam ratio in both parts; `random_state=42` makes the result reproducible.  
**✅ Conclusion:** **4,160** training samples and **1,040** testing samples, with the same class proportions in both.

---

### 🔵 PART C — K-Nearest Neighbors (Q9–Q12)

#### Q9. Implement a KNN classifier
```python

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

```
**What it does:** KNN needs numbers, so only numeric columns are kept (`message_text` and `timestamp` are dropped). The data is split again with 13 numeric features.  
**✅ Conclusion:** KNN is trained on **13 numerical features** — 4,160 training rows and 1,040 testing rows.

#### Q10. Experiment with different values of K
```python
numeric_cols = X.select_dtypes(include=["number"]).columns

preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Keep only numerical columns for KNN
X = X[numeric_cols]

# Split dataset again
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Numerical features:", X.columns.tolist())
print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)
```
**What it does:** Trains KNN with five K values and plots accuracy against K.

| K | 3 | 5 | 7 | 9 | 11 |
|---|---|---|---|---|---|
| Accuracy | 1.000 | 1.000 | 1.000 | 0.999 | 0.999 |

**✅ Conclusion:** K = 3, 5 and 7 give perfect accuracy; accuracy drops very slightly at K = 9 and 11 because a larger K smooths the boundary too much. **K = 5** is chosen as a balanced value.

#### Q11. Analyze how distance metrics affect predictions
```python

distance_metrics = ["euclidean", "manhattan", "minkowski"]

for metric in distance_metrics:
    model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", KNeighborsClassifier(
            n_neighbors=5,
            metric=metric
        ))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nDistance Metric:", metric)
    print("Accuracy:",
          accuracy_score(y_test, predictions))
    print("F1 Score:",
          f1_score(y_test, predictions, zero_division=0))


```
| Metric | Accuracy | F1 Score |
|---|---|---|
| Euclidean | 1.0 | 1.0 |
| Manhattan | 1.0 | 1.0 |
| Minkowski | 1.0 | 1.0 |

**✅ Conclusion:** All three metrics give identical results on this dataset, so the metric choice has **no visible effect here**. (Euclidean and Minkowski with the default p = 2 are the same formula.)

#### Q12. Identify cases where KNN misclassifies messages
**What it does:** Finds test rows where `y_test != y_pred`, prints the misclassified rows, the confusion matrix and the classification report.
```


knn_model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", KNeighborsClassifier(n_neighbors=5))
])

knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)

y_pred_knn = pd.Series(y_pred_knn, index=X_test.index)

misclassified = X_test[
    y_test != y_pred_knn
].copy()

misclassified["Actual"] = y_test.loc[misclassified.index]
misclassified["Predicted"] = y_pred_knn.loc[misclassified.index]

print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Total Misclassified Messages:", len(misclassified))

print("\nFirst 10 Misclassified Messages:")
print(misclassified.head(10))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn, zero_division=0))
```

Confusion Matrix: `[[845, 0], [0, 195]]`  
**✅ Conclusion:** **0 misclassified messages** — all 845 non-spam and 195 spam test messages are predicted correctly (100 % precision and recall for both classes).

---

### 🟣 PART D — Support Vector Machine (Q13–Q15)

#### Q13. Implement SVM with Linear, RBF and Polynomial kernels
```python

kernels = ["linear", "rbf", "poly"]
svm_models = {}

for kernel in kernels:
    model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", SVC(
            kernel=kernel,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    svm_models[kernel] = model

    print("\nKernel:", kernel)
    print("Accuracy:",
          accuracy_score(y_test, predictions))
    print("Precision:",
          precision_score(
              y_test, predictions, zero_division=0
          ))
    print("Recall:",
          recall_score(
              y_test, predictions, zero_division=0
          ))
    print("F1 Score:",
          f1_score(
              y_test, predictions, zero_division=0
          ))

```
**What it does:** Trains three SVMs inside the same pipeline. `class_weight="balanced"` gives more importance to the minority (spam) class.

| Kernel | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Linear | 1.0 | 1.0 | 1.0 | 1.0 |
| RBF | 1.0 | 1.0 | 1.0 | 1.0 |
| Polynomial | 1.0 | 1.0 | 1.0 | 1.0 |

**✅ Conclusion:** All three kernels reach perfect scores, which means the two classes are **linearly separable** — a simple linear kernel is enough.

#### Q14. Analyze margin separation and support vectors
```python

svm_model = svm_models["linear"]

svm_classifier = svm_model.named_steps["classifier"]

print("Support vectors per class:")
print(svm_classifier.n_support_)

print("\nTotal Support Vectors:")
print(svm_classifier.n_support_.sum())

print("\nSupport vectors as a percentage of training data:")
print(round(
    svm_classifier.n_support_.sum()
    / len(X_train) * 100, 2
), "%")

print("\nLinear SVM Coefficients:")
print(svm_classifier.coef_)

```
- Support vectors: **8** (class 0) + **4** (class 1) = **12**
- That is only **0.29 %** of the training data
- The largest linear coefficient (≈ 1.02) belongs to **`num_special_chars`**, far above all others (the next largest are about 0.15)

**✅ Conclusion:** Only 12 points define the whole boundary, which indicates a **wide, clean margin** between spam and non-spam. `num_special_chars` is by far the most influential feature in the linear SVM.

#### Q15. Compare SVM performance with KNN

```

comparison = []

# KNN evaluation
knn_predictions = knn_model.predict(X_test)

comparison.append({
    "Model": "KNN (K=5)",
    "Accuracy": accuracy_score(y_test, knn_predictions),
    "Precision": precision_score(
        y_test, knn_predictions, zero_division=0
    ),
    "Recall": recall_score(
        y_test, knn_predictions, zero_division=0
    ),
    "F1 Score": f1_score(
        y_test, knn_predictions, zero_division=0
    )
})

# SVM evaluation
for kernel, model in svm_models.items():
    predictions = model.predict(X_test)

    comparison.append({
        "Model": "SVM (" + kernel + ")",
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        )
    })

comparison_df = pd.DataFrame(comparison)

print(comparison_df.round(3).to_string(index=False))

plt.figure(figsize=(9, 5))
plt.bar(comparison_df["Model"], comparison_df["Accuracy"])
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("KNN vs SVM Performance Comparison")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
```
**What it does:** Collects Accuracy / Precision / Recall / F1 for KNN (K = 5) and the three SVMs in one table and draws a bar chart.  
**✅ Conclusion:** KNN and all SVM kernels score **1.0 on every metric**, so on this dataset there is no performance difference. SVM is more compact (uses only 12 support vectors) and predicts faster, while KNN must store and search all training data.

---

### 🟢 PART E — Naive Bayes & Probability (Q16–Q19)

#### Q16. Implement Naive Bayes classifier
```python
nb_model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", GaussianNB())
])

# Train the model
nb_model.fit(X_train, y_train)

# Make predictions
y_pred_nb = nb_model.predict(X_test)

# Evaluate model
print("Naive Bayes Accuracy:",
      accuracy_score(y_test, y_pred_nb))

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred_nb, zero_division=0
))
```
**What it does:** Trains **Gaussian** Naive Bayes (suitable for continuous numerical features) and prints accuracy and the classification report.  
**✅ Conclusion:** Naive Bayes also reaches **Accuracy = 1.0** with perfect precision and recall for both classes.

#### Q18. Demonstrate Bayes' Theorem
```python

# Example probabilities
p_spam = 0.40
p_url_given_spam = 0.75
p_url_given_not_spam = 0.20

p_not_spam = 1 - p_spam

# Total probability of a URL
p_url = (
    p_url_given_spam * p_spam
    + p_url_given_not_spam * p_not_spam
)

# Bayes' theorem
p_spam_given_url = (
    p_url_given_spam * p_spam / p_url
)

print("Probability of Spam:", p_spam)
print("Probability of URL:", p_url)
print("Probability of Spam given URL:", p_spam_given_url)

```
**What it does:** Works through Bayes' theorem by hand. First the total probability of a URL is found (0.42), then the posterior.  
**✅ Conclusion:** A message containing a URL has a **71.4 % probability of being spam**, up from the prior of 40 %. This shows how evidence (a URL) updates our belief.

#### Q19. Compare theoretical probabilities with model predictions
```python

# Compare actual labels and Naive Bayes predictions
comparison = pd.DataFrame({
    "Actual": y_test,
    "Naive_Bayes_Predicted": y_pred_nb
})

print(comparison.head(10))

print("\nPrediction Agreement:")
print(
    (comparison["Actual"] ==
     comparison["Naive_Bayes_Predicted"]).mean()
)

```
**What it does:** Places actual labels beside Naive Bayes predictions and measures the agreement.  
**✅ Conclusion:** Agreement is **100 %**. The theoretical example (Q18) and the trained model follow the same principle — pick the class with the highest posterior — and in this dataset the strong signals (URLs, spam keywords) make the predictions match the real labels.

---

### 🟠 PART F — Model Comparison & Evaluation (Q20–Q22)

#### Q20–Q21. Compare KNN vs SVM vs Naive Bayes
```python
y_pred_knn = knn_model.predict(X_test)
y_pred_svm = svm_model.predict(X_test)
y_pred_nb = nb_model.predict(X_test)

# Compare models
models = {
    "KNN": y_pred_knn,
    "SVM": y_pred_svm,
    "Naive Bayes": y_pred_nb
}

results = []

for name, predictions in models.items():
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, average="binary",
            zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, average="binary",
            zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, average="binary",
            zero_division=0
        )
    })

results_df = pd.DataFrame(results)

print("Model Comparison:")
print(results_df.round(4))

# Create comparison graph
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

ax = results_df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(10, 6),
    width=0.8
)

plt.title("KNN vs SVM vs Naive Bayes")
plt.xlabel("Classification Models")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.legend(title="Metrics")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()
```
**What it does:** Computes Accuracy, Precision, Recall and F1 for all three models, stores them in `results_df` and plots a grouped bar chart.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| KNN | 1.0 | 1.0 | 1.0 | 1.0 |
| SVM | 1.0 | 1.0 | 1.0 | 1.0 |
| Naive Bayes | 1.0 | 1.0 | 1.0 | 1.0 |

**✅ Conclusion:** All three algorithms perform identically on this dataset.

#### Q22. Identify the best model
```python

best_precision_model = results_df.loc[
    results_df["Precision"].idxmax()
]

best_recall_model = results_df.loc[
    results_df["Recall"].idxmax()
]

print("Best Model for High Precision:")
print(best_precision_model)

print("\nBest Model for High Recall:")
print(best_recall_model)

```
**What it does:** Finds the model with the highest precision and the highest recall.  
**✅ Conclusion:** Because all scores are tied at 1.0, `idxmax()` simply returns the first row (**KNN**). In practice:
- **High precision** (avoid marking real messages as spam) and **high recall** (catch more spam) are equal here.
- **Naive Bayes** is the most practical choice for real-time spam filtering because it is the fastest to train and predict.
- **SVM** is best when new data is high-dimensional or harder to separate.
- **KNN** is simple but slow on large data.

---

## 📊 Final Model Comparison

| Model | Accuracy | Precision | Recall | F1 | Speed | Notes |
|---|---|---|---|---|---|---|
| KNN (K=5) | 1.00 | 1.00 | 1.00 | 1.00 | Slow prediction | Needs scaling, stores all data |
| SVM (Linear / RBF / Poly) | 1.00 | 1.00 | 1.00 | 1.00 | Fast prediction | Only 12 support vectors |
| Gaussian Naive Bayes | 1.00 | 1.00 | 1.00 | 1.00 | Fastest | Assumes independent features |

---

## ✅ Overall Conclusion

- Three classifiers — **KNN, SVM and Naive Bayes** — were built and evaluated for message spam detection.
- The data was cleaned (median imputation), scaled (StandardScaler) and split 80/20 with stratification.
- All models reached **100 % accuracy, precision, recall and F1** on the 1,040 test messages, with **zero misclassifications**.
- The dataset has very strong, clearly separable signals: `spam_keyword_score` and `num_special_chars` are 0 for every non-spam message, `num_special_chars` is at least 1 for every spam message, and `legit_keyword_score` is 0 for every spam message. This is why even a simple linear SVM separates the classes with only 12 support vectors.
- Precision helps reduce **false spam alarms**; recall helps **catch more real spam**. The best model depends on project needs, and this project gave hands-on practice in preprocessing, supervised learning, probability, evaluation and visualization.

### ⚠️ Notes & Future Improvements
- **Perfect scores are a warning sign.** Because features like `num_special_chars` and the keyword scores almost fully reveal the label, the dataset is probably easier than real-world messages. Testing on messy, real spam data would give a more honest picture.
- **`message_id` should not be a feature.** It is only an identifier and carries no meaning; it should be dropped before training (it is currently included in the numeric features).
- **Text is not used by the models.** `message_text` is ignored; adding **TF-IDF / Multinomial Naive Bayes** would use the actual words.
- Add **cross-validation**, **hyperparameter tuning** (`GridSearchCV` for K, C, gamma) and **feature engineering** (e.g. from `timestamp`).

---

## 📄 Part A — Theory File

> 📌 **Add your Part A PDF here.**

📎 **[Click here to open Part A: Probability & Conceptual Foundation (Theory) →](https://drive.google.com/file/d/17jr4pwl7UiExw8zIFdinm6t3LmkJGTOT/view?usp=sharing)**

<!-- Optional: if you upload the PDF to Google Drive, replace the link above with:
[Part A PDF](PASTE_YOUR_DRIVE_LINK_HERE) -->

---

## 🎥 Explanation Video

> 📌 **Add your explanation video link here.**

🔗 **Video Link:** (https://drive.google.com/file/d/16q9O_D5oxeWiJbRp2LAePa0uGd42ieGm/view?usp=sharing)

<!-- Example:
[![Watch the video](https://img.shields.io/badge/▶_Watch-Explanation_Video-red?style=for-the-badge&logo=youtube)](PASTE_YOUR_VIDEO_LINK_HERE)
-->

---

## ▶️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Message-Intelligence-System.git
cd Message-Intelligence-System

# 2. Install required libraries
pip install pandas numpy matplotlib scikit-learn jupyter

# 3. Open the notebook
jupyter notebook main.ipynb
```
Keep the dataset CSV in the same folder as `main.ipynb`. The notebook reads it as `Message_Intelligence_Dataset_5200 - Message_Intelligence_Dataset_5200.csv`, so rename the file if your copy has a different name.

---

## 👤 Author

**Darshil Kotadiya**  
Diploma in Computer Engineering | AI, Machine Learning & Data Science  
📧 kotadiyadarshil03@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/darshil-kotadiya-154b15418)
