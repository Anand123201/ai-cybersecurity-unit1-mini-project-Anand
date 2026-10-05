# AI Cybersecurity Unit 1 Mini Project

## AI-Based Social Media Threat Analysis and Classification

This project is my Unit 1 mini-project for **AI in Cybersecurity**. I used a social-media dataset and applied basic data analysis, text preprocessing, machine learning, and neural-network approaches to understand how AI can be used for classification in a cybersecurity-related setting.

I completed the project in **Google Colab** and kept the workflow simple enough to understand and explain during a viva. Instead of only focusing on getting a high accuracy score, I also focused on data cleaning, avoiding data leakage, using a fair train/test split, and comparing different models using multiple evaluation metrics.

---

## 1. Project Objectives

The main objectives of this project are:

1. Set up a Python/Google Colab environment for the project.
2. Understand the provided social-media dataset through exploratory data analysis (EDA).
3. Check the dataset for missing values, duplicates, data types, and target distribution.
4. Prepare tweet text for machine-learning classification.
5. Convert text into numerical features using **TF-IDF**.
6. Reduce the TF-IDF feature space using **TruncatedSVD**.
7. Build a **Random Forest** classifier for the traditional machine-learning task.
8. Build basic neural-network models using **MLPClassifier** and a **Keras Sequential Neural Network**.
9. Compare the models using accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC, confusion matrices, and ROC curves.
10. Understand the ethical issues involved in using AI with social-media data.

---

## 2. About the Dataset

The main dataset used in this project is:

```text
Features_For_Traditional_ML_Techniques.csv
```

The provided dataset contains **134,198 records and 64 columns**.

It contains information related to Twitter/social-media posts, statements, account activity, and linguistic features. Some important columns used or discussed in the project are:

- `tweet` – the Twitter post related to the statement.
- `statement` – the statement/headline associated with the tweets.
- `BinaryNumTarget` – the binary target used for classification.
  - `1` = True
  - `0` = False
- `followers_count` – follower-related account information.
- `friends_count` – number of accounts followed.
- `favourites_count` – favourite/like activity.
- `statuses_count` – number of statuses/posts.
- `listed_count` – list-related account information.
- Other linguistic and social-media activity features.

The original target distribution is:

| Class | Meaning | Records |
|---|---|---:|
| 0 | False | 65,268 |
| 1 | True | 68,930 |
| **Total** | | **134,198** |

The dataset therefore contains both classes and is suitable for binary classification.

### Why I used `BinaryNumTarget`

The dataset README describes `BinaryNumTarget` as the truth value of the statement. I therefore use this column as my target variable.

For the text-classification part, I use the related `tweet` text as the main input. This lets me demonstrate a complete text-processing workflow instead of directly using a feature that could reveal the target.

---

## 3. Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Understanding / EDA
   ↓
Data Cleaning
   ↓
Statement-Level Train/Test Split
   ↓
Tweet Text Extraction
   ↓
TF-IDF Vectorization
   ↓
TruncatedSVD
   ↓
 ┌─────────────────────────────────────┐
 │                                     │
Random Forest                 Neural Network Models
 │                           │
 │                    ┌──────┴───────────┐
 │                    │                  │
 │             MLPClassifier     Keras Sequential
 │
 └──────────────────┬────────────────────┘
                    ↓
             Model Evaluation
                    ↓
 Accuracy / Precision / Recall
 F1 / ROC-AUC / PR-AUC
 Confusion Matrix / ROC Curve
```

---

# 4. Task 1 – Workspace Bootstrap and Environment Setup

I used **Google Colab** because it provides a ready-to-use Python environment and makes it easier to run the notebook without configuring everything locally.

The main libraries used in the project are:

- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- TensorFlow / Keras

The notebook first imports the required libraries and loads the provided CSV dataset.

---

# 5. Task 2 – Cybersecurity Threat Landscape Analysis

## 5.1 Understanding the Dataset

Before training any model, I first checked the basic structure of the dataset.

I looked at:

- Number of rows and columns
- Column names
- Data types
- First few records
- Descriptive statistics
- Missing values
- Duplicate rows
- Target distribution
- Social-media activity patterns

This helped me understand what type of data I was working with before selecting the machine-learning approach.

## 5.2 Data Cleaning

I performed basic data-cleaning checks before starting the modelling stage.

### Missing values

I checked every column for missing values. The EDA showed missing values in some of the linguistic feature columns.

For text processing, missing tweet values were handled by replacing them with an empty string:

```python
df["tweet"].fillna("").astype(str)
```

This prevents missing text values from causing problems during TF-IDF vectorization.

### Duplicate records

I also checked for duplicate rows. The EDA showed no duplicate rows in the dataset used for the final experiment.

### Target validation

I checked the target column to make sure that both classes were available:

```python
df["BinaryNumTarget"].value_counts()
```

This was important because a binary classifier needs examples from both classes.

## 5.3 Exploratory Data Analysis

I explored different social-media features such as:

- Followers
- Friends
- Favourites
- Statuses
- Mentions
- Quotes
- Replies
- Retweets
- Hashtags
- URLs

I also plotted the target distribution and compared selected activity features across the two target classes.

The purpose of the EDA was not just to make graphs. It helped me understand the type of behaviour represented in the dataset and decide how I wanted to build the classification pipeline.

---

# 6. Statement-Level Train/Test Split

One of the most important decisions in my project was how I split the dataset.

A statement can have multiple tweets associated with it. If I randomly split individual rows, tweets belonging to the same statement could appear in both the training and testing datasets. That could make the model appear more accurate than it really is.

To avoid this, I first created a list of unique statements and split the statements into training and testing groups.

The final experiment used:

- **1,058 unique statements**
- **80% statements for training**
- **20% statements for testing**
- **109,923 training rows**
- **24,275 testing rows**
- **0 overlapping statements between train and test**

This makes the evaluation more realistic because the same statement is not intentionally represented in both sets.

---

# 7. Text Preprocessing

For the classification task, I used the `tweet` column as the main text input.

The text was converted into numerical values using **TF-IDF**.

## What is TF-IDF?

TF-IDF stands for **Term Frequency–Inverse Document Frequency**. In simple terms, it gives higher importance to words that are useful for distinguishing documents while reducing the importance of very common words.

I used:

```python
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=3,
    sublinear_tf=True,
    stop_words="english"
)
```

This configuration uses both unigrams (individual words) and bigrams (pairs of words).

I fitted the vectorizer only on the training text and then transformed the test text using the same fitted vectorizer. This prevents information from the test set from influencing the feature-building process.

---

# 8. TruncatedSVD

TF-IDF can produce a large and sparse feature matrix. To make the representation more manageable, I used **TruncatedSVD**.

```python
TruncatedSVD(
    n_components=50,
    random_state=42
)
```

This reduced the TF-IDF representation to **50 numerical components**.

The same reduced representation was used by the models so that the model comparison remained consistent.

---

# 9. Task 3 – Random Forest

For the traditional machine-learning part, I used **Random Forest**.

Random Forest combines multiple decision trees and uses their combined predictions to make the final classification.

The model configuration used in the project is:

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)
```

I selected Random Forest because it is a well-known classification algorithm and gives me a good traditional machine-learning baseline.

The model was trained on the reduced TF-IDF/SVD representation.

---

# 10. Task 4 – Basic Deep Learning

For Task 4, I used **two basic neural-network approaches**:

1. MLPClassifier
2. Keras Sequential Neural Network

Both models use the same processed features and the same statement-level train/test split. This makes the comparison more consistent.

## 10.1 MLPClassifier

`MLPClassifier` is a feed-forward neural network available in Scikit-learn. The model uses fully connected layers and learns non-linear relationships between the input features and the target.

The MLP model was trained using the same reduced TF-IDF/SVD representation used for Random Forest.

## 10.2 Keras Sequential Neural Network

I also added a Keras Sequential Neural Network as a more direct introduction to neural-network modelling.

The basic architecture is:

```text
Input
  ↓
Dense Layer – 64 neurons
  ↓
Dropout – 30%
  ↓
Dense Layer – 32 neurons
  ↓
Dropout – 20%
  ↓
Output Layer – 1 neuron
  ↓
Sigmoid
```

The main configuration is:

```python
Sequential([
    Input(shape=(number_of_features,)),
    Dense(64, activation="relu"),
    Dropout(0.30),
    Dense(32, activation="relu"),
    Dropout(0.20),
    Dense(1, activation="sigmoid")
])
```

I used:

- **ReLU** activation in the hidden layers
- **Sigmoid** activation in the output layer
- **Adam** optimizer
- **Binary cross-entropy** loss
- **Early stopping**
- Batch size of 256

The features were standardized before training the Keras model.

---

# 11. Model Evaluation

I did not rely only on accuracy.

I evaluated the models using:

### Accuracy
Shows the overall percentage of correctly classified samples.

### Precision
Shows how many samples predicted as positive were actually positive.

### Recall
Shows how many actual positive samples were correctly detected.

### F1-score
Combines precision and recall into one score.

### ROC-AUC
Measures how well the model separates the two classes across different classification thresholds.

### PR-AUC
Provides another view of classification performance using the precision-recall relationship.

### Confusion Matrix
Shows True Positives, True Negatives, False Positives, and False Negatives.

---

# 12. ROC Curve Comparison

I also compare the ROC curves of the models.

The ROC curve plots True Positive Rate against False Positive Rate. The AUC value is included in the graph.

A higher ROC-AUC generally means that the model has better ability to distinguish between the two classes across different thresholds.

For the final notebook, I compare the ROC curves of:

- Random Forest
- MLPClassifier
- Keras Sequential Neural Network

The comparison is performed on the same test set.

---

# 13. Results and Observations

The final results depend on the exact notebook run because neural-network training can vary slightly.

For the Random Forest and MLP experiment, I observed that the statement-level split produced a much more realistic evaluation than the earlier row-level experiment.

The important observation for me was that changing the data-splitting strategy can have a major effect on the reported accuracy.

I compare:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Training time

Rather than deciding that one model is always better, I use these metrics to understand the strengths and weaknesses of each approach.

### What I learned from the comparison

- Random Forest provides a strong traditional baseline.
- MLPClassifier provides a simple neural-network baseline.
- Keras gives me more direct control over the neural-network architecture.
- ROC-AUC is useful because it evaluates model discrimination across different thresholds.
- A model with higher accuracy is not automatically the best model for every cybersecurity application.
- The data split and feature representation are very important for trustworthy evaluation.

---

# 14. Ethical Considerations

Since the project uses social-media data, I also considered some ethical issues.

### Privacy

Social-media data can contain information about real users. The data should only be used for legitimate academic or research purposes and should be handled responsibly.

### False Positives

An AI model can incorrectly classify a normal post as suspicious or incorrect. In a real cybersecurity system, such predictions should not automatically be treated as proof of malicious behaviour.

### False Negatives

A model can also fail to detect a genuinely suspicious or misleading example. This can be important in cybersecurity because missed detections can have consequences.

### Bias

Social-media language differs between communities, topics, writing styles and languages. If the training data is not representative, the model may perform differently on different types of content.

### Human Oversight

I would use the model as a decision-support system rather than allowing it to make high-impact decisions completely on its own.

### Data Leakage

I specifically considered data leakage while designing the train/test split. Since multiple tweets can belong to the same statement, I used a statement-level split to avoid placing related statements into both training and testing sets.

---

# 15. Limitations

There are some limitations in this project:

1. The project uses a provided dataset, so the quality and representativeness of the dataset directly affect the models.
2. The models are basic introductory models rather than production-level cybersecurity systems.
3. TF-IDF does not understand language in the same way as modern transformer-based language models.
4. The project does not use real-time social-media streams.
5. Model predictions should not be treated as final decisions without additional validation.
6. The results depend on the selected preprocessing steps, train/test split and model parameters.

---

# 16. Project Structure

The GitHub repository is organised as follows:

```text
ai-cybersecurity-unit1-mini-project-anand/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── Features_For_Traditional_ML_Techniques.csv
│
└── src/
    └── AI_Cybersecurity_Unit1_Mini_Project_Anand.ipynb
```

### Dataset note

The CSV dataset is relatively large, so it may not be uploaded directly to GitHub if it exceeds the repository/browser upload limit.

If the dataset is kept outside GitHub, the notebook expects it at:

```text
data/Features_For_Traditional_ML_Techniques.csv
```

The dataset should not be committed to the repository simply to make the repository look complete if the file is too large. The README can be used to explain where the dataset is obtained from.

---

# 17. Technologies Used

### Programming Language

- Python

### Environment

- Google Colab
- Jupyter Notebook

### Data Processing

- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- Random Forest
- MLPClassifier
- TF-IDF
- TruncatedSVD
- StandardScaler

### Deep Learning

- TensorFlow
- Keras
- Sequential Neural Network
- Dense layers
- Dropout
- Early Stopping

### Visualization

- Matplotlib
- Seaborn

---

# 18. How to Run the Project

## Option 1 – Google Colab

1. Open the `.ipynb` file in Google Colab.
2. Upload the dataset:

```text
Features_For_Traditional_ML_Techniques.csv
```

3. Make sure the CSV path matches the path used in the notebook.
4. Run the notebook from the first cell to the last cell.
5. Check the EDA outputs.
6. Check the Random Forest results.
7. Check the MLPClassifier results.
8. Check the Keras results.
9. Check the confusion matrices and ROC curves.

### Important

I recommend using **Runtime → Restart session** before the final run and then executing the notebook from top to bottom.

This helps avoid old variables or outputs from previous experiments affecting the final results.

---

# 19. Requirements

The main dependencies are listed in `requirements.txt`:

```text
numpy
pandas
scikit-learn
tensorflow
matplotlib
seaborn
jupyter
notebook
```

---

# 20. Final Conclusion

Overall, this project helped me understand how AI techniques can be applied to a cybersecurity-related social-media classification problem.

I started with basic data understanding and EDA, then cleaned and prepared the data before building the models. One of the most important things I learned was that the way the dataset is split can have a major effect on the final evaluation.

For the traditional machine-learning part, I used Random Forest. For the neural-network part, I used MLPClassifier and a Keras Sequential Neural Network. All the models were evaluated using the same statement-level test split so that the comparison was more consistent.

The project also helped me understand why accuracy alone is not enough. Metrics such as precision, recall, F1-score, ROC-AUC and PR-AUC give a more complete picture of model performance.

From a cybersecurity perspective, I also learned that an AI model should be treated as a support tool. Privacy, bias, false positives, false negatives, data leakage and human oversight are important considerations before using such a system in a real-world environment.

