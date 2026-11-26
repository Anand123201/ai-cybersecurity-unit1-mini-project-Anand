# AI Cybersecurity Unit 1 Mini Project

## AI-Based Social Media Threat Analysis and Classification

This project is my Unit 1 mini-project for **AI in Cybersecurity**. I
used a social-media dataset and applied data analysis, text
preprocessing, traditional machine learning, neural-network approaches,
and hyperparameter tuning to understand how AI can be used for
classification in a cybersecurity-related setting.

I completed the practical work in **Google Colab**. Along with model
accuracy, I focused on data cleaning, avoiding data leakage, using a
statement-level train/test split, comparing different models with
multiple evaluation metrics, and improving the Keras model through a
controlled validation-based tuning experiment.

> **Project status:** Tasks 1--5 are included in the current notebook.
> Task 5 adds structured hyperparameter tuning and model improvement
> after the initial deep-learning models in Task 4.

------------------------------------------------------------------------

## 1. Project Objectives

The main objectives of this project are:

1.  Set up a Python/Google Colab environment for the project.
2.  Understand the provided social-media dataset through exploratory
    data analysis (EDA).
3.  Check the dataset for missing values, duplicates, data types, and
    target distribution.
4.  Prepare tweet text for machine-learning classification.
5.  Convert text into numerical features using **TF-IDF**.
6.  Reduce the TF-IDF feature space using **TruncatedSVD**.
7.  Build a **Random Forest** classifier for the traditional
    machine-learning task.
8.  Build neural-network models using **MLPClassifier** and a **Keras
    Sequential Neural Network**.
9.  Improve the Keras model through structured hyperparameter tuning.
10. Compare the models using accuracy, precision, recall, F1-score,
    ROC-AUC, PR-AUC, and confusion matrices.
11. Understand ethical issues involved in using AI with social-media
    data.

------------------------------------------------------------------------

## 2. About the Dataset

The main dataset used in this project is:

``` text
Features_For_Traditional_ML_Techniques.csv
```

The dataset contains **134,198 records and 64 columns**. It includes
information related to social-media posts, statements, account activity,
and linguistic features.

Some important columns are:

-   `tweet` -- the social-media post related to the statement.
-   `statement` -- the statement/headline associated with the tweets.
-   `BinaryNumTarget` -- the binary target used for classification.
    -   `1` = True
    -   `0` = False
-   `followers_count` -- follower-related account information.
-   `friends_count` -- number of accounts followed.
-   `favourites_count` -- favourite/like activity.
-   `statuses_count` -- number of statuses/posts.
-   `listed_count` -- list-related account information.
-   `mentions`, `quotes`, `replies`, `retweets`, `favourites`,
    `hashtags`, and `URLs` -- social-media interaction features.
-   Linguistic features such as word counts, punctuation counts, capital
    letters, digits, and named-entity percentages.

### Target distribution

  Class       Meaning         Records   Percentage
  ----------- --------- ------------- ------------
  0           False            65,268       48.64%
  1           True             68,930       51.36%
  **Total**               **134,198**     **100%**

The target classes are relatively balanced.

### Data quality checks

In the executed notebook used for the final results:

-   Missing values reported by the EDA: **0**
-   Duplicate rows: **0**
-   Dataset shape after cleaning: **134,198 × 64**

For the text-processing stage, the notebook also handles missing tweet
text safely by filling missing text with an empty string before
vectorization.

### Why I used `BinaryNumTarget`

The dataset documentation describes `BinaryNumTarget` as the truth value
of the statement. I therefore use this column as the target variable.
The `tweet` column is used as the main text input for the
text-classification workflow.

------------------------------------------------------------------------

## 3. Project Workflow

``` text
Dataset
   |
   v
Data Loading
   |
   v
Data Understanding and EDA
   |
   v
Data Cleaning
   |
   v
Statement-Level Train/Test Split
   |
   v
Tweet Text Extraction
   |
   v
TF-IDF Vectorization
   |
   v
TruncatedSVD (50 Components)
   |
   +--------------------------+
   |                          |
   v                          v
Random Forest           Neural Network Models
                              |
                    +---------+----------+
                    |                    |
                    v                    v
              MLPClassifier       Keras Sequential
                                         |
                                         v
                               Task 5: Hyperparameter
                                      Tuning
   |                          |                    |
   +--------------------------+--------------------+
                              |
                              v
                       Model Evaluation
                              |
                              v
          Accuracy / Precision / Recall / F1-score
                    ROC-AUC / PR-AUC
```

------------------------------------------------------------------------

## 4. Task 1 -- Workspace Bootstrap and Environment Setup

I used **Google Colab** because it provides a ready-to-use Python
environment and makes it easier to run the notebook without configuring
everything locally.

The main libraries used are:

-   NumPy
-   Pandas
-   Scikit-learn
-   Matplotlib
-   Seaborn
-   TensorFlow / Keras

The notebook imports the required libraries and loads the provided CSV
dataset.

------------------------------------------------------------------------

## 5. Task 2 -- Cybersecurity Threat Landscape Analysis

### 5.1 Understanding the dataset

Before training any model, I checked:

-   Number of rows and columns
-   Column names and data types
-   First few records
-   Descriptive statistics
-   Missing values
-   Duplicate rows
-   Target distribution
-   Social-media activity patterns

This helped me understand the data before selecting the machine-learning
approach.

### 5.2 Data cleaning

I checked the dataset for missing values and duplicate rows. The
executed EDA for this notebook reported **zero missing values** and
**zero duplicate rows**.

The notebook also prepares the text and target columns for modelling.
Missing tweet text is handled safely using an empty string, and
`BinaryNumTarget` is converted to integer form for classification.

### 5.3 Exploratory data analysis

I explored social-media features such as:

-   Followers
-   Friends
-   Favourites
-   Statuses
-   Mentions
-   Quotes
-   Replies
-   Retweets
-   Hashtags
-   URLs

I plotted the target distribution and compared selected activity
features across the two target classes. These comparisons are
descriptive and do not prove that any single feature causes a particular
target label.

------------------------------------------------------------------------

## 6. Statement-Level Train/Test Split

One important decision in my project was how I split the dataset.

A statement can have multiple tweets associated with it. If I randomly
split individual rows, tweets related to the same statement could appear
in both training and testing data. This could make the evaluation look
better than it really is.

To reduce this risk, I split the data at the **statement level**.

The final experiment used:

-   **1,058 unique statements**
-   **80% of statements for training**
-   **20% of statements for testing**
-   **109,923 training rows**
-   **24,275 testing rows**
-   **0 overlapping statements between train and test**

This makes the evaluation more realistic because the same statement is
not intentionally represented in both sets.

------------------------------------------------------------------------

## 7. Text Preprocessing -- TF-IDF

For the classification task, I used the `tweet` column as the main text
input. The text was converted into numerical values using **TF-IDF**.

TF-IDF stands for **Term Frequency--Inverse Document Frequency**. It
gives more importance to words that help distinguish documents while
reducing the importance of very common words.

The configuration used was:

``` python
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=3,
    sublinear_tf=True,
    stop_words="english"
)
```

This configuration uses:

-   `max_features=5000` -- keeps up to 5,000 text features.
-   `ngram_range=(1, 2)` -- uses individual words (unigrams) and
    two-word combinations (bigrams).
-   `min_df=3` -- ignores terms appearing in fewer than three documents.
-   `sublinear_tf=True` -- applies sublinear scaling to term frequency.
-   `stop_words="english"` -- removes common English stop words.

I fitted the vectorizer on the training text and used the same fitted
vectorizer to transform the test text. This prevents the test set from
influencing feature construction.

------------------------------------------------------------------------

## 8. TruncatedSVD

TF-IDF can produce a large and sparse feature matrix. To make the
representation more manageable, I used **TruncatedSVD**.

``` python
TruncatedSVD(
    n_components=50,
    random_state=42
)
```

This reduced the TF-IDF representation from 5,000 features to **50
numerical components**. The notebook reported an explained variance of
**0.1267** for this representation.

The same reduced representation was used for the model comparison.

------------------------------------------------------------------------

## 9. Task 3 -- Traditional Machine Learning with Random Forest

For the traditional machine-learning part, I used **Random Forest**.

Random Forest combines multiple decision trees and uses their combined
predictions to make the final classification. It provides a traditional
machine-learning baseline for comparison with the neural-network models.

The configuration used in the project was:

``` python
RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)
```

The model was trained using the reduced TF-IDF/SVD representation and
evaluated on the held-out statement-level test set.

------------------------------------------------------------------------

## 10. Task 4 -- Basic Deep Learning

For Task 4, I implemented two neural-network approaches:

1.  **MLPClassifier**
2.  **Keras Sequential Neural Network**

Both models use the prepared features and the same statement-level
train/test split, making the comparison more consistent.

### 10.1 MLPClassifier

`MLPClassifier` is a feed-forward neural network available in
Scikit-learn. It uses fully connected layers and learns non-linear
relationships between input features and the target.

The MLP model was trained using the reduced TF-IDF/SVD representation.

### 10.2 Keras Sequential Neural Network

I also created a Keras Sequential Neural Network using TensorFlow/Keras.
The baseline architecture is:

``` text
Input
  |
Dense Layer – 64 neurons (ReLU)
  |
Dropout – 30%
  |
Dense Layer – 32 neurons (ReLU)
  |
Dropout – 20%
  |
Output Layer – 1 neuron (Sigmoid)
```

The main configuration includes:

-   ReLU activation in hidden layers
-   Sigmoid activation in the output layer
-   Adam optimizer
-   Binary cross-entropy loss
-   Early stopping
-   Batch size of 256
-   Standardized input features

Task 4 established the baseline neural-network models. I then extended
this work in Task 5 by testing different Keras training configurations.

------------------------------------------------------------------------

## 11. Task 5 -- Structured Hyperparameter Tuning and Model Improvement

After completing Task 4, I added a separate improvement stage to check
whether the Keras neural network could perform better with different
training settings.

Instead of changing parameters randomly or selecting a model based on
test accuracy, I compared three configurations using a **validation
set**. The held-out test set was kept separate and used only for the
final evaluation of the selected configuration.

### 11.1 Why hyperparameter tuning was needed

The initial Keras model achieved **76.85% test accuracy**. I wanted to
investigate whether changing the learning rate, dropout, regularization
strength, and batch size could improve the model's generalization.

The purpose was not to guarantee a higher score, but to conduct a
controlled experiment and compare the results fairly.

### 11.2 Training and validation split

The training portion was further divided into:

-   Training rows: **90,889**
-   Validation rows: **19,034**

The notebook used this validation set to compare the configurations. The
original held-out test set of 24,275 rows was not used for
hyperparameter selection.

### 11.3 Hyperparameter configurations

The network used hidden layers with **128, 64, and 32 neurons**. I
tested the following configurations:

  ------------------------------------------------------------------------------
  Configuration     Learning Rate        Dropout               L2     Batch Size
                                                   Regularization 
  ---------------- -------------- -------------- ---------------- --------------
  Config 1 --               0.001           0.30           0.0005            128
  Balanced                                                        

  Config 2 --              0.0005           0.35            0.001            128
  Lower Learning                                                  
  Rate                                                            

  Config 3 --              0.0001           0.45           0.0015            256
  Stronger                                                        
  Regularization                                                  
  ------------------------------------------------------------------------------

### 11.4 Training callbacks

I used two callbacks to manage training:

**EarlyStopping**

-   Monitors validation loss.
-   Stops training when validation loss stops improving.
-   Restores the model weights from the best epoch.

This helps avoid unnecessary training and can reduce overfitting.

**ReduceLROnPlateau**

-   Monitors validation loss.
-   Reduces the learning rate when validation loss stops improving.
-   Allows the model to continue training with smaller updates.

### 11.5 Validation results

The validation accuracies reported by the notebook were:

  Configuration                           Validation Accuracy
  ------------------------------------- ---------------------
  Config 1 -- Balanced                                 81.35%
  Config 2 -- Lower Learning Rate                      79.22%
  Config 3 -- Stronger Regularization              **83.48%**

**Config 3 -- Stronger Regularization** was selected because it achieved
the highest validation accuracy of **83.48%** and a validation loss of
**0.6328**.

Its settings were:

-   Learning rate: `0.0001`
-   Dropout: `0.45`
-   L2 regularization: `0.0015`
-   Batch size: `256`

### 11.6 Final test results of the tuned Keras model

After selecting Config 3 using validation performance, I evaluated it on
the original held-out test set.

  Metric        Tuned Keras Result
  ----------- --------------------
  Accuracy              **78.12%**
  Precision                 74.81%
  Recall                    87.91%
  F1-score                  80.83%
  ROC-AUC                   84.24%
  PR-AUC                    82.01%

The tuned Keras model improved test accuracy from **76.85% to 78.12%**,
an increase of **1.27 percentage points** compared with the original
Keras model.

However, the tuned Keras model did not outperform Random Forest, which
achieved 78.72% test accuracy. This is an important result:
hyperparameter tuning improved the Keras baseline, but it did not make
Keras the best model in the overall comparison.

------------------------------------------------------------------------

## 12. Model Evaluation Metrics

I did not rely only on accuracy. I evaluated the models using:

-   **Accuracy** -- overall percentage of correctly classified samples.
-   **Precision** -- how many samples predicted as positive were
    actually positive.
-   **Recall** -- how many actual positive samples were correctly
    detected.
-   **F1-score** -- harmonic mean of precision and recall.
-   **ROC-AUC** -- measures how well the model separates the two classes
    across different classification thresholds.
-   **PR-AUC** -- summarizes the precision-recall relationship.
-   **Confusion Matrix** -- shows true positives, true negatives, false
    positives, and false negatives.

Using multiple metrics gives a more complete view of the strengths and
weaknesses of each model.

------------------------------------------------------------------------

## 13. Final Model Comparison

The following table contains the final test results reported by the
executed notebook:

  --------------------------------------------------------------------------------------------
  Model               Accuracy   Precision       Recall     F1-score      ROC-AUC       PR-AUC
  --------------- ------------ ----------- ------------ ------------ ------------ ------------
  Random Forest     **78.72%**      75.17%       88.79%   **81.41%**       85.28%   **84.47%**

  Tuned Keras           78.12%      74.81%       87.91%       80.83%       84.24%       82.01%

  Original Keras        76.85%      71.90%   **91.74%**       80.62%   **85.74%**       83.21%

  MLPClassifier         73.22%      74.68%       74.10%       74.39%       81.57%       81.90%
  --------------------------------------------------------------------------------------------

### Observations

-   Random Forest achieved the highest test accuracy at **78.72%**.
-   Random Forest also achieved the highest F1-score at **81.41%**.
-   The original Keras model achieved the highest recall at **91.74%**.
-   The tuned Keras model improved accuracy over the original Keras
    model by **1.27 percentage points**.
-   MLPClassifier had the lowest accuracy among the four models in this
    experiment.
-   The model with the highest accuracy is not automatically the best
    choice for every application; the importance of precision and recall
    depends on the use case.

------------------------------------------------------------------------

## 14. ROC Curve and Confusion Matrix Comparison

I also use ROC curves and confusion matrices to examine model behaviour.

The ROC curve plots True Positive Rate against False Positive Rate. A
higher ROC-AUC generally indicates better ability to distinguish between
the two classes across different thresholds.

The confusion matrix helps identify:

-   True Positives
-   True Negatives
-   False Positives
-   False Negatives

The notebook includes evaluation outputs for the baseline models and the
tuned Keras model.

------------------------------------------------------------------------

## 15. Ethical Considerations

Since the project uses social-media data, I considered the following
ethical issues:

### Privacy

Social-media data can contain information about real users. Data should
only be used for legitimate academic or research purposes and handled
responsibly.

### False Positives

An AI model can incorrectly classify a normal post. A prediction should
not automatically be treated as proof of malicious or misleading
behaviour.

### False Negatives

A model can fail to detect a genuinely relevant example. Missed
detections can be important in cybersecurity applications.

### Bias

Social-media language differs across communities, topics, writing
styles, and languages. If training data is not representative, model
performance may vary across different types of content.

### Human Oversight

I would use the model as a decision-support system rather than allowing
it to make high-impact decisions completely on its own.

### Data Leakage

I considered data leakage while designing the train/test split. Since
multiple tweets can belong to the same statement, I used a
statement-level split to avoid placing related statements into both
training and testing sets.

------------------------------------------------------------------------

## 16. Limitations

There are some limitations in this project:

1.  The project uses a provided dataset, so its quality and
    representativeness affect model performance.
2.  The project uses introductory classification models rather than a
    production-level cybersecurity detection platform.
3.  TF-IDF does not understand language in the same way as modern
    transformer-based language models.
4.  The project does not use real-time social-media streams.
5.  The current notebook focuses on the binary target in the supplied
    dataset; it does not train separate enterprise models for phishing,
    malware, account compromise, DDoS, or data exfiltration.
6.  Model predictions should not be treated as final decisions without
    additional validation.
7.  Results depend on preprocessing, data splitting, feature
    representation, and model parameters.

------------------------------------------------------------------------

## 17. Project Structure

The repository can be organised as follows:

``` text
ai-cybersecurity-unit1-mini-project-anand/
|
├── README.md
├── requirements.txt
├── .gitignore
|
├── data/
|   └── Features_For_Traditional_ML_Techniques.csv
|
└── src/
    └── AI_Cybersecurity_Unit1_Mini_Project_Anand.ipynb
```

### Dataset note

The CSV dataset is relatively large, so it may not be uploaded directly
to GitHub if it exceeds repository limits.

If the dataset is kept outside GitHub, the notebook expects it at:

``` text
data/Features_For_Traditional_ML_Techniques.csv
```

Do not commit the dataset if its licence or size makes that
inappropriate. Explain in the repository where the dataset can be
obtained.

------------------------------------------------------------------------

## 18. Technologies Used

### Programming Language and Environment

-   Python
-   Google Colab
-   Jupyter Notebook

### Data Processing

-   Pandas
-   NumPy

### Machine Learning

-   Scikit-learn
-   Random Forest
-   MLPClassifier
-   TF-IDF
-   TruncatedSVD
-   StandardScaler

### Deep Learning and Model Improvement

-   TensorFlow
-   Keras
-   Sequential Neural Network
-   Dense layers
-   Dropout
-   L2 regularization
-   EarlyStopping
-   ReduceLROnPlateau
-   Hyperparameter tuning

### Visualization

-   Matplotlib
-   Seaborn

------------------------------------------------------------------------

## 19. How to Run the Project

### Option 1 -- Google Colab

1.  Open the `.ipynb` file in Google Colab.

2.  Upload the dataset:

    ``` text
    Features_For_Traditional_ML_Techniques.csv
    ```

3.  Make sure the CSV path matches the path used in the notebook.

4.  Run the notebook from the first cell to the last cell.

5.  Check the EDA outputs and data-cleaning results.

6.  Check the statement-level train/test split.

7.  Check the TF-IDF and TruncatedSVD outputs.

8.  Check the Random Forest results.

9.  Check the MLPClassifier results.

10. Check the original Keras results.

11. Run Task 5 to compare the three hyperparameter configurations.

12. Check the tuned Keras test results and final model comparison.

**Important:** I recommend using **Runtime → Restart session** before
the final run and then executing the notebook from top to bottom. This
helps avoid old variables or outputs from previous experiments affecting
the final results.

### Option 2 -- Local Jupyter Notebook

1.  Clone or download the repository.
2.  Install the dependencies listed in `requirements.txt`.
3.  Place the dataset in the expected `data/` folder or update the
    notebook path.
4.  Open the notebook using Jupyter Notebook or JupyterLab.
5.  Run all cells in order.

------------------------------------------------------------------------

## 20. Requirements

The main dependencies are:

``` text
numpy
pandas
scikit-learn
tensorflow
matplotlib
seaborn
jupyter
notebook
```

------------------------------------------------------------------------

## 21. Final Conclusion

In this mini-project, I worked through data understanding, exploratory
analysis, cleaning, text processing, traditional machine learning, and
deep learning. I used TF-IDF to represent tweet text and TruncatedSVD to
reduce the feature dimension before training the models.

For the machine-learning task, I trained a Random Forest classifier. For
deep learning, I explored MLPClassifier and a Keras Sequential Neural
Network.

After completing Task 4, I extended the project with **Task 5 --
Structured Hyperparameter Tuning and Model Improvement**. I tested three
Keras configurations with different learning rates, dropout values, L2
regularization strengths, and batch sizes. I used validation data to
select the configuration and kept the test set separate for final
evaluation. EarlyStopping and ReduceLROnPlateau were used to manage
training.

The tuned Keras model achieved **78.12% test accuracy**, compared with
**76.85%** for the original Keras model. This was an improvement of
**1.27 percentage points**. However, Random Forest remained the best
model by test accuracy at **78.72%**.

This work helped me understand that changing model settings does not
automatically make a model better. Models must be compared using a fair
validation process and multiple evaluation metrics. From a cybersecurity
perspective, privacy, bias, false positives, false negatives, data
leakage, and human oversight are also important considerations.

------------------------------------------------------------------------
