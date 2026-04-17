# CSCI-4907-NLU Project

## Team Contributors
| S/N | Team Members | Part |
| :-: | :- | :- |
| 1 | Zhan You Lau | Data Preparation & Cleaning, Data Visualization, Error Partitioning, Conclusion, Consolidation, Statistical Analysis, Final Report |
| 2 | Yu Chen Law | Data Preparation & Cleaning, Machine Learning Models, Master Error Table, Cross Model Behaviour Analysis, Statistical Analysis |
| 3 | Kieran E Kai Voo | Machine Learning Models, Weights saving, Misclassification Pattern Analysis, Bi-LSTM refinemeent, Final Report |
| 4 | Joshua, Tse Ern Foo | Data Visualization, Machine Learning Models, Qualitative Error Analysis, LLM Benchmarking |

## About/ Problem Statement
Social media has become ubiquitous in our everyday communication which contributes to the increased prevalence of cyberbullying. However, cyberbullying detection is particularly challenging in a multilabel setting as harmful content may belong to multiple overlapping categories like threats, insults, or implicit aggression. These categories often rely on subtle linguistic cues such as sarcasm and contextual ambiguity. This makes accurate classification difficult.

Our project examines how different modeling approaches handle the complexities of multilabel cyberbullying detection as well as evaluates their strengths and limitations beyond overall performance metrics.

### Aim
To develop and compare multiple machine learning and deep learning models for multilabel cyberbullying detection in tweets, examining their performance and behavioral limitations.

### Objective
- Perform data preparation, preprocessing, and exploratory analysis
- Implement classical machine learning models and Bi-LSTM for comparison
- Evaluate and compare model performance using multilabel metrics
- Conduct structured error analysis to examine linguistic failure cases

### Possibilities that this insights can be beneficial for
- Help determine which models perform better for multilabel cyberbullying detection
- Understand common linguistic challenges such as sarcasm and implicit aggression
- Improve how overlapping labels are handled in multilabel tasks

## Datasets
Our dataset is taken from Kaggle: [Cyberbullying Classification](https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification)

## <a id ="repository">🔎 Repository Overview </a>
> - Use this section links to quickly and conveniently jump to each section.  
> - At every section there is the "[Back to `Main` Content Page](#repository)" to jump back and forth seeamlessly.
1) [Set up](#setup)  
2) [Source Code](#source)  
3) [Data Preparation & Cleaning](#data)
4) [Exploratory Analysis](#analysis)
   - [Number Game](#number)
   - [Tokenization](#token)
   - [Word Cloud](#word)
   - [Sentiment Analysis](#sentiment)
5) [Machine Learning](#machine)
   - [Naive Bayes](#naive)
   - [Multinomial Logistic Regression](#logistic)
   - [Support Vector Machine](#support)
   - [Random Forest Classifier](#random)
   - [Bi-LSTM](#bert)
6) [Model Results + Comparison](#results)
7) [Structured Error Analysis](#error-analysis)
8) [Statistical Analysis](#statistical-analysis)
9) [Challenges Faced](#challenges)
10) [Conclusion](#conclusion)

## <a id="setup"> ⚙️ Set up </a>
[Back to `Main` Content Page](#repository)

Clone the repository and install dependencies:

```bash
# Clone the repo
git clone https://github.com/donkey-king-kong/CSCI-4907-NLU.git
cd CSCI-4907-NLU

# Create and activate a virtual environment
# Windows:
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux:
python3 -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

If you use **NLTK** in the notebooks, run once in Python to download stopwords:

```python
import nltk
nltk.download('stopwords')
```

## <a id="source"> 💻 Source Code </a>
[Back to `Main` Content Page](#repository)  

> Source Code on Google Colab: https://colab.research.google.com/drive/1KTm6iu-XBtcy4Y7H8wsIBvkC3YWAYt4G

## <a id = "data">🧼 Data Preparation & Cleaning</a>
[Back to `Main` Content Page](#repository)
### What we removed
> - Remove mentions (@username)  
> - Remove punctuations  
> - Remove URLs  
> - Remove extra whitespaces  
> - Remove stopwords  
> - Remove HTML characters (EG: "&amp")  
> - Remove numbers  
> - Remove picture links (EG: pic.twitter.com)  
> - Remove shortwords (Length <= 2)  

## <a id = "analysis">🔬 Exploratory Analysis</a> 
[Back to `Main` Content Page](#repository)

To analyse and visualze the data we have cleaned in order to understand its underlying patterns, relationships and anomalies. We will be using data visualization techniques in hopes of generating insights that could help us better understand the data before applying any models or conducting any hypothesis testing.

### <a id = "number">🔢 Number Game</a>
> The "numbers game" is used in our exploratory data analysis where we systematically examine the numerical data to identify patterns, trends and anmoalies. 
>
> Here, we plot the number of tweets belonging to each category in the dataset as well as their relative percentages. 

### <a id = "token">🪙 Tokenization</a>
> Here, we used tokenization to break down a piece of text like sentences or paragraphs into individual worlds or "tokens". From this plot, we can see the most common words in the tweets of our data.

### <a id = "word">🔠 Word Cloud</a>
> For this section, we used a WordCloud to present the most commonly seen words according to each **classified** category.
> The presence of each words in a tweet will increase its corresponding probability towards being classified into its respective category.

#### Word Cloud was generated for the following 
> - Gender Categories 
> - Religion Categories  
> - Age Categories  
> - Ethnicity Categories  
> - Other Cyberbullying Categories 
> - Not Cyberbullying Related    

### <a id = "sentiment">📈 Sentiment Analysis</a>
> - For sentiment analysis, we used the the module TextBlob for natural language processing tasks. The sentiment analysis model considers various factors such as word polarity, intensity of sentiment, and context to determine the sentiment score for a given text.  
>
> - This would help us in identifying sentiments - positive, negative and neutral, from a piece of text.
    
The sentiment score represents the polarity of the text (Positive, Negative, Neutral). It is a floating point number ranging from -1.0 to +1.0. 
 
> - If the sentiment score is close to 1.0, it indicates a very positive sentiment.  
> - If the sentiment score is close to -1.0, it indicates a very negative sentiment.  
> - If the sentiment score is around 0.0, it indicates a neutral sentiment.  

## <a id = "machine">🤖 Machine Learning</a>
[Back to `Main` Content Page](#repository) 

We used algorithms and statistical models that allow us to learn from our data and make any predicitons or decisions without explicitly programming it. It helps us identify patterns across our large datasets efficiently.

> - [Naive Bayes](#naive)  
> - [Multinomial Logistic Regression](#logistic)  
> - [Support Vector Machine](#support)  
> - [Random Forest Classifier](#random)  
> - [Bi-LSTM](#bert)

#### 📇 Results for each model are:
Statistical Results  
> - Shows a classification report on:
>   - Precision
>   - Recall
>   - f1-score
>   - Support
>   - Accuracy
>   - Macro average
>   - Weighted average 

Confusion Matrix  
> Shows the matrix of true vs predicted for each category    
  
ROC Curve
> - We included this ROC curve to illustrate the balance between true positive rate (TPR) and false positive rate (FPR) across different thresholds.
> - A model excels when its curve hugs the top-left corner, indicating high TPR and low FPR. Conversely, a curve closer to the diagonal line signifies poor ability to discriminate, no better than random chance.
  
Learning Curve
> A learning curve is a plot that shows how a model's performance, often measured by accuracy, changes as the size of the training dataset increases. It helps assess if the model benefits from more data and can reveal issues like overfitting or underfitting. Cross-validation scores are often included for a more reliable estimate of performance.

Difference between Learning Curve & ROC Curve  
> Learning Curve:  
> - Shows how a model's performance changes with varying training dataset sizes.  
> - Plots training and validation (or test) error/accuracy against the size of the training dataset.  
> - Helps identify whether a model suffers from underfitting (high bias) or overfitting (high variance).
>   
> ROC Curve:  
> - Evaluates the performance of a binary classification model across different classification thresholds.  
> - Plots the true positive rate (TPR) against the false positive rate (FPR) for various threshold values.  
> - Provides insights into the trade-off between sensitivity (true positive rate) and specificity (true negative rate).  
> - The area under the ROC curve (AUC-ROC) summarizes the overall performance of the classifier.

[Back to Machine Learning Content Page](#machine)  
[Back to `Main` Content Page](#repository) 
  
### <a id = "naive"> 1️⃣ Naive Bayes</a>
 
- It is a classification algorithm that assumes all predictors are independent of one another.  
- Naive Bayes Model is a simple yet powerful machine learning algorithm used for NLP applications like text classification tasks, particularly in natural language processing (NLP). It's based on Bayes' theorem with the "naive" assumption of feature independence. Despite its simplicity, Naive Bayes often performs well in practice. In our classification, it performs moderately accurate.

### <a id = "logistic"> 2️⃣ Multinomial Logistic Regression</a>
  
- Multinomial Logistic Regression extends Logistic Regression to handle multi-class classification tasks.  
- This is done by predicting probabilities for each class and selecting the class with the highest probability as the predicted output. 

### <a id = "support"> 3️⃣ Support Vector Machine</a>
    
- SVM classification finds the best hyperplane to separate data into different classes, maximizing the margin between them.  
- It's effective for various classification tasks due to its ability to handle linear and non-linear separations through kernel functions. 

### <a id = "random"> 4️⃣ Random Forrest Classifier</a>
      
- Random Forest Classifier is an ensemble learning technique for classification tasks.  
- It builds multiple decision trees and outputs the mode of the classes predicted by individual trees.  
- It's effective, versatile, and resistant to overfitting.tions through kernel functions.   

### <a id = "bert"> 5️⃣ Bi-LSTM</a>
    
- Also known as Bidirectional Long Short-Term Memory.    
- A type of recurrent neural network (RNN) that consists of 2 LSTM layers - processing in forward and backward directions.  

## <a id = "results"> 🏆 Comparison of Results</a>
[Back to `Main` Content Page](#repository)  
  
| PARAMETER     | Naive Bayes | Logistic Regression | SVM  | Random Forest | Bi-LSTM |
|--------------|------------|---------------------|------|---------------|--------|
| Accuracy     | 0.724      | 0.815               | 0.814| 0.810         | 0.80   |
| Precision    | 0.71       | 0.82                | 0.82 | 0.82          | 0.81   |
| Recall       | 0.73       | 0.82                | 0.82 | 0.81          | 0.80   |
| F1-Score     | 0.70       | 0.82                | 0.82 | 0.81          | 0.81   |
| Support      | 11923      | 11923               | 11923| 11923         | 11904  |
| Running Time | 5–10s      | 1.5–2 min           | 20 min | 30 min      | 2 hours|

Considering both predictive performance and computational cost, `Logistic Regression` appears to be the most efficient model for this dataset. It achieves the highest overall F1-score (0.82), the highest accuracy (0.815), and comparable precision and recall to SVM, while requiring significantly less training time than SVM, Random Forest, and Bi-LSTM.

Although `SVM` achieves nearly identical overall performance, its training time is substantially longer. Random Forest and Bi-LSTM also perform competitively, but they do not provide sufficient improvement to justify their additional computational cost. Therefore, `Logistic Regression` offers the best balance between effectiveness and efficiency for this task.

| PRECISION            | Religion | Age  | Ethnicity | Gender | Other Cyberbullying | Not Cyberbullying |
|---------------------|----------|------|-----------|--------|---------------------|-------------------|
| Naive Bayes         | 0.76     | 0.64 | 0.81      | 0.79   | 0.61                | 0.66              |
| Logistic Regression | 0.94     | 0.95 | 0.97      | 0.92   | 0.57                | 0.58              |
| SVM                 | 0.96     | 0.96 | 0.97      | 0.92   | 0.55                | 0.59              |
| Random Forest       | 0.95     | 0.97 | 0.98      | 0.90   | 0.53                | 0.57              |
| Bi-LSTM             | 0.94     | 0.97 | 0.95      | 0.90   | 0.52                | 0.57              |

| RECALL               | Religion | Age  | Ethnicity | Gender | Other Cyberbullying | Not Cyberbullying |
|----------------------|----------|------|-----------|--------|---------------------|-------------------|
| Naive Bayes          | 0.97     | 0.99 | 0.91      | 0.82   | 0.35                | 0.32              |
| Logistic Regression  | 0.95     | 0.97 | 0.98      | 0.83   | 0.63                | 0.55              |
| SVM                  | 0.94     | 0.98 | 0.98      | 0.81   | 0.70                | 0.53              |
| Random Forest        | 0.95     | 0.98 | 0.98      | 0.83   | 0.67                | 0.46              |
| Bi-LSTM              | 0.95     | 0.98 | 0.98      | 0.79   | 0.66                | 0.51              |

| F1-SCORE            | Religion | Age  | Ethnicity | Gender | Other Cyberbullying | Not Cyberbullying |
|---------------------|----------|------|-----------|--------|---------------------|-------------------|
| Naive Bayes         | 0.85     | 0.78 | 0.86      | 0.81   | 0.45                | 0.43              |
| Logistic Regression | 0.95     | 0.96 | 0.97      | 0.87   | 0.60                | 0.56              |
| SVM                 | 0.95     | 0.97 | 0.97      | 0.86   | 0.62                | 0.53              |
| Random Forest       | 0.95     | 0.98 | 0.98      | 0.86   | 0.59                | 0.51              |
| Bi-LSTM             | 0.94     | 0.97 | 0.97      | 0.86   | 0.60                | 0.53              |

We can infer that the classification models are generally strong in identifying more explicit forms of cyberbullying, such as religion, age, ethnicity, and gender-based categories. This is reflected by their consistently high precision, recall, and F1-scores for these classes, indicating that such categories are both accurately predicted and reliably detected across most models.

In contrast, all models perform noticeably worse on the more ambiguous categories, namely other cyberbullying and not cyberbullying. These classes show substantially lower precision, recall, and F1-scores, suggesting that the models struggle both to distinguish them clearly and to capture all true instances. This is especially evident in Naive Bayes, which records particularly low recall and F1-scores for these classes, indicating weaker performance on subtle or context-dependent language.

Among the stronger models, Logistic Regression provides the most balanced overall performance, combining high scores on the clearer classes with relatively better consistency across the harder categories, while remaining computationally efficient. SVM and Random Forest achieve similar strengths on explicit categories and in some cases slightly better recall for other cyberbullying, but these gains are modest when compared against their higher training cost.

These findings are further supported by the confusion matrices, which show substantial misclassification between other cyberbullying and not cyberbullying. The ROC and learning curves also align with these results, indicating similar overall performance trends across models and highlighting the continued difficulty of handling nuanced and overlapping language in cyberbullying detection.

## <a id = "error-analysis">🧠 Structured Error Analysis</a>
[Back to `Main` Content Page](#repository)

While evaluation metrics such as accuracy, precision, and recall provide a high-level overview of model performance, they do not fully explain *why* models make mistakes. To address this, we perform a structured error analysis to systematically examine model behavior, identify recurring failure patterns, and uncover underlying linguistic challenges.

---

### 🔍 Error Partitioning

We first categorize predictions into:
- Correct predictions across all models  
- Incorrect predictions across all models  
- Mixed cases where models disagree  

This allows us to distinguish between:
- **Easy samples** (clear patterns)
- **Difficult samples** (ambiguous or noisy)
- **Model-dependent cases** (different model behaviors)

---

### 🔁 Misclassification Pattern Analysis

We analyze the most frequent confusion pairs (true label → predicted label) to identify systematic errors.

**Key observation:**
- The most dominant confusion occurs between `not_cyberbullying` and `other_cyberbullying`

This suggests that:
- The boundary between non-abusive and implicitly abusive content is not well-defined  
- These categories contain overlapping linguistic signals  

**Statistical Support:**
- Across all models, `other_cyberbullying` and `not_cyberbullying` consistently show:
  - Lower recall ($\approx$ 0.32–0.70)
  - Lower F1-scores ($\approx$ 0.43–0.62)
- In contrast, explicit categories (religion, age, ethnicity) achieve near-saturated performance (F1 $\approx$ 0.95+)

This suggests that:
- Errors are not random  
- They are concentrated in ambiguous class boundaries   

---

### 🧪 Qualitative Error Analysis

We examine representative misclassified tweets to understand *why* these errors occur.

#### Key Findings:
- Models rely heavily on **surface-level lexical cues** (e.g., profanity)
- Subtle or **implicit cyberbullying** is often missed
- Models struggle to identify the **target of abuse** (e.g., gender-specific)
- Short or ambiguous tweets lack sufficient context for accurate classification  

---

### 🧩 Error Categories Identified

From qualitative analysis, errors can be grouped into:

- **Lexical bias errors**  
  Misclassification due to strong words (e.g., profanity without abuse)

- **Keyword-triggered errors (Naive Bayes)**  
  Over-reliance on isolated keywords without context

- **Target ambiguity errors**  
  Difficulty identifying *who* the abuse is directed at

- **Implicit abuse errors**  
  Failure to detect subtle or indirect cyberbullying

- **Short or context-poor text errors**  
  Insufficient information for classification

- **Class boundary overlap errors**  
  Ambiguity between `not_cyberbullying` and `other_cyberbullying`

---

### ⚖️ Cross-Model Behaviour Analysis

We compare model performance using class-wise precision, recall, and F1-score.

- **Naive Bayes**
  - Performs adequately on explicit categories  
  - Very low recall and F1-score for ambiguous classes (as low as $\approx$ 0.32–0.45)  
  - Indicates strong reliance on surface-level lexical cues  

- **Logistic Regression**
  - Balanced performance across all categories  
  - Significant improvement over Naive Bayes on ambiguous classes  
  - However, performance remains moderate for these classes (F1 $\approx$ 0.55–0.60)  

- **SVM**
  - Strong and consistent performance across most categories  
  - Slightly better handling of difficult cases  
  - Still limited by ambiguity in implicit cyberbullying  

- **Random Forest**
  - High performance on explicit categories  
  - Similar behaviour to SVM with minimal additional gains  
  - Indicates diminishing returns from increased model complexity  

- **Bi-LSTM**
  - Captures contextual patterns  
  - Does not significantly outperform classical models  
  - Suggests dataset ambiguity is the primary bottleneck  

---

## <a id="statistical-analysis">📊 Statistical Analysis</a>
[Back to `Main` Content Page](#repository)

This section presents a quantitative evaluation of model performance using class wise precision, recall, F1-score, and statistical significance testing. 

### 📌 Class-wise Performance

Refer to tables [above](#results).

### 📊 Key Observations

- All models achieve strong performance on explicit categories (religion, age, ethnicity, gender), with F1-scores close to 0.95 and above  
- Performance drops significantly for:
  - `other_cyberbullying`
  - `not_cyberbullying`  
- Naive Bayes shows the weakest performance on these ambiguous classes, particularly in recall  
- Logistic Regression, SVM, Random Forest, and Bi-LSTM show clear improvements, but gains are relatively modest among them  

### 📈 Statistical Significance Testing

We conducted McNemar’s Test to compare model predictions across pairs of classifiers.

- Improvements from Naive Bayes to other models are statistically significant  
- Differences among Logistic Regression, SVM, Random Forest, and Bi-LSTM are comparatively small  

This suggests that:
- Performance improvements plateau after a certain level of model complexity  
- Remaining errors are driven more by task ambiguity than model capability 

## <a id = "challenges"> 😢 Challenges Faced</a>
[Back to `Main` Content Page](#repository)  
  
> - This was our first proper NLP project! So the learning curve was very steep, especially involving natural language proccessing w/ text data.  
> - We had to come up with unique exploratory data analysis that is relevant for our topic unlike conventional projects.  
> - We also faced a few issues in handling the installation of several of the packages used initially. We had to troubleshoot a few times at the start.  
> - Naturally working on such realistic projects that we did not have experience for results in a plethora of errors. The error handling was very time consuming.  

## <a id = "conclusion"> 🥳 Conclusion</a>
[Back to `Main` Content Page](#repository)  

The findings from both the statistical analysis and structured error analysis converge to provide a comprehensive understanding of model behaviour.

### Model Behaviour Summary

| Model | Strengths | Weaknesses |
|------|--------|----------|
| Naive Bayes | Fast, captures strong keywords | Over-relies on lexical cues, severely underperforms on ambiguous classes |
| Logistic Regression | Balanced performance, most efficient | Still struggles with implicit and ambiguous cases |
| SVM | Strong performance on difficult classes | Higher computational cost, only marginal improvement over Logistic Regression |
| Random Forest | High accuracy on explicit classes | Limited gains on ambiguous categories despite higher complexity |
| BiLSTM | Captures contextual patterns | Does not significantly outperform simpler models |

**Key Insights:**
> - Across all analyses, a consistent pattern emerges: the primary challenge lies in distinguishing between not_cyberbullying and other_cyberbullying, rather than identifying explicit categories.  
> - Improvements across models are not uniform. Even though all models perform similarly well on explicit classes (age, ethnicity, religion), performance gains from stronger models are concentrated almost entirely on the ambiguous classes.  
> - This is clearly reflected in the cross-model comparison, where correct predictions for ambiguous categories increase significantly from Naive Bayes to Logistic Regression, SVM, and other models, while performance on explicit categories remains largely saturated.

**Why the errors occurs:**  
> - While most tweets are correctly classified, a substantial portion of errors arises from ambiguity in language, where the distinction between non-abusive and implicitly abusive content is unclear.  
> - The misclassification pattern analysis and qualitative error analysis show that these errors are driven by emotionally charged but non-abusive language, as well as subtle or indirect forms of cyberbullying that lack explicit indicators.  
> - These cases do not contain strong lexical signals, making them inherently difficult for models that rely primarily on surface-level textual features.

**What the model comparison tells us:**  
> - Although Logistic Regression, SVM, Random Forest, and Bi-LSTM significantly outperform Naive Bayes, the performance gap among these stronger models remains relatively small.  
> - This suggests that increasing model complexity yields diminishing returns, as improvements are mainly limited to ambiguous classes rather than across all categories.  
> - Even the best-performing models fail to achieve strong performance on these classes, indicating that the limitation is not purely due to model choice.  
> - These observations highlight that improvements across models are not uniform, but concentrated on specific challenging cases.

**What this means:**  
> - The main limitation lies in the nature of the task itself.  
> - Cyberbullying detection requires understanding context, intent, and nuance. There are factors that are difficult to capture using surface-level textual features.  
> - This suggests that the challenge is not solely a modeling problem, but also a task formulation issue, where class definitions and boundaries may be inherently ambiguous.

### Potential Future Directions

- Improving performance may require more context-aware approaches, such as transformer-based models, as well as clearer class definitions or additional contextual information.  
- Incorporating richer contextual signals (e.g., conversational context or user intent) may help reduce ambiguity between labels that overlap each other.  
- More importantly, addressing dataset ambiguity through refined labeling guidelines or hierarchical class structures may be necessary to achieve substantial performance improvements.