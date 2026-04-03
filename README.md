# CSCI-4907-NLU Project

## Team Contributors
| S/N | Team Members | Part |
| :-: | :- | :- |
| 1 | Zhan You Lau | Data Preparation & Cleaning, Data Visualization |
| 2 | Yu Chen Law | Data Preparation & Cleaning, Machine Learning Models |
| 3 | Kieran E Kai Voo | Machine Learning Models |
| 4 | Joshua, Tse Ern Foo | Data Visualization, Machine Learning Models |

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
  
<img width="794" alt="comparison1" src="">  

<img width="784" alt="Comparison2" src="">  