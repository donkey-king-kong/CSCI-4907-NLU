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
1) [Source Code](#source)  
2) [Data Preparation & Cleaning](#data)
3) [Exploratory Analysis](#analysis)
   - [Number Game](#number)
   - [Tokenization](#token)
   - [Word Cloud](#word)
   - [Sentiment Analysis](#sentiment)
4) [Machine Learning](#machine)
   - [Naive Bayes](#naive)
   - [Multinomial Logistic Regression](#logistic)
   - [Support Vector Machine](#support)
   - [Random Forest Classifier](#random)
   - [Bi-LSTM](#bert)

## <a id="source"> 💻 Source Code </a>
Source Code on Google Colab:
> https://colab.research.google.com/drive/1KTm6iu-XBtcy4Y7H8wsIBvkC3YWAYt4G

## <a id = "data">🧼 Data Preparation & Cleaning</a>
[Back to `Main` Content Page]
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
[Back to `Main` Content Page]  

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

> - This would help us in identifying sentiments - positive, negative and neutral, from a piece of text.
    
The sentiment score represents the polarity of the text (Positive, Negative, Neutral). It is a floating point number ranging from -1.0 to +1.0. 
 
> - If the sentiment score is close to 1.0, it indicates a very positive sentiment.  
> - If the sentiment score is close to -1.0, it indicates a very negative sentiment.  
> - If the sentiment score is around 0.0, it indicates a neutral sentiment.  

## <a id = "machine">🤖 Machine Learning</a>
[Back to `Main` Content Page](#repository) 

TO BE UPDATED