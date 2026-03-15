# CSCI-4907-NLU Project

## 1. Introduction
### Motivation
> Social media platforms have become ubiquitous in our everyday communication as it allows us to share information, ideas and opinions. However, with its widespread use it has also resulted in an increased prevalence of online harassment like cyberbullying.
>
> Given this, detecting cyberbullying on social media presents several challenges. One such challenge is that harmful content can often be expressed through subtle linguistic cues like sarcasm, slang or contextual ambiguity. Additionally, cyberbullying can also occur in different forms. For example, a single post may contain insults, threats and/or implicit aggression. This therefore introduces a multilabel classification challenge, where models need to learn to assign multiple categories to a single piece of text.
>
> As a result these complexity, traditional methods becomes less effective to accurately identify cyberbullying behaviour. Hence, we aim to utilize different machine learning approaches for multilabel cyberbullying detection in tweets.

### Task Definition
> Our project is a multilabel text classification task for cyberbullying detection. Given a tweet as input, the model aims to determine whether the tweet contains cyberbullying content. It then assigns labels that correspond to different cyberbullying categories in the dataset (eg: age-based, gender-based, or religion-based). Given that a single tweet may contain multiple forms of cyberbullying, the model may assign more than one label to a tweet.

### Contributions
> Our project aims to provide a systematic analysis of how different machine learning approaches perform in multilabel cyberbullying detection. Our contributions will include:
>
> - Comparative evaluation of multiple models (Naive Bayes, Logistic Regression, Support Vector Machines, Random Forest and Bi-LSTM)  
> - Evaluation using multilabel classification metrics
> - Structured error analysis to examine linguistic failure cases in cyberbullying detection
>
> Hence, rather than just evaluating the models' predictive performance, we also aim to provide insights into model behaviour and limitations when detecting cyberbullying in noisy tweets.

## 2. Related Work
> Cyberbullying detection on social media have been widely studied in natural language processing. For example, Waseem and Hovy (2016) applied logistic regression with character n-gram features to detect racist and sexist content on Twitter. This demonstrated that traditional machine learning models can also perform effectively when combined with engineered textual features. In later years, Salawu et al. (2021) introduced a large-scale multi-label dataset for cyberbullying and online abuse detection. This enabled more fine-grained categorization of abusive behaviors.
>
> In a similar vein, deep learning models have also been used to explore abusive language detection. For example, Pavlopoulos et al. (2017) used neural network models for moderation tasks. This showed that deep learning architectures can better capture contextual relationships in text. Similarly, Park and Fung (2017) investigated the use of convolutional neural networks for abusive language detection on Twitter. They subsequently reported that there were improvements over traditional bag-of-words methods. In terms of challenges, Wiegand et al. (2018) highlighted a few challenges in abusive language detection, particularly how abusive language is annotated in datasets.

### Gap Analysis
> From what we have read, these studies often focus primarily on improving detection performance or introducing new datasets. For our project, we instead focus on the comparative analysis of classical machine learning models and a neural architecture within a multilabel cyberbullying detection setting. Additionally, we would also conduct a structured error analysis to better understand model limitations when handling subtle and overlapping forms of cyberbullying in tweets.

### References
> Park, J. H., & Fung, P. (2017). **One-step and two-step classification for abusive language detection on Twitter.** In Proceedings of the First Workshop on Abusive Language Online (pp. 41–45). https://aclanthology.org/W17-3006/
>
> Pavlopoulos, J., Malakasiotis, P., & Androutsopoulos, I. (2017). **Deep learning for user comment moderation.** In Proceedings of the First Workshop on Abusive Language Online (pp. 25–35). https://aclanthology.org/W17-3004/
>
> Salawu, S., Lumsden, J., & He, Y. (2021). **A large-scale English multi-label Twitter dataset for cyberbullying and online abuse detection.** In Proceedings of the 5th Workshop on Online Abuse and Harms (pp. 58–69). https://aclanthology.org/2021.woah-1.16/
>
> Waseem, Z., & Hovy, D. (2016). **Hateful symbols or hateful people? Predictive features for hate speech detection on Twitter.** In Proceedings of NAACL-HLT 2016 (pp. 88–93). https://aclanthology.org/N16-2013/
>
> Wiegand, M., Ruppenhofer, J., & Kleinbauer, T. (2019). **Detection of abusive language: The problem of biased datasets.** In Proceedings of the 2nd Workshop on Abusive Language Online (pp. 138–148). https://aclanthology.org/N19-1060/

## 3. Data Strategy
### Source
> The dataset is obtained from Kaggle: [`Cyberbullying Classification Dataset`](https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification)

### Preprocessing
> - Converting all text to lowercase
> - Removing user mentions (e.g., @username)
> - Removing URLs and picture links
> - Removing punctuation
> - Removing stray HTML entities
> - Removing numbers
> - Removing English stopwords
> - Removing short words with length less than or equal to 2
> - Removing extra whitespace

### After Preprocessing
> - Cleaned data will be tokenized and converted into numerical representations suitable for model training. 
> 
> For classical machine learning models, TF-IDF vectorization will be used to transform the text into feature vectors.

## 4. Plans for Models
### Baseline Models
> The simplest baseline model will be `Naive Bayes`. 
>
> Other classical machine learning models:
>
> - Logistic Regression  
> - Support Vector Machine
> - Random Forest Classifier  

### Proposed Architecture
> On top of the aforementioned classical models, we will implement a `Bidirectional Long Short-Term Memory (Bi-LSTM)` neural network. 
>
> Bi-LSTM models process text sequences in both forward and backward directions. This allows the model to capture contextual information from the surrounding words in a sentence.
>
> This is especially useful for natural language processing tasks where word order and context are critical in determining meaning.

### Implementation
> Tools and libraries we plan to use:
> 
> - `scikit-learn` - For classical machine learning models and TF-IDF feature extraction
> - `PyTorch` - For implementing Bi-LSTM
> - `NLTK` - For text preprocessing tasks
> - `gensim` - For training Word2Vec embeddings
> - `imbalanced-learn (imblearn)` - For handling class imbalance through oversampling techniques
> 
> We will be using **Google Colab** to facilitate model training, evaluation and analysis.

## 5. Plans for Evaluation
### Metrics
### Validation Strategy

## 6. Team Members & Responsibilities

## 7. Project Timeline & Milestones

## 8. GitHub Repository

## References