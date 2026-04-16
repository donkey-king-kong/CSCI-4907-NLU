# 1. Introduction
## 1.1 Overview of Task & Research Questions

Our project is a **single-label multi-class text classification for cyberbullying detection in tweets**. For a given tweet, the goal is to assign it exactly one category in the `cyberbullying_type` column. The labels for cyberbullying can be categorized by age, gender, religion, ethnicity, other cyberbullying or not cyberbullying. 

Unlike traditional hate speech detection, which is typically framed as a binary (yes/no) classification task, our project would involve classifying each tweet into one of several closely related categories. This makes the task more challenging, as the classes often overlap in meaning. Additionally, the tweets are short, informal, and lacks context, which increases ambiguity. This therefore makes it harder for models to be able to accurately distinguish between categories.

As part of our project, our team came up with the following questions in order to guide the direction of our project.

1. How do classical machine learning models compare against a deep learning model (Bi-LSTM) for multi-class cyberbullying detection?
2. Which cyberbullying categories are the most difficult to classify accurately?
3. What linguistic patterns and dataset characteristics contribute to model errors?

---