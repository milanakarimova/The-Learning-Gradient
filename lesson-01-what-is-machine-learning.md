# Lesson 01: Introduction to Machine Learning

Welcome to the first lesson of **The Learning Gradient**.

In this lesson, I will explain the basic idea of Machine Learning, the main types of learning, the ML development workflow, and common mistakes beginners should avoid.

---

## 1. The Core Idea of Machine Learning

The main idea of Machine Learning is simple:

```text
Data + Algorithm + Compute → Model → Predictions
```

Machine Learning is about giving data to an algorithm so that it can learn patterns and make predictions.

Instead of writing every rule manually, we allow the model to learn from examples.

---

## 2. What is a Model?

A **model** is the result of the learning process.

The model learns patterns from data and then uses those patterns to make predictions on new data.

Example:

```text
Input: study hours
Output: exam score prediction
```

The model looks at previous examples and learns the relationship between study hours and exam scores.

---

## 3. Three Main Ways Machines Can Learn

There are three main types of Machine Learning:

1. Supervised Learning
2. Unsupervised Learning
3. Reinforcement Learning

---

## 4. Supervised Learning

**Supervised Learning** is a type of Machine Learning where the model learns from labeled training data.

Labeled data means that both the input and the correct output are given.

```text
Labeled training data + ML algorithm → Trained model
```

Example:

| Input | Output |
|---|---|
| House size | House price |
| Study hours | Exam score |
| Email text | Spam or not spam |

The model learns from these examples and tries to predict the output for new inputs.

---

## 5. Examples of Supervised Learning Algorithms

Common supervised learning algorithms:

- Linear Regression
- Decision Trees
- Support Vector Machines
- Neural Networks

Supervised learning is mostly used for:

- Regression
- Classification

---

## 6. Regression

**Regression** is used when we want to predict a continuous numerical value.

Examples:

- predicting house prices
- predicting temperature
- predicting exam scores
- predicting sales amount

Example:

```text
Input: house size
Output: predicted house price
```

---

## 7. Classification

**Classification** is used when we want to predict a category or class.

Examples:

- spam or not spam
- sick or healthy
- cat or dog
- accepted or rejected

Example:

```text
Input: email text
Output: spam / not spam
```

---

## 8. Unsupervised Learning

**Unsupervised Learning** is a type of Machine Learning where the model works with unlabeled data.

This means the model does not receive the correct answers.

The model tries to find hidden patterns or structure in the data by itself.

---

## 9. Examples of Unsupervised Learning

Common unsupervised learning tasks:

- Clustering
- Dimensionality Reduction
- Anomaly Detection

---

## 10. Clustering

**Clustering** means grouping similar data points together.

Example:

```text
Group customers based on similar behavior.
```

The model does not know the group names in advance.  
It creates groups based on similarity.

---

## 11. Dimensionality Reduction

**Dimensionality Reduction** means reducing the number of features while keeping the most important information.

It is useful when the dataset has many columns.

Example:

```text
A dataset has 100 features.
We reduce it to 2 or 3 important features for visualization or faster training.
```

---

## 12. Anomaly Detection

**Anomaly Detection** means finding unusual or abnormal data points.

Examples:

- detecting fraud transactions
- finding unusual network activity
- detecting machine failure
- finding abnormal medical results

---

## 13. Reinforcement Learning

**Reinforcement Learning** is a type of Machine Learning where an agent learns by interacting with an environment.

The agent takes actions and receives rewards or penalties.

```text
Agent → Action → Environment
Agent ← State + Reward ← Environment
```

The goal of the agent is to learn actions that maximize total reward.

Example:

```text
A robot learning how to walk.
A game AI learning how to win.
```

---

## 14. Beyond the Big Three

There are also other important learning approaches:

- Semi-Supervised Learning
- Self-Supervised Learning
- Transfer Learning

---

## 15. Semi-Supervised Learning

**Semi-Supervised Learning** uses a small amount of labeled data and a large amount of unlabeled data.

```text
Small labeled data + Large unlabeled data → Better learning
```

This is useful because labeled data is often expensive and difficult to collect.

Example:

```text
We have 100 labeled images and 10,000 unlabeled images.
The model uses both to learn better.
```

---

## 16. Self-Supervised Learning

**Self-Supervised Learning** is when the model learns from unlabeled data by creating its own labels from the data.

The model creates a learning task for itself.

Example:

```text
The model hides part of an image or sentence and tries to predict the missing part.
```

This method is widely used in modern AI systems.

---

## 17. Transfer Learning

**Transfer Learning** means taking a model that was already trained on a large dataset and fine-tuning it for a new task.

```text
Pre-trained model → Fine-tune on your task
```

Example:

```text
A model trained on millions of images can be fine-tuned to classify medical images.
```

Transfer learning saves time and usually needs less data.

---

## 18. Machine Learning Development Workflow

A typical ML project follows these steps:

```text
1. Define the problem
2. Collect and label data
3. Explore and engineer features
4. Train and evaluate models
5. Tune and improve
6. Deploy and monitor
```

---

## 19. Step 1 — Define the Problem

Before building a model, we need to clearly understand the problem.

Questions to ask:

- What are we trying to predict?
- Is it regression or classification?
- What data do we need?
- How will we measure success?

Example:

```text
Problem: Predict house prices.
Type: Regression.
Target: House price.
Features: Size, location, number of rooms.
```

---

## 20. Step 2 — Collect and Label Data

The next step is collecting data.

For supervised learning, we also need labels.

Example:

```text
Input: image of a fruit
Label: apple, banana, orange
```

Good data is very important because a model can only learn from the data we give it.

---

## 21. Step 3 — Explore and Engineer Features

Before training, we need to understand the data.

This step includes:

- checking missing values
- understanding columns
- visualizing relationships
- finding important features
- creating new useful features

Feature engineering means creating or improving input features to help the model learn better.

---

## 22. Step 4 — Train and Evaluate Models

Training means teaching the model using data.

Evaluation means checking how well the model performs.

Example evaluation questions:

- Are the predictions accurate?
- Is the model overfitting?
- Does it work well on new data?
- Which metric should we use?

---

## 23. Step 5 — Tune and Improve

After the first model, we improve it.

This can include:

- changing hyperparameters
- trying different algorithms
- improving features
- collecting more data
- using regularization
- reducing overfitting

The first model is rarely the best model.

---

## 24. Step 6 — Deploy and Monitor

Deployment means making the model available for real use.

Monitoring means checking the model after deployment.

This is important because real-world data can change over time.

Example:

```text
A model works well today, but after a few months the data changes.
So we need to monitor its performance.
```

---

## 25. Common Machine Learning Pitfalls

Some common ML mistakes are:

- Overfitting
- Data Leakage
- Biased Training Data
- Distribution Shift

---

## 26. Overfitting

**Overfitting** happens when a model learns the training data too well, including noise and unnecessary details.

It performs well on training data but poorly on new data.

Simple idea:

```text
Good on training data
Bad on test data
```

---

## 27. Data Leakage

**Data Leakage** happens when information from the test data accidentally enters the training process.

This makes the model look better than it really is.

Example:

```text
Using future information while training a model.
```

Data leakage is dangerous because it gives unrealistic results.

---

## 28. Biased Training Data

**Biased Training Data** means the dataset does not represent the real-world situation properly.

Example:

```text
A face recognition model trained mostly on one group of people may perform badly on other groups.
```

If the data is biased, the model can also become biased.

---

## 29. Distribution Shift

**Distribution Shift** happens when the data changes after the model is trained.

Example:

```text
Training data: old customer behavior
Real-world data: new customer behavior
```

The model may become less accurate because the real data is no longer similar to the training data.

---

## 30. Summary

In this lesson, I learned:

- the core idea of Machine Learning
- what a model is
- supervised learning
- unsupervised learning
- reinforcement learning
- semi-supervised learning
- self-supervised learning
- transfer learning
- the ML development workflow
- common ML pitfalls

---

## Key Takeaway

Machine Learning is not magic.

It is a process where a model learns patterns from data and uses those patterns to make predictions or decisions.

```text
Data + Algorithm + Compute → Model → Predictions
```

---

## Next Lesson

In the next lesson, I will learn more about:

```text
Features, labels, targets, training data, and test data
```
