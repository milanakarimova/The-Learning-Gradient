# 02 — Python Lists, Dictionaries, and Loops for Machine Learning

## Why This Lesson Exists

Before I can understand datasets, features, labels, model outputs, or experiment results, I need to understand three very important Python ideas: lists, dictionaries, and loops.

At first, these topics may look like basic programming concepts. But in Machine Learning, they appear everywhere. A list can store numbers, labels, predictions, losses, or accuracy values. A dictionary can store model settings, experiment results, feature meanings, or class counts. A loop can repeat an operation over many data points, many files, many features, or many experiments.

So this lesson is not only about Python syntax. It is about learning how to organize repeated information and repeated actions. That is one of the first real steps toward thinking like someone who works with data.

---

## 1. The Big Picture

Machine Learning is built around data. Data usually does not come as one single value. It comes as many values.

For example, imagine I trained a model five times and got five accuracy scores:

```python
accuracies = [0.72, 0.75, 0.78, 0.80, 0.82]
```

This is a list. It stores multiple related values together.

Now imagine I want to store information about one experiment:

```python
experiment = {
    "model": "KNN",
    "k": 5,
    "accuracy": 0.82
}
```

This is a dictionary. It stores information using names and values.

Now imagine I want to print every accuracy score one by one:

```python
for score in accuracies:
    print(score)
```

This is a loop. It repeats an action for every item.

These three ideas work together:

```text
list        -> stores many values
dictionary  -> stores named information
loop        -> repeats actions over data
```

In Machine Learning, these are not optional. They are part of the basic language of experiments.

---

## 2. Lists: Storing Many Values

A list is a Python data structure that can store multiple values in one variable.

```python
numbers = [10, 20, 30, 40, 50]
```

The variable `numbers` does not store one value. It stores a collection of values.

In simple words:

```text
list = an ordered collection of values
```

The word "ordered" means that the position of each value matters. Python remembers which value is first, second, third, and so on.

---

## 3. Why Lists Matter in Machine Learning

In ML, lists can represent many things.

```python
temperatures = [21.5, 23.0, 24.2, 22.8]
labels = ["cat", "dog", "cat", "bird"]
predictions = [1, 0, 1, 1, 0]
losses = [0.95, 0.72, 0.51, 0.39, 0.31]
```

When a model trains, we often track how the loss changes. If the loss decreases over time, it usually means the model is learning something useful.

Mathematically, a list of losses can be imagined as a sequence:

$$
L_1, L_2, L_3, \dots, L_n
$$

For example:

```python
losses = [0.95, 0.72, 0.51, 0.39, 0.31]
```

Here, `losses[0]` is the first loss value, `losses[1]` is the second loss value, and so on.

---

## 4. Indexing: Accessing Items in a List

Python uses indexing to access values inside a list. The important detail is that Python indexing starts from `0`.

```python
numbers = [10, 20, 30, 40, 50]

print(numbers[0])
print(numbers[1])
print(numbers[2])
```

The output is:

```text
10
20
30
```

This means:

```text
numbers[0] -> first item
numbers[1] -> second item
numbers[2] -> third item
```

This can feel strange at first because humans usually count from 1. But Python counts positions from 0. This matters in Machine Learning because datasets, arrays, rows, columns, tokens, and batches are often accessed by index.

---

## 5. Negative Indexing

Python also supports negative indexing. This means I can access items from the end of a list.

```python
numbers = [10, 20, 30, 40, 50]

print(numbers[-1])
print(numbers[-2])
```

The output is:

```text
50
40
```

So `numbers[-1]` means the last item, and `numbers[-2]` means the second item from the end.

This is useful when I want to quickly access the final result, such as the last loss value after training.

```python
losses = [0.95, 0.72, 0.51, 0.39, 0.31]

final_loss = losses[-1]

print(final_loss)
```

The final loss is `0.31`.

---

## 6. Slicing: Taking Part of a List

Slicing means taking a smaller part of a list.

```python
numbers = [10, 20, 30, 40, 50]

print(numbers[0:3])
```

The output is:

```text
[10, 20, 30]
```

The syntax is:

```text
list[start:end]
```

The `start` index is included, but the `end` index is not included. So `numbers[0:3]` gives index 0, 1, and 2.

In Machine Learning, slicing becomes very important when working with datasets, train/test splits, batches, and arrays.

---

## 7. Adding Items to a List

We can add a new item to a list using `.append()`.

```python
accuracies = [0.72, 0.75, 0.78]

accuracies.append(0.80)

print(accuracies)
```

The output is:

```text
[0.72, 0.75, 0.78, 0.8]
```

This is useful when collecting experiment results.

```python
results = []

results.append(0.72)
results.append(0.75)
results.append(0.78)

print(results)
```

This pattern is very common:

```text
create empty list -> run experiment -> append result
```

---

## 8. Loops: Repeating an Action

A loop lets me repeat an action multiple times.

```python
accuracies = [0.72, 0.75, 0.78]

for score in accuracies:
    print(score)
```

The output is:

```text
0.72
0.75
0.78
```

The loop reads each value from the list and temporarily stores it in the variable `score`.

In simple words:

```text
for each score in accuracies:
    do something with score
```

This is very close to how we think about data.

---

## 9. Looping with Meaningful Variable Names

Variable names inside loops matter. Good names make code easier to read.

This works, but it is not very clear:

```python
for x in accuracies:
    print(x)
```

This is better:

```python
for accuracy in accuracies:
    print(accuracy)
```

The second version tells me what the value means. In Machine Learning, clear names are important because code can become complex very quickly.

---

## 10. Calculating an Average with a Loop

Earlier, I used `sum()` and `len()` to calculate the mean. Now I can write the same idea with a loop.

```python
accuracies = [0.72, 0.75, 0.78, 0.80]

total = 0

for accuracy in accuracies:
    total = total + accuracy

average_accuracy = total / len(accuracies)

print(average_accuracy)
```

Mathematically, the average is:

$$
\bar{x} = \frac{1}{n}\sum_{i=1}^{n}x_i
$$

In code, the loop is doing the summation part:

```python
for accuracy in accuracies:
    total = total + accuracy
```

The symbol $\sum$ means "add things together." A loop is one way to make that idea happen in Python.

---

## 11. Dictionaries: Storing Named Information

A dictionary stores information as key-value pairs.

```python
experiment = {
    "model": "KNN",
    "k": 5,
    "accuracy": 0.82
}
```

In this dictionary:

```text
"model"    -> "KNN"
"k"        -> 5
"accuracy" -> 0.82
```

The left side is called the key. The right side is called the value.

In simple words:

```text
dictionary = named information
```

Dictionaries are useful because they make data more descriptive.

---

## 12. Why Dictionaries Matter in Machine Learning

Machine Learning experiments have many settings and results. A dictionary is a clean way to store them.

```python
experiment = {
    "model": "Logistic Regression",
    "learning_rate": 0.01,
    "epochs": 100,
    "accuracy": 0.88
}
```

In real ML projects, dictionaries are often used for:

```text
model parameters
training settings
evaluation results
class mappings
feature descriptions
configuration files
```

---

## 13. Accessing Values in a Dictionary

To access a value in a dictionary, I use the key.

```python
experiment = {
    "model": "KNN",
    "k": 5,
    "accuracy": 0.82
}

print(experiment["model"])
print(experiment["accuracy"])
```

The output is:

```text
KNN
0.82
```

This is useful because I can directly ask for the information I need by name.

---

## 14. Adding New Information to a Dictionary

I can add new key-value pairs to a dictionary.

```python
experiment = {
    "model": "KNN",
    "k": 5,
    "accuracy": 0.82
}

experiment["dataset"] = "toy classification data"

print(experiment)
```

This is useful when I start with a small experiment and later add more details, such as dataset name, metric, preprocessing method, or model version.

---

## 15. Looping Through a Dictionary

I can loop through a dictionary using `.items()`.

```python
experiment = {
    "model": "KNN",
    "k": 5,
    "accuracy": 0.82
}

for key, value in experiment.items():
    print(key, ":", value)
```

The output is:

```text
model : KNN
k : 5
accuracy : 0.82
```

This is useful when I want to print or inspect all settings of an experiment.

---

## 16. Lists of Dictionaries

A very important pattern in ML is a list of dictionaries.

```python
experiments = [
    {"model": "KNN", "accuracy": 0.82},
    {"model": "Logistic Regression", "accuracy": 0.86},
    {"model": "Random Forest", "accuracy": 0.89}
]
```

This structure is powerful because each dictionary stores one experiment, and the list stores all experiments.

```text
one dictionary       -> one experiment
list of dictionaries -> many experiments
```

Now I can loop through all experiments:

```python
for experiment in experiments:
    print(experiment["model"], experiment["accuracy"])
```

The output is:

```text
KNN 0.82
Logistic Regression 0.86
Random Forest 0.89
```

This is a small version of experiment tracking.

---

## 17. Finding the Best Result

Suppose I have several experiments and want to find the best model.

```python
experiments = [
    {"model": "KNN", "accuracy": 0.82},
    {"model": "Logistic Regression", "accuracy": 0.86},
    {"model": "Random Forest", "accuracy": 0.89}
]

best_experiment = experiments[0]

for experiment in experiments:
    if experiment["accuracy"] > best_experiment["accuracy"]:
        best_experiment = experiment

print(best_experiment)
```

The output is:

```text
{'model': 'Random Forest', 'accuracy': 0.89}
```

This example is important because it feels like real ML logic. I have multiple results, compare them, and choose the best one based on a metric.

---

## 18. Conditions Inside Loops

Loops become more powerful when combined with conditions.

```python
experiments = [
    {"model": "KNN", "accuracy": 0.82},
    {"model": "Logistic Regression", "accuracy": 0.86},
    {"model": "Random Forest", "accuracy": 0.89}
]

for experiment in experiments:
    if experiment["accuracy"] > 0.85:
        print(experiment["model"])
```

The output is:

```text
Logistic Regression
Random Forest
```

This pattern is common:

```text
loop through data -> check condition -> keep or print useful items
```

Later, this idea becomes filtering in Pandas, selecting good models, cleaning data, and analyzing errors.

---

## 19. A More ML-Like Example: Manual Accuracy

Imagine I have true labels and predicted labels:

```python
true_labels = [1, 0, 1, 1, 0]
predicted_labels = [1, 0, 0, 1, 0]
```

I can count how many predictions are correct.

```python
true_labels = [1, 0, 1, 1, 0]
predicted_labels = [1, 0, 0, 1, 0]

correct = 0

for i in range(len(true_labels)):
    if true_labels[i] == predicted_labels[i]:
        correct = correct + 1

accuracy = correct / len(true_labels)

print("Correct:", correct)
print("Accuracy:", accuracy)
```

The output is:

```text
Correct: 4
Accuracy: 0.8
```

This is a simplified version of accuracy.

Mathematically, accuracy can be written as:

$$
\mathrm{Accuracy} = \frac{\text{number of correct predictions}}{\text{total number of predictions}}
$$

This example shows how lists and loops can implement an ML metric.

---

## 20. The `range()` Function

The function `range()` creates a sequence of numbers.

```python
for i in range(5):
    print(i)
```

The output is:

```text
0
1
2
3
4
```

Notice that it starts at 0 and stops before 5.

This is useful when I need indices.

```python
true_labels = [1, 0, 1]
predicted_labels = [1, 1, 1]

for i in range(len(true_labels)):
    print(true_labels[i], predicted_labels[i])
```

Here, `i` is used to access matching positions from both lists.

---

## 21. Common Mistakes

One common mistake is forgetting that Python indexing starts from 0. If a list has five items, the last index is 4, not 5.

```python
numbers = [10, 20, 30, 40, 50]

print(numbers[5])
```

This will cause an error because index 5 does not exist.

Another mistake is using unclear names inside loops. A loop like `for x in y` may work, but it does not explain the meaning. If I write `for accuracy in accuracies`, the code becomes easier to understand.

A third mistake is mixing up lists and dictionaries. Lists use positions. Dictionaries use keys. If I want ordered values, I use a list. If I want named information, I use a dictionary.

---

## 22. What I Learned From This Lesson

Lists help me store many values. Dictionaries help me store named information. Loops help me repeat actions over data.

These ideas are basic, but they are not small. They are the foundation of many Machine Learning workflows. When I track losses, store experiment results, calculate accuracy, or compare models, I use these concepts.

This lesson also shows an important pattern:

```text
store data -> loop through it -> calculate result -> interpret result
```

That pattern will appear again and again in Machine Learning.

---

## Mini Exercise

Create a file called `03-python-collections-loops-example.py` inside the `code` folder.

Write code that does the following:

```text
1. Create a list of experiments.
2. Each experiment should be a dictionary.
3. Each dictionary should contain model name and accuracy.
4. Loop through the experiments.
5. Print only models with accuracy greater than 0.85.
6. Find and print the best model.
```

A possible starting point:

```python
experiments = [
    {"model": "KNN", "accuracy": 0.82},
    {"model": "Logistic Regression", "accuracy": 0.86},
    {"model": "Random Forest", "accuracy": 0.89}
]
```

Run the file from PowerShell:

```powershell
python code\03-python-collections-loops-example.py
```

---

## Final Reflection

This lesson helped me see that basic Python structures are directly connected to Machine Learning. Lists are not just lists. They can store predictions, losses, labels, or scores. Dictionaries are not just dictionaries. They can store experiments and configurations. Loops are not just repetition. They are how I process data step by step.

If I understand these tools well, I will be much more ready for NumPy, Pandas, Scikit-learn, and real ML workflows.
