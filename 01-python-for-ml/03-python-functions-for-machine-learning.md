# 03 — Python Functions for Machine Learning

## Why This Lesson Exists

So far, I have learned how to store values with variables, how to group many values with lists, how to store named information with dictionaries, and how to repeat actions with loops. These tools are already useful, but there is one problem: if I keep writing the same code again and again, my project will become messy.

This is where functions become important.

A function is a reusable block of code. It lets me write a piece of logic once and use it many times. In Machine Learning, this is extremely useful because ML workflows are full of repeated steps: loading data, cleaning values, calculating metrics, training models, evaluating models, and saving results.

So this lesson is not only about Python syntax. It is about learning how to organize repeated logic clearly.

---

## 1. The Big Idea

A function is like a small machine.

It can take an input, do something with it, and return an output.

```text
input -> function -> output
```

For example, if I give a function a list of numbers, it can return the mean of those numbers.

Mathematically, the mean is:

$$
\bar{x} = \frac{1}{n}\sum_{i=1}^{n}x_i
$$

In Python, I can write this idea as a function:

```python
def calculate_mean(values):
    return sum(values) / len(values)
```

Now I can use it many times:

```python
scores = [0.72, 0.75, 0.80]
mean_score = calculate_mean(scores)

print(mean_score)
```

The important idea is this: functions help me turn repeated thinking into reusable code.

---

## 2. What is a Function?

A function is a named block of code that performs a specific task.

The basic structure is:

```python
def function_name(parameters):
    # code
    return result
```

Here:

```text
def             -> tells Python that I am defining a function
function_name   -> the name of the function
parameters      -> inputs the function receives
return          -> the output the function gives back
```

For example:

```python
def greet(name):
    return "Hello, " + name
```

If I call the function:

```python
message = greet("Milana")
print(message)
```

The output is:

```text
Hello, Milana
```

This is a simple example, but the pattern is the same in Machine Learning. A function receives something, processes it, and returns something useful.

---

## 3. Why Functions Matter in Machine Learning

Machine Learning code often has repeated steps. If I do not use functions, I may copy the same code again and again. This makes the project harder to read, harder to debug, and harder to improve.

For example, imagine I want to calculate accuracy for different model predictions. Without a function, I might write the same loop many times.

With a function, I can write the logic once:

```python
def calculate_accuracy(true_labels, predicted_labels):
    correct = 0

    for i in range(len(true_labels)):
        if true_labels[i] == predicted_labels[i]:
            correct = correct + 1

    accuracy = correct / len(true_labels)
    return accuracy
```

Now I can reuse it:

```python
true_labels = [1, 0, 1, 1, 0]
model_a_predictions = [1, 0, 0, 1, 0]
model_b_predictions = [1, 0, 1, 1, 1]

accuracy_a = calculate_accuracy(true_labels, model_a_predictions)
accuracy_b = calculate_accuracy(true_labels, model_b_predictions)

print(accuracy_a)
print(accuracy_b)
```

This is already close to a real ML workflow: compare models using the same metric.

---

## 4. Parameters and Arguments

The words parameter and argument are related, but they are not exactly the same.

A parameter is the variable name used when defining the function.

```python
def calculate_mean(values):
    return sum(values) / len(values)
```

Here, `values` is a parameter.

An argument is the actual value passed into the function when calling it.

```python
scores = [0.72, 0.75, 0.80]
calculate_mean(scores)
```

Here, `scores` is the argument.

In simple words:

```text
parameter -> placeholder in the function definition
argument  -> real value given to the function
```

This distinction is useful because ML functions often receive many inputs, such as features, labels, model settings, or predictions.

---

## 5. Return Values

A function can either print something or return something. These are not the same.

This function prints the result:

```python
def show_accuracy(accuracy):
    print("Accuracy:", accuracy)
```

This function returns the result:

```python
def calculate_accuracy_score(correct, total):
    return correct / total
```

The difference is important.

If a function only prints a value, I can see the value, but I cannot easily reuse it later. If a function returns a value, I can store it in a variable and use it in more calculations.

```python
accuracy = calculate_accuracy_score(8, 10)

if accuracy > 0.80:
    print("Good result")
```

In Machine Learning, returning values is usually more useful because I often need to store metrics, compare models, or save experiment results.

---

## 6. A Function for Mean Squared Error

Mean Squared Error is a common regression metric. It measures the average squared difference between true values and predicted values.

The formula is:

$$
\mathrm{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

I can implement this formula using a function:

```python
def calculate_mse(y_true, y_pred):
    total_error = 0

    for i in range(len(y_true)):
        error = y_true[i] - y_pred[i]
        squared_error = error ** 2
        total_error = total_error + squared_error

    mse = total_error / len(y_true)
    return mse
```

Then I can test it:

```python
y_true = [3, 5, 2, 7]
y_pred = [2.5, 5.5, 2, 8]

mse = calculate_mse(y_true, y_pred)

print(mse)
```

This example is important because it connects four things:

```text
ML concept -> formula -> loop -> function
```

That is exactly how I want to learn.

---

## 7. Functions Make Experiments Cleaner

Suppose I have multiple prediction lists and want to compare them.

```python
true_labels = [1, 0, 1, 1, 0]

model_a = [1, 0, 0, 1, 0]
model_b = [1, 0, 1, 1, 1]
model_c = [1, 0, 1, 1, 0]
```

Instead of rewriting the accuracy logic three times, I can use one function.

```python
def calculate_accuracy(true_labels, predicted_labels):
    correct = 0

    for i in range(len(true_labels)):
        if true_labels[i] == predicted_labels[i]:
            correct = correct + 1

    return correct / len(true_labels)
```

Then I can write:

```python
print(calculate_accuracy(true_labels, model_a))
print(calculate_accuracy(true_labels, model_b))
print(calculate_accuracy(true_labels, model_c))
```

This makes the experiment shorter, cleaner, and easier to understand.

---

## 8. Default Parameters

A default parameter gives a function a value to use if no argument is provided.

```python
def describe_model(model_name, metric="accuracy"):
    return model_name + " will be evaluated using " + metric
```

Now I can call it in two ways:

```python
print(describe_model("KNN"))
print(describe_model("Linear Regression", "RMSE"))
```

The output is:

```text
KNN will be evaluated using accuracy
Linear Regression will be evaluated using RMSE
```

This is useful in Machine Learning because many functions have default settings. For example, a model may use a default number of neighbors, a default learning rate, or a default metric.

---

## 9. Keyword Arguments

Keyword arguments make function calls clearer.

```python
def train_model(model_name, epochs, learning_rate):
    print("Model:", model_name)
    print("Epochs:", epochs)
    print("Learning rate:", learning_rate)
```

I can call it like this:

```python
train_model(model_name="Neural Network", epochs=50, learning_rate=0.01)
```

This is clearer than:

```python
train_model("Neural Network", 50, 0.01)
```

Both can work, but keyword arguments make the meaning more obvious. In ML code, clarity matters because functions often have many parameters.

---

## 10. Functions and Dictionaries Together

Functions become even more useful when combined with dictionaries.

```python
def summarize_experiment(experiment):
    print("Model:", experiment["model"])
    print("Accuracy:", experiment["accuracy"])
```

Now I can pass a dictionary:

```python
experiment = {
    "model": "Random Forest",
    "accuracy": 0.89
}

summarize_experiment(experiment)
```

This is useful because experiment results are often stored as dictionaries. A function can receive that dictionary and format, analyze, or compare the result.

---

## 11. Returning a Dictionary

A function can return a dictionary too.

```python
def create_experiment_result(model_name, accuracy):
    result = {
        "model": model_name,
        "accuracy": accuracy
    }

    return result
```

Now I can write:

```python
result = create_experiment_result("Logistic Regression", 0.86)

print(result)
```

The output is:

```text
{'model': 'Logistic Regression', 'accuracy': 0.86}
```

This pattern is useful for experiment tracking. A function can train a model, evaluate it, and return a dictionary of results.

---

## 12. Common Mistakes

One common mistake is forgetting to return a value. If a function calculates something but does not return it, the result may be lost.

```python
def calculate_mean(values):
    mean = sum(values) / len(values)
```

This function calculates the mean, but it does not return it. If I call it, Python returns `None`.

The correct version is:

```python
def calculate_mean(values):
    mean = sum(values) / len(values)
    return mean
```

Another mistake is writing functions that do too many things. A good function should usually have one clear responsibility. For example, `calculate_accuracy()` should calculate accuracy. It should not also load data, train a model, save a file, and draw a plot.

A third mistake is using unclear function names. A name like `do_stuff()` does not help. A name like `calculate_accuracy()` is much clearer.

---

## 13. What Makes a Good Function?

A good function should be:

```text
clear
small
reusable
easy to test
easy to explain
```

For example, this name is unclear:

```python
def process(x):
    return sum(x) / len(x)
```

This name is better:

```python
def calculate_mean(values):
    return sum(values) / len(values)
```

The second version tells me what the function does and what kind of input it expects.

In Machine Learning, this matters because clean functions make experiments easier to trust.

---

## 14. A Small ML-Style Workflow

Now I can combine everything into a small workflow.

```python
def calculate_accuracy(true_labels, predicted_labels):
    correct = 0

    for i in range(len(true_labels)):
        if true_labels[i] == predicted_labels[i]:
            correct = correct + 1

    return correct / len(true_labels)


def create_result(model_name, accuracy):
    return {
        "model": model_name,
        "accuracy": accuracy
    }


true_labels = [1, 0, 1, 1, 0]

model_a_predictions = [1, 0, 0, 1, 0]
model_b_predictions = [1, 0, 1, 1, 0]

accuracy_a = calculate_accuracy(true_labels, model_a_predictions)
accuracy_b = calculate_accuracy(true_labels, model_b_predictions)

result_a = create_result("Model A", accuracy_a)
result_b = create_result("Model B", accuracy_b)

print(result_a)
print(result_b)
```

This is not a full ML model yet, but it has the shape of ML work:

```text
predictions -> metric function -> result dictionary -> comparison
```

That shape will appear again in Scikit-learn, PyTorch, and real projects.

---

## 15. What I Learned From This Lesson

Functions help me organize logic. Instead of repeating code, I can write a function once and reuse it many times.

In Machine Learning, functions are useful for metrics, preprocessing, training, evaluation, experiment tracking, and reporting results. A function is not only a programming tool. It is a way to make my thinking reusable.

This lesson also shows that formulas and functions can be connected. A mathematical formula describes what should happen. A function turns that idea into code.

---

## Mini Exercise

Create a file called `04-python-functions-ml-metrics.py` inside the `code` folder.

Write three functions:

```text
1. calculate_mean(values)
2. calculate_accuracy(true_labels, predicted_labels)
3. calculate_mse(y_true, y_pred)
```

Then test them with small example lists.

Run the file from PowerShell:

```powershell
python code\04-python-functions-ml-metrics.py
```

---

## Final Reflection

Functions are one of the first tools that make code feel professional. Without functions, my code becomes a long list of repeated instructions. With functions, I can name ideas, reuse logic, and build cleaner experiments.

For Machine Learning, this is essential. Models may be complex later, but the foundation is simple: clear inputs, clear logic, clear outputs.
