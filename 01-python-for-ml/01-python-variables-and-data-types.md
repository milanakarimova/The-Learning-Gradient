# 01 — Python Variables and Data Types

## Why This Lesson Exists

Before I can work with datasets, models, or experiments, I need to understand how Python stores information. In Python, stored information usually begins with variables.

A variable is like a name attached to a value. In Machine Learning, variables can represent many things: a number, a label, a dataset path, a model name, a learning rate, or an evaluation score.

This lesson is small, but it is important because almost every future Python and ML example will use variables.

---

## 1. What is a Variable?

A variable is a name that stores a value. The value can be a number, text, list, or another kind of object.

For example:

```python
age = 20
name = "Milana"
accuracy = 0.87
```

Here, `age`, `name`, and `accuracy` are variable names. The values are `20`, `"Milana"`, and `0.87`.

In simple words:

```text
variable name -> stored value
```

This idea matters in Machine Learning because we constantly store values during experiments.

---

## 2. Variables in a Machine Learning Context

In normal beginner Python examples, variables may look too simple. But in Machine Learning, variables become part of the experiment.

For example:

```python
learning_rate = 0.01
epochs = 100
model_name = "Logistic Regression"
accuracy = 0.92
```

These variables describe an experiment. The `learning_rate` controls how fast a model learns. The `epochs` variable tells how many times the model sees the training data. The `model_name` stores which algorithm is being used. The `accuracy` stores the model result.

So variables are not just syntax. They help us organize our thinking.

---

## 3. Common Python Data Types

Python has different data types. A data type tells us what kind of value we are working with.

The most common beginner data types are:

| Type | Meaning | Example |
|---|---|---|
| `int` | whole number | `10` |
| `float` | decimal number | `0.95` |
| `str` | text | `"model"` |
| `bool` | true or false value | `True` |
| `list` | collection of values | `[1, 2, 3]` |
| `dict` | key-value pairs | `{"k": 3}` |

These types appear often in ML code. For example, a model accuracy is usually a `float`, a list of scores is a `list`, and model settings can be stored in a `dict`.

---

## 4. A Small Example

Suppose I trained a simple model and got these results:

```python
model_name = "KNN"
k = 5
accuracy = 0.84
is_good_result = accuracy > 0.80

print(model_name)
print(k)
print(accuracy)
print(is_good_result)
```

The output will be:

```text
KNN
5
0.84
True
```

This code is simple, but it already looks like something that could appear in a real ML experiment.

The variable `is_good_result` stores the result of a comparison. Since `0.84 > 0.80` is true, Python stores `True`.

---

## 5. Checking Data Types

Python lets us check the type of a value using `type()`.

```python
model_name = "KNN"
k = 5
accuracy = 0.84
is_good_result = accuracy > 0.80

print(type(model_name))
print(type(k))
print(type(accuracy))
print(type(is_good_result))
```

The output will be similar to this:

```text
<class 'str'>
<class 'int'>
<class 'float'>
<class 'bool'>
```

This is useful because many bugs happen when I think a value has one type, but Python sees it as another type.

---

## 6. Why Types Matter in ML

Data types matter because Machine Learning code expects values in specific forms. A model cannot train properly if numbers are stored as text. A plotting function may fail if the data is not in the expected format.

For example, this is a number:

```python
accuracy = 0.91
```

But this is text:

```python
accuracy = "0.91"
```

They look similar to a human, but Python treats them differently. The first one can be used in mathematical operations. The second one is a string, so it behaves like text.

---

## 7. Common Mistakes

A common mistake is using unclear variable names. For example, `x` and `y` are sometimes acceptable in math, but in learning notes it is often better to write clearer names like `features`, `target`, `accuracy`, or `learning_rate`.

Another mistake is mixing text and numbers. If a number is inside quotation marks, Python treats it as a string. This can cause errors during calculations.

A third mistake is changing a variable without noticing. Python allows reassignment, so the same variable name can store a new value later. This is useful, but it can also create confusion if the code is messy.

---

## 8. What I Learned From This Lesson

Variables help me store information, and data types tell me what kind of information I am storing. In Machine Learning, this matters because experiments are full of values: parameters, scores, labels, predictions, and settings.

A variable is not only a programming detail. It is a way to name and organize an idea.

---

## Mini Exercise

Create a file called `02-python-variables-example.py` inside the `code` folder and write this code:

```python
model_name = "Logistic Regression"
learning_rate = 0.01
epochs = 50
accuracy = 0.88

print("Model:", model_name)
print("Learning rate:", learning_rate)
print("Epochs:", epochs)
print("Accuracy:", accuracy)

print(type(model_name))
print(type(learning_rate))
print(type(epochs))
print(type(accuracy))
```

Run it from PowerShell:

```powershell
python code\02-python-variables-example.py
```

If Python prints the values and their types, the exercise is complete.

---

## Final Reflection

This lesson is small, but it gives me one of the first building blocks of Python. If I can name values clearly, I can organize code clearly. If I can organize code clearly, I can understand experiments more easily.
