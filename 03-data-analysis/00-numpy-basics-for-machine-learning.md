# 00 — NumPy Basics for Machine Learning

## Why This Lesson Exists

Before I can train real Machine Learning models, I need to become comfortable with numerical data. Machine Learning is built on numbers. Images become numbers. Text becomes numbers. Audio becomes numbers. Sensor readings are already numbers. Tables are numbers mixed with categories. Even a neural network is mostly a large system of numerical operations.

This is where **NumPy** becomes important.

NumPy stands for **Numerical Python**. It is one of the most important libraries in the Python data science ecosystem because it gives Python a powerful way to work with arrays, vectors, matrices, and numerical operations. Many later libraries, including Pandas, Scikit-learn, SciPy, and parts of the deep learning ecosystem, are connected to the same numerical way of thinking.

This lesson is not only about learning NumPy syntax. It is about learning a new way to think about data: not as separate values, but as structured numerical objects.

---

## 1. The Big Idea

In pure Python, I can store numbers in a list:

```python
numbers = [10, 20, 30, 40, 50]
```

This is useful, but it is not ideal for numerical computing. If I want to multiply every value by 2, I might write a loop:

```python
doubled = []

for number in numbers:
    doubled.append(number * 2)

print(doubled)
```

This works, but it is not the style used in most Machine Learning code.

With NumPy, I can write:

```python
import numpy as np

numbers = np.array([10, 20, 30, 40, 50])

doubled = numbers * 2

print(doubled)
```

The result is:

```text
[ 20  40  60  80 100]
```

This is the first big idea of NumPy:

```text
Instead of looping manually over values, I can perform operations on whole arrays.
```

This style is called **vectorized computation**. It is shorter, clearer, and usually much faster than writing manual Python loops.

---

## 2. Why NumPy Matters in Machine Learning

Machine Learning models do not understand raw human concepts. They understand numerical representations.

For example:

```text
house price prediction:
features -> size, number of rooms, distance to center

image classification:
features -> pixel values

text classification:
features -> word counts, TF-IDF values, embeddings

seismic signal classification:
features -> amplitude, frequency, energy, statistical summaries
```

All of these features eventually become arrays of numbers.

A single data point can be represented as a vector:

$$
x = [x_1, x_2, x_3, \dots, x_d]
$$

A dataset with many data points can be represented as a matrix:

$$
X =
\begin{bmatrix}
x_{11} & x_{12} & x_{13} \\
x_{21} & x_{22} & x_{23} \\
x_{31} & x_{32} & x_{33}
\end{bmatrix}
$$

In Machine Learning, this matrix usually has the shape:

```text
number of samples x number of features
```

For example, if I have 1000 houses and 3 features for each house, then the dataset shape is:

```text
1000 x 3
```

NumPy helps me create, inspect, transform, and calculate with these numerical structures.

---

## 3. Importing NumPy

The standard way to import NumPy is:

```python
import numpy as np
```

The name `np` is a convention. It is not required, but almost everyone uses it. This makes NumPy code easier to read because people immediately understand what `np` means.

For example:

```python
import numpy as np

array = np.array([1, 2, 3])
print(array)
```

Output:

```text
[1 2 3]
```

From now on, I will use `np` whenever I use NumPy.

---

## 4. What is a NumPy Array?

A NumPy array is a structured collection of values. It looks similar to a Python list, but it is designed for numerical computation.

```python
import numpy as np

values = np.array([1, 2, 3, 4, 5])

print(values)
print(type(values))
```

Output:

```text
[1 2 3 4 5]
<class 'numpy.ndarray'>
```

The type is `numpy.ndarray`. The name `ndarray` means **n-dimensional array**.

This means NumPy arrays can represent:

```text
1D array -> vector
2D array -> matrix
3D array -> stack of matrices
higher-dimensional arrays -> tensors
```

Deep Learning frameworks like PyTorch and TensorFlow use similar ideas, although they use their own tensor objects.

---

## 5. Python List vs NumPy Array

A Python list is flexible. It can store different types of values:

```python
mixed_list = [1, "hello", 3.5, True]
```

A NumPy array is more strict. It is usually designed to store values of the same type.

```python
numbers = np.array([1, 2, 3, 4])
```

This strictness is one reason NumPy can be efficient. Numerical computation is easier when the data has a consistent type.

The difference becomes clear when doing operations.

Python list:

```python
numbers = [1, 2, 3]

print(numbers * 2)
```

Output:

```text
[1, 2, 3, 1, 2, 3]
```

NumPy array:

```python
import numpy as np

numbers = np.array([1, 2, 3])

print(numbers * 2)
```

Output:

```text
[2 4 6]
```

The Python list repeats the list. The NumPy array multiplies each value.

This is a very important difference.

---

## 6. Array Shapes

The shape of an array tells me its dimensions.

```python
import numpy as np

features = np.array([10, 20, 30])

print(features.shape)
```

Output:

```text
(3,)
```

This is a 1D array with 3 values.

Now let me create a 2D array:

```python
dataset = np.array([
    [1.70, 65],
    [1.80, 80],
    [1.60, 55]
])

print(dataset.shape)
```

Output:

```text
(3, 2)
```

This means:

```text
3 rows
2 columns
```

In Machine Learning language:

```text
3 samples
2 features
```

Shape is one of the most important ideas in ML coding. Many errors happen because the shape is not what the model expects.

---

## 7. Thinking in Samples and Features

A typical ML dataset matrix is written as:

$$
X \in \mathbb{R}^{n \times d}
$$

This means:

```text
X is a matrix with n rows and d columns
```

Where:

```text
n -> number of samples
d -> number of features
```

For example:

```python
X = np.array([
    [170, 65],
    [180, 80],
    [160, 55],
    [175, 70]
])
```

Here:

```text
n = 4 samples
d = 2 features
```

The shape is:

```python
print(X.shape)
```

Output:

```text
(4, 2)
```

This is the first ML mental model for NumPy:

```text
Rows are examples.
Columns are features.
```

This idea will appear again in Pandas, Scikit-learn, and deep learning.

---

## 8. Creating Arrays

NumPy provides many ways to create arrays.

### From a Python list

```python
import numpy as np

arr = np.array([1, 2, 3, 4])
```

### Zeros

```python
zeros = np.zeros(5)
print(zeros)
```

Output:

```text
[0. 0. 0. 0. 0.]
```

### Ones

```python
ones = np.ones(5)
print(ones)
```

Output:

```text
[1. 1. 1. 1. 1.]
```

### A range of numbers

```python
values = np.arange(0, 10, 2)
print(values)
```

Output:

```text
[0 2 4 6 8]
```

### Evenly spaced values

```python
values = np.linspace(0, 1, 5)
print(values)
```

Output:

```text
[0.   0.25 0.5  0.75 1.  ]
```

This is useful when creating numerical grids, plotting functions, or testing formulas.

---

## 9. Array Data Types

NumPy arrays have data types.

```python
arr = np.array([1, 2, 3])

print(arr.dtype)
```

Output:

```text
int64
```

If I use decimals:

```python
arr = np.array([1.0, 2.0, 3.0])

print(arr.dtype)
```

Output may be:

```text
float64
```

Data types matter in Machine Learning because numerical precision, memory usage, and library compatibility can depend on them.

For example, deep learning often uses floating point numbers such as `float32`.

---

## 10. Elementwise Operations

Elementwise operations apply to each element of an array.

```python
import numpy as np

x = np.array([1, 2, 3])

print(x + 10)
print(x * 2)
print(x ** 2)
```

Output:

```text
[11 12 13]
[2 4 6]
[1 4 9]
```

This is one reason NumPy feels powerful. I do not need to manually loop over each number.

Mathematically, if:

$$
x = [1, 2, 3]
$$

Then:

$$
x^2 = [1^2, 2^2, 3^2] = [1, 4, 9]
$$

NumPy lets me express this directly:

```python
x_squared = x ** 2
```

---

## 11. Vectorized Mean Squared Error

Earlier, I wrote MSE using loops. With NumPy, the code becomes shorter.

The formula is:

$$
\mathrm{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

Loop version:

```python
y_true = [3, 5, 2, 7]
y_pred = [2.5, 5.5, 2, 8]

total_error = 0

for i in range(len(y_true)):
    error = y_true[i] - y_pred[i]
    total_error = total_error + error ** 2

mse = total_error / len(y_true)
print(mse)
```

NumPy version:

```python
import numpy as np

y_true = np.array([3, 5, 2, 7])
y_pred = np.array([2.5, 5.5, 2, 8])

errors = y_true - y_pred
mse = np.mean(errors ** 2)

print(mse)
```

This is cleaner because the code looks closer to the formula:

```text
subtract -> square -> average
```

This is one of the main reasons NumPy is important for ML.

---

## 12. Indexing NumPy Arrays

Indexing a 1D NumPy array is similar to indexing a Python list.

```python
arr = np.array([10, 20, 30, 40])

print(arr[0])
print(arr[-1])
```

Output:

```text
10
40
```

For 2D arrays, I use row and column indices.

```python
X = np.array([
    [170, 65],
    [180, 80],
    [160, 55]
])

print(X[0, 0])
print(X[0, 1])
```

Output:

```text
170
65
```

The syntax is:

```text
array[row_index, column_index]
```

This is important because datasets are usually 2D.

---

## 13. Selecting Rows and Columns

Suppose I have this dataset:

```python
X = np.array([
    [170, 65],
    [180, 80],
    [160, 55],
    [175, 70]
])
```

If I want the first row:

```python
print(X[0])
```

Output:

```text
[170  65]
```

If I want the first column:

```python
print(X[:, 0])
```

Output:

```text
[170 180 160 175]
```

The `:` means "take all rows."

So:

```text
X[:, 0] -> all rows, column 0
X[:, 1] -> all rows, column 1
```

This is very common in ML because sometimes I need one feature column from a dataset.

---

## 14. Reshaping Arrays

Reshaping means changing the shape of an array without changing the data.

```python
arr = np.array([1, 2, 3, 4, 5, 6])

reshaped = arr.reshape(2, 3)

print(reshaped)
```

Output:

```text
[[1 2 3]
 [4 5 6]]
```

The original shape was:

```text
(6,)
```

The new shape is:

```text
(2, 3)
```

The number of elements must match. Six values can become a `2 x 3` matrix or a `3 x 2` matrix, but not a `4 x 4` matrix.

In Machine Learning, reshaping appears often when models expect inputs in a specific shape.

---

## 15. Broadcasting

Broadcasting is a NumPy feature that allows operations between arrays with different shapes, when the shapes are compatible.

Simple example:

```python
x = np.array([1, 2, 3])

print(x + 10)
```

Output:

```text
[11 12 13]
```

Here, NumPy treats `10` as if it can be added to every element.

A more ML-like example:

```python
X = np.array([
    [170, 65],
    [180, 80],
    [160, 55]
])

feature_means = np.array([170, 66.67])

centered = X - feature_means

print(centered)
```

This subtracts the feature means from every row.

The idea is:

```text
each row - feature_means
```

Broadcasting is powerful, but it can also cause confusing bugs if I do not understand shapes.

---

## 16. Axis: One of the Most Important NumPy Ideas

The word `axis` tells NumPy the direction of an operation.

For a 2D array:

```text
axis=0 -> operate down the rows, column by column
axis=1 -> operate across the columns, row by row
```

Example:

```python
X = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])
```

Mean of each column:

```python
print(np.mean(X, axis=0))
```

Output:

```text
[3. 4.]
```

Mean of each row:

```python
print(np.mean(X, axis=1))
```

Output:

```text
[1.5 3.5 5.5]
```

This is very important in ML.

If rows are samples and columns are features, then:

```text
np.mean(X, axis=0)
```

calculates the mean of each feature.

---

## 17. Feature Scaling with NumPy

Many ML algorithms work better when features are on similar scales.

A common scaling method is standardization:

$$
z = \frac{x - \mu}{\sigma}
$$

Where:

```text
x -> original value
mu -> mean
sigma -> standard deviation
z -> standardized value
```

With NumPy:

```python
X = np.array([
    [170, 65],
    [180, 80],
    [160, 55],
    [175, 70]
])

mean = np.mean(X, axis=0)
std = np.std(X, axis=0)

X_scaled = (X - mean) / std

print(X_scaled)
```

This is a huge ML connection. Later, Scikit-learn has tools like `StandardScaler`, but the mathematical idea is already visible here.

NumPy helps me understand what scaling actually does.

---

## 18. Dot Product

The dot product is one of the most important operations in Machine Learning.

For two vectors:

$$
a = [a_1, a_2, a_3]
$$

$$
b = [b_1, b_2, b_3]
$$

The dot product is:

$$
a \cdot b = a_1b_1 + a_2b_2 + a_3b_3
$$

In NumPy:

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

dot_product = np.dot(a, b)

print(dot_product)
```

Output:

```text
32
```

Because:

$$
1 \cdot 4 + 2 \cdot 5 + 3 \cdot 6 = 32
$$

The dot product appears in linear regression, logistic regression, neural networks, embeddings, attention, and many other ML ideas.

---

## 19. Linear Model Prediction with NumPy

A simple linear model can be written as:

$$
\hat{y} = Xw + b
$$

Where:

```text
X -> feature matrix
w -> weight vector
b -> bias
y_hat -> prediction
```

In NumPy:

```python
X = np.array([
    [50, 2],
    [80, 3],
    [120, 4]
])

w = np.array([0.3, 10])
b = 5

y_pred = X @ w + b

print(y_pred)
```

The `@` symbol performs matrix multiplication.

This is one of the first moments where NumPy starts to feel like real Machine Learning. The formula and code look very close.

---

## 20. Random Numbers

Machine Learning often uses randomness.

Examples:

```text
random train/test split
random weight initialization
random sampling
random data augmentation
```

NumPy has random number tools.

```python
rng = np.random.default_rng(seed=42)

random_values = rng.random(5)

print(random_values)
```

The seed makes the randomness reproducible. If I use the same seed, I get the same random values again.

Reproducibility matters in ML because I want experiments to be easier to debug and compare.

---

## 21. Common Mistakes

One common mistake is confusing Python lists with NumPy arrays. A list and an array may look similar, but operations behave differently.

Another mistake is ignoring shape. Many ML errors are shape errors. If a model expects `(100, 3)` but receives `(3, 100)`, the result may fail or become logically wrong.

A third mistake is misunderstanding `axis`. When using `np.mean`, `np.sum`, or `np.std`, I should always ask: am I calculating across rows or down columns?

A fourth mistake is using loops everywhere. Loops are useful, but NumPy is designed for vectorized operations. When possible, I should use array operations.

---

## 22. What I Learned From This Lesson

NumPy is the bridge between basic Python and Machine Learning. It helps me work with numerical data as arrays, vectors, and matrices.

The most important ideas from this lesson are:

```text
array
shape
indexing
slicing
vectorized operations
broadcasting
axis
feature scaling
dot product
matrix multiplication
random numbers
```

These ideas will appear again and again in Machine Learning. If I understand NumPy well, later topics like Pandas, Scikit-learn, regression, neural networks, embeddings, and transformers will become easier.

---

## Mini Exercise

Create a file called `06-numpy-basics-example.py` inside the `code` folder.

The script should:

```text
1. Create a feature matrix X.
2. Print its shape.
3. Calculate feature means.
4. Standardize the features.
5. Create a weight vector w and bias b.
6. Calculate predictions using y_hat = Xw + b.
7. Calculate MSE between y_true and y_hat.
```

Run the file:

```powershell
python code\06-numpy-basics-example.py
```

---

## Further Reading and Resources

### Official Documentation

- [NumPy: Absolute Basics for Beginners](https://numpy.org/doc/stable/user/absolute_beginners.html)
- [NumPy User Guide](https://numpy.org/doc/stable/user/index.html)
- [NumPy Reference](https://numpy.org/doc/stable/reference/index.html)

### Books and Longer Reading

- [Python Data Science Handbook by Jake VanderPlas](https://jakevdp.github.io/PythonDataScienceHandbook/)
- [Python for Data Analysis, 3rd Edition by Wes McKinney](https://wesmckinney.com/book/)
- [NumPy paper: The NumPy Array, A Structure for Efficient Numerical Computation](https://arxiv.org/abs/1102.1523)

### Practice Ideas

- Rewrite the MSE function using pure Python loops, then rewrite it using NumPy.
- Create a small dataset with 5 samples and 3 features.
- Calculate column means with `axis=0`.
- Standardize the dataset manually.
- Try changing the shape of arrays and observe which operations fail.

### What to Study Next

The next natural step is **Pandas**. NumPy is excellent for numerical arrays, but real datasets often come as tables with column names, missing values, mixed types, and messy structure. Pandas builds on this world and gives us tools for real data analysis.
