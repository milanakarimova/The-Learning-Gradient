# 01 — Pandas Basics for Machine Learning

## Why This Lesson Exists

After learning NumPy, I can work with numerical arrays, vectors, matrices, and mathematical operations. That is a strong foundation, but real-world data usually does not arrive as a clean NumPy matrix. Real data often comes as tables with column names, missing values, mixed data types, categories, dates, repeated values, strange formatting, and human mistakes.

This is where **Pandas** becomes important.

Pandas is a Python library for working with structured data. Structured data usually means data that looks like a table: rows and columns. In Machine Learning, most beginner and intermediate projects start from this kind of data. A CSV file, an Excel file, a database table, or a spreadsheet can usually be loaded and explored with Pandas.

This lesson is not only about syntax. It is about learning how to look at data before modeling it. A model should not be trained blindly. Before I choose an algorithm, I need to understand what the data looks like, what each column means, whether values are missing, whether the target is balanced, and whether the dataset makes sense.

In simple words:

```text
Pandas helps me turn raw tables into understandable data.
```

---

## 1. The Big Picture

A Machine Learning workflow usually starts before the model.

A simplified workflow looks like this:

```text
raw data -> inspect data -> clean data -> prepare features -> train model -> evaluate model
```

Pandas is mostly used in the early and middle parts:

```text
inspect data
clean data
select columns
create features
summarize patterns
prepare tables for modeling
```

For example, imagine I have a small dataset about students:

```text
name,study_hours,attendance,passed
Aylin,5,90,1
Rauf,2,60,0
Leyla,4,85,1
```

A Machine Learning model does not understand this table directly as a human story. But with Pandas, I can load it, inspect it, summarize it, clean it, and eventually convert the useful columns into numerical features.

---

## 2. Importing Pandas

The standard way to import Pandas is:

```python
import pandas as pd
```

The name `pd` is a convention. Almost everyone uses it, so I will use it too.

```python
import pandas as pd
```

From now on, whenever I write `pd`, it means Pandas.

---

## 3. What is a DataFrame?

The most important Pandas object is the **DataFrame**.

A DataFrame is a table with rows and columns.

```python
import pandas as pd

data = {
    "name": ["Aylin", "Rauf", "Leyla"],
    "study_hours": [5, 2, 4],
    "attendance": [90, 60, 85],
    "passed": [1, 0, 1]
}

df = pd.DataFrame(data)

print(df)
```

Output:

```text
    name  study_hours  attendance  passed
0  Aylin            5          90       1
1   Rauf            2          60       0
2  Leyla            4          85       1
```

The left side is the index. The column names are at the top. Each row represents one observation.

In Machine Learning language:

```text
row    -> one sample / one observation
column -> one variable / one feature or target
```

This is similar to the NumPy idea:

$$
X \in \mathbb{R}^{n \times d}
$$

But Pandas adds labels to rows and columns, which makes data easier to understand.

---

## 4. DataFrame vs NumPy Array

A NumPy array is excellent for numerical computation. A Pandas DataFrame is excellent for labeled tabular data.

NumPy:

```text
fast numerical arrays
mostly same data type
great for math operations
```

Pandas:

```text
tables with column names
mixed data types
missing values
data cleaning
grouping and summarizing
```

For Machine Learning, I often use both.

A common workflow is:

```text
Pandas for reading and cleaning data
NumPy for numerical operations
Scikit-learn for modeling
```

Later, when training models, many libraries will accept either Pandas DataFrames or NumPy arrays. But before modeling, Pandas is often more comfortable because the column names help me understand the data.

---

## 5. Creating a DataFrame

I can create a DataFrame from a dictionary.

```python
import pandas as pd

data = {
    "model": ["KNN", "Logistic Regression", "Random Forest"],
    "accuracy": [0.82, 0.86, 0.89],
    "training_time": [0.2, 0.5, 1.4]
}

df = pd.DataFrame(data)

print(df)
```

Output:

```text
                 model  accuracy  training_time
0                  KNN      0.82            0.2
1  Logistic Regression      0.86            0.5
2        Random Forest      0.89            1.4
```

This already looks like an experiment table. Each row is one experiment, and each column stores information about that experiment.

This is one reason Pandas is useful in ML: it helps me organize results, not only datasets.

---

## 6. Reading CSV Files

Most real datasets are stored in files. A common format is CSV, which means Comma-Separated Values.

A CSV file may look like this:

```text
name,study_hours,attendance,passed
Aylin,5,90,1
Rauf,2,60,0
Leyla,4,85,1
```

Pandas can read it using:

```python
df = pd.read_csv("data/students.csv")
```

Then I can inspect the first rows:

```python
print(df.head())
```

This is one of the most common first lines in a data analysis project.

In a real project, I should not immediately train a model after loading the file. I should first inspect the data carefully.

---

## 7. Inspecting the First Rows

The method `.head()` shows the first five rows by default.

```python
df.head()
```

I can also choose the number of rows:

```python
df.head(10)
```

This helps me answer basic questions:

```text
Did the file load correctly?
Are the column names correct?
Do the values look reasonable?
Is the target column present?
Are there obvious missing values?
```

The first inspection is not a small step. It protects me from many mistakes.

---

## 8. Checking Shape

The shape tells me how many rows and columns the DataFrame has.

```python
df.shape
```

Example output:

```text
(1000, 8)
```

This means:

```text
1000 rows
8 columns
```

In Machine Learning language:

```text
1000 samples
8 variables
```

Not every column is necessarily a feature. Some columns may be identifiers, text, dates, labels, or target values. But shape gives the first overview.

---

## 9. Checking Column Names

Column names are very important because they tell me what the dataset contains.

```python
df.columns
```

Example:

```text
Index(['name', 'study_hours', 'attendance', 'passed'], dtype='object')
```

In ML, I often need to separate features and target.

For example:

```text
features -> study_hours, attendance
target   -> passed
```

So before modeling, I need to know which column is the target.

---

## 10. Understanding Data Types

The method `.dtypes` shows the data type of each column.

```python
df.dtypes
```

Example:

```text
name            object
study_hours      int64
attendance       int64
passed           int64
dtype: object
```

This matters because ML models usually need numerical features. If a number is accidentally stored as text, the model may fail or behave incorrectly.

For example:

```text
"90" -> text
90   -> number
```

A human sees both as similar, but Python does not.

---

## 11. The `.info()` Method

The method `.info()` gives a compact summary of the DataFrame.

```python
df.info()
```

It shows:

```text
number of rows
column names
non-null counts
data types
memory usage
```

This is especially useful for detecting missing values and incorrect data types.

A typical habit:

```python
df.head()
df.info()
df.describe()
```

These three methods give a quick first understanding of a dataset.

---

## 12. The `.describe()` Method

The method `.describe()` summarizes numerical columns.

```python
df.describe()
```

It usually shows:

```text
count
mean
standard deviation
minimum
25% percentile
50% percentile
75% percentile
maximum
```

For example, if I have a column `study_hours`, `.describe()` helps me see whether values are reasonable.

If the maximum value of `study_hours` is 1000, something may be wrong. Maybe the unit is different, or maybe there is an outlier.

This is why summary statistics matter. They help me detect suspicious data before modeling.

---

## 13. Selecting One Column

To select one column, I can use square brackets.

```python
study_hours = df["study_hours"]
```

This returns a Pandas Series.

A Series is like one column of a DataFrame.

```python
print(type(study_hours))
```

Output:

```text
<class 'pandas.core.series.Series'>
```

Simple mental model:

```text
DataFrame -> table
Series    -> one column
```

---

## 14. Selecting Multiple Columns

To select multiple columns, I pass a list of column names.

```python
features = df[["study_hours", "attendance"]]
```

This returns a DataFrame with only those columns.

This is very important in ML because features are often selected as a group.

```python
X = df[["study_hours", "attendance"]]
y = df["passed"]
```

Here:

```text
X -> feature matrix
y -> target vector
```

This is one of the most important patterns in Machine Learning.

Mathematically:

$$
X =
\begin{bmatrix}
x_{11} & x_{12} \\
x_{21} & x_{22} \\
x_{31} & x_{32}
\end{bmatrix}
$$

$$
y =
\begin{bmatrix}
y_1 \\
y_2 \\
y_3
\end{bmatrix}
$$

In Pandas:

```python
X = df[["study_hours", "attendance"]]
y = df["passed"]
```

This is where tabular data starts becoming ML-ready.

---

## 15. Filtering Rows

Filtering means selecting rows based on a condition.

```python
high_attendance = df[df["attendance"] > 80]
```

This returns only rows where attendance is greater than 80.

Filtering is useful for exploring data.

Examples:

```python
passed_students = df[df["passed"] == 1]
low_attendance = df[df["attendance"] < 70]
high_study_hours = df[df["study_hours"] >= 4]
```

In ML, filtering helps with:

```text
understanding groups
detecting suspicious values
checking class distribution
analyzing model errors
cleaning data
```

---

## 16. Creating New Columns

Feature engineering often means creating new columns from existing columns.

For example:

```python
df["study_attendance_score"] = df["study_hours"] * df["attendance"]
```

This creates a new column.

This is a tiny example of feature engineering. The new feature combines two pieces of information: how much the student studied and how often the student attended.

In real ML, feature engineering can be very powerful. Sometimes a simple model with good features can beat a complex model with weak features.

---

## 17. Missing Values

Real datasets often have missing values.

Pandas represents missing values as `NaN`.

```python
df.isna()
```

This shows whether each value is missing.

More useful:

```python
df.isna().sum()
```

This gives the number of missing values per column.

Example:

```text
name           0
study_hours    1
attendance     0
passed         0
dtype: int64
```

Missing values matter because many ML models cannot handle them directly.

---

## 18. Handling Missing Values

There are different strategies for missing values.

One option is to drop rows with missing values:

```python
df_clean = df.dropna()
```

Another option is to fill missing values:

```python
df["study_hours"] = df["study_hours"].fillna(df["study_hours"].mean())
```

This fills missing study hours with the mean study hours.

There is no universal best method. The right strategy depends on the problem, the amount of missing data, and the reason values are missing.

This is why data cleaning is not mechanical. It requires thinking.

---

## 19. Grouping Data

Grouping means splitting data into groups and calculating summaries.

For example:

```python
df.groupby("passed")["study_hours"].mean()
```

This calculates the average study hours for students who passed and students who did not pass.

This is useful for understanding relationships.

If passed students studied much more on average, then `study_hours` may be an important feature.

Grouping is one of the most useful tools in exploratory data analysis.

---

## 20. Value Counts

The method `.value_counts()` counts how often each value appears.

```python
df["passed"].value_counts()
```

Example:

```text
1    70
0    30
Name: passed, dtype: int64
```

This is very important for classification problems.

If one class appears much more often than another, the dataset is imbalanced.

For example:

```text
passed = 95%
failed = 5%
```

In that case, accuracy can become misleading. A model that always predicts "passed" would get 95% accuracy, but it would be useless for detecting failing students.

So checking class distribution is an important ML habit.

---

## 21. Sorting Values

Sorting helps me inspect extremes.

```python
df.sort_values("study_hours")
```

To sort from highest to lowest:

```python
df.sort_values("study_hours", ascending=False)
```

This can help find outliers.

For example, if one student has `study_hours = 500`, I should investigate. It might be a data entry error.

---

## 22. Converting Pandas to NumPy

Sometimes ML libraries need NumPy arrays.

I can convert a DataFrame or Series using `.to_numpy()`.

```python
X = df[["study_hours", "attendance"]].to_numpy()
y = df["passed"].to_numpy()
```

Now `X` and `y` are NumPy arrays.

This connects Pandas to the NumPy and ML world.

```text
Pandas DataFrame -> clean and inspect data
NumPy array      -> numerical computation / model input
```

---

## 23. A Tiny ML Preparation Example

Suppose I have this DataFrame:

```python
import pandas as pd

df = pd.DataFrame({
    "study_hours": [5, 2, 4, 1, 6],
    "attendance": [90, 60, 85, 50, 95],
    "passed": [1, 0, 1, 0, 1]
})
```

I can separate features and target:

```python
X = df[["study_hours", "attendance"]]
y = df["passed"]
```

Then convert to NumPy:

```python
X_np = X.to_numpy()
y_np = y.to_numpy()
```

Now the data is closer to what a model expects.

This is not model training yet. But it is the step right before model training.

---

## 24. Common Mistakes

One common mistake is training a model without inspecting the data. This is dangerous because the dataset may contain missing values, wrong types, duplicate rows, or strange outliers.

Another mistake is confusing features and target. If the target accidentally appears inside the features, the model may cheat. This is called data leakage.

A third mistake is ignoring class balance. If one class dominates the dataset, accuracy may look good even when the model is not useful.

A fourth mistake is modifying a DataFrame without realizing it. Pandas operations can sometimes return a copy, and sometimes they modify data depending on how they are used. When learning, it is better to write clear steps and inspect the result often.

---

## 25. What I Learned From This Lesson

Pandas helps me work with tabular data. It gives me tools to load, inspect, clean, summarize, filter, group, and prepare data.

The most important ideas from this lesson are:

```text
DataFrame
Series
head
shape
columns
dtypes
info
describe
selecting columns
filtering rows
creating new columns
missing values
groupby
value_counts
sorting
to_numpy
features and target
```

Pandas is not just a library. It is a way to ask questions about data before trusting a model.

---

## Mini Exercise

Create a file called `07-pandas-basics-example.py` inside the `code` folder.

The script should:

```text
1. Create a DataFrame about students.
2. Print the first rows.
3. Print shape, columns, info, and summary statistics.
4. Check missing values.
5. Fill missing study hours with the mean.
6. Create a new feature.
7. Check target distribution with value_counts.
8. Separate X and y.
9. Convert X and y to NumPy arrays.
```

Run the file:

```powershell
python code\07-pandas-basics-example.py
```

---

## Further Reading and Resources

### Official Documentation

- [Pandas Official Website](https://pandas.pydata.org/)
- [Pandas Getting Started](https://pandas.pydata.org/docs/getting_started/index.html)
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)
- [10 Minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html)

### Books and Longer Reading

- [Python for Data Analysis, 3rd Edition by Wes McKinney](https://wesmckinney.com/book/)
- [Python Data Science Handbook by Jake VanderPlas](https://jakevdp.github.io/PythonDataScienceHandbook/)
- [Pandas Cookbook by Theodore Petrou](https://github.com/PacktPublishing/Pandas-Cookbook)

### Practice

- [Kaggle Learn: Pandas](https://www.kaggle.com/learn/pandas)
- [Kaggle Learn: Data Cleaning](https://www.kaggle.com/learn/data-cleaning)
- [W3Schools Pandas Tutorial](https://www.w3schools.com/python/pandas/default.asp)

### What to Study Next

The next step is **Matplotlib and basic data visualization**. Pandas helps me inspect tables, but visualization helps me see patterns, outliers, class imbalance, and relationships between variables more clearly.
