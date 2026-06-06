# 04 — Python Files, Modules, and Virtual Environments for Machine Learning

## Why This Lesson Exists

So far, I have written Python code in small files and notebooks. That is enough for learning basic syntax, but Machine Learning projects quickly become bigger than one file.

A real ML project usually has many parts: data files, notebooks, helper functions, configuration files, experiment results, saved plots, and sometimes trained models. If everything stays in one file, the project becomes hard to understand and hard to reuse.

This lesson is about three important project skills: working with files, creating modules, and using virtual environments. These topics may not look like Machine Learning at first, but they are part of the professional ML workflow.

Before I build models, I need to learn how to organize code and experiments properly.

---

## 1. The Big Picture

A beginner often writes code like this:

```text
one file -> all code inside it
```

That is fine at the beginning. But later, the project starts to grow.

A better structure looks like this:

```text
project/
├── data/
├── notebooks/
├── code/
├── outputs/
├── requirements.txt
└── README.md
```

Each folder has a role. The `data` folder stores datasets. The `notebooks` folder stores experiments. The `code` folder stores reusable Python scripts. The `outputs` folder stores generated results. The `requirements.txt` file stores the libraries needed for the project.

This is how learning becomes engineering.

---

## 2. Why File Handling Matters in ML

Machine Learning is data-centered. Data usually lives in files. It can be stored as CSV, Excel, JSON, text, images, audio, or other formats.

Even before learning Pandas, I should understand the basic idea:

```text
open file -> read or write content -> close file safely
```

Python gives us a clean way to work with files using `with open(...)`.

```python
with open("notes.txt", "w", encoding="utf-8") as file:
    file.write("Machine Learning starts with data.")
```

This creates a text file and writes one sentence into it.

The `with` keyword is important because it automatically closes the file when the writing is finished. This is safer than opening and closing files manually.

---

## 3. Writing to a File

Writing to a file means saving information from Python into a file on the computer.

For example:

```python
model_name = "KNN"
accuracy = 0.86

with open("experiment_result.txt", "w", encoding="utf-8") as file:
    file.write("Model: " + model_name + "\n")
    file.write("Accuracy: " + str(accuracy) + "\n")
```

This creates a file called `experiment_result.txt`.

The important part is that `accuracy` is a number, so I convert it to text using `str(accuracy)` before writing it.

This matters because files store text unless we use special formats. Python numbers, lists, and dictionaries often need to be converted before writing.

---

## 4. Reading From a File

Reading from a file means loading information back into Python.

```python
with open("experiment_result.txt", "r", encoding="utf-8") as file:
    content = file.read()

print(content)
```

The mode `"r"` means read. The mode `"w"` means write.

Basic file modes:

```text
"r" -> read
"w" -> write and replace existing content
"a" -> append to the end of a file
```

The `"w"` mode is powerful but dangerous because it replaces the old content. If I want to add new content without deleting the old one, I should use `"a"`.

---

## 5. Appending to a File

Appending means adding new content to the end of an existing file.

```python
with open("experiment_log.txt", "a", encoding="utf-8") as file:
    file.write("New experiment finished.\n")
```

This is useful for experiment logs. For example, each time I test a model, I can append a new result.

A simple experiment log may look like this:

```text
Model: KNN, Accuracy: 0.82
Model: Logistic Regression, Accuracy: 0.86
Model: Random Forest, Accuracy: 0.89
```

Later, tools like MLflow and Weights & Biases do this in a much more advanced way, but the basic idea is the same: record experiments so I do not lose information.

---

## 6. Why Modules Matter

A module is a Python file that contains reusable code.

For example, suppose I have a file called `ml_metrics.py`:

```python
def calculate_accuracy(true_labels, predicted_labels):
    correct = 0

    for i in range(len(true_labels)):
        if true_labels[i] == predicted_labels[i]:
            correct = correct + 1

    return correct / len(true_labels)
```

Now another file can import and use this function:

```python
from ml_metrics import calculate_accuracy
```

This is useful because I do not want to rewrite the same metric functions in every script.

In simple words:

```text
module = a Python file that I can import
```

Modules help me move from messy scripts to reusable project code.

---

## 7. Why Reusable Code Matters in ML

In Machine Learning, I often repeat the same operations:

```text
load data
clean data
split data
calculate metrics
train model
evaluate model
save results
```

If I write these steps from scratch every time, my project becomes repetitive. If I place reusable functions in modules, the project becomes cleaner.

For example:

```text
code/
├── ml_metrics.py
└── train_model.py
```

The file `ml_metrics.py` can store metric functions. The file `train_model.py` can import those functions and use them.

This is how small learning code slowly becomes project code.

---

## 8. Importing a Module

Imagine I have this file:

```text
code/ml_metrics.py
```

Inside it:

```python
def calculate_mean(values):
    return sum(values) / len(values)
```

In another file in the same folder, I can write:

```python
from ml_metrics import calculate_mean

scores = [0.72, 0.80, 0.85]
print(calculate_mean(scores))
```

The import statement tells Python:

```text
go to ml_metrics.py
find calculate_mean
bring it here
```

This makes code easier to reuse.

---

## 9. The Problem of One Huge File

Without modules, I may end up with one huge file:

```text
train_model_final_final.py
```

Inside it, I might have:

```text
data loading
cleaning
metrics
model training
plotting
saving
debugging
random experiments
```

This becomes hard to read.

A better project separates responsibilities:

```text
ml_metrics.py       -> metric functions
data_utils.py       -> data loading and cleaning
train_model.py      -> training workflow
evaluate_model.py   -> evaluation workflow
```

This separation makes the project easier to understand and easier to debug.

---

## 10. What is a Virtual Environment?

A virtual environment is an isolated Python environment for one project.

This means each project can have its own libraries and versions.

For example, one project may need:

```text
numpy==1.26.0
pandas==2.2.0
scikit-learn==1.4.0
```

Another project may need different versions.

Without a virtual environment, library versions can conflict. A virtual environment helps keep the project clean.

In simple words:

```text
virtual environment = separate Python workspace for a project
```

---

## 11. Creating a Virtual Environment

In PowerShell, I can create a virtual environment like this:

```powershell
python -m venv .venv
```

This creates a folder called `.venv`.

Then I activate it:

```powershell
.\.venv\Scripts\Activate
```

After activation, the terminal usually shows `(.venv)` at the beginning of the line.

That means I am now using the project environment.

To deactivate it:

```powershell
deactivate
```

---

## 12. What is `requirements.txt`?

A `requirements.txt` file lists the libraries needed for the project.

Example:

```text
numpy
pandas
matplotlib
scikit-learn
jupyter
```

After creating a virtual environment, I can install the libraries:

```powershell
pip install -r requirements.txt
```

This makes the project easier to reproduce.

If another person downloads my repository, they can install the same dependencies using the same file.

---

## 13. Why Reproducibility Matters

Reproducibility means that the same project can be run again and produce the same or similar result.

In Machine Learning, reproducibility is very important because experiments can depend on many things:

```text
library versions
dataset versions
random seeds
model settings
preprocessing steps
hardware
```

If I do not track my environment, I may run the same code later and get errors because the library versions changed.

A virtual environment and `requirements.txt` are the first steps toward reproducible ML work.

---

## 14. A Small Project Example

Suppose I want to organize a tiny ML-style project.

The structure can be:

```text
The-Learning-Gradient/
├── code/
│   ├── ml_metrics.py
│   └── 05-python-files-modules-example.py
├── outputs/
│   └── experiment_summary.txt
└── requirements.txt
```

The file `ml_metrics.py` stores reusable functions. The file `05-python-files-modules-example.py` imports those functions, calculates metrics, and writes results to an output file.

This is a small example, but the workflow is realistic:

```text
import reusable code -> calculate result -> save output
```

---

## 15. Example: Metric Module

The module file can look like this:

```python
def calculate_accuracy(true_labels, predicted_labels):
    correct = 0

    for i in range(len(true_labels)):
        if true_labels[i] == predicted_labels[i]:
            correct = correct + 1

    return correct / len(true_labels)
```

This function does not print anything. It returns the result.

Returning is better here because another script can decide what to do with the result: print it, save it, compare it, or plot it.

---

## 16. Example: Main Script

The main script can import the metric function.

```python
from ml_metrics import calculate_accuracy

true_labels = [1, 0, 1, 1, 0]
predicted_labels = [1, 0, 0, 1, 0]

accuracy = calculate_accuracy(true_labels, predicted_labels)

print("Accuracy:", accuracy)
```

This is cleaner than rewriting the function every time.

---

## 17. Example: Saving the Result

A script can also save the result.

```python
from pathlib import Path

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

with open(output_dir / "experiment_summary.txt", "w", encoding="utf-8") as file:
    file.write("Accuracy: 0.8\n")
```

Here, `Path("outputs")` represents the output folder. The line `output_dir.mkdir(exist_ok=True)` creates the folder if it does not already exist.

This is useful because code should not fail just because the output folder is missing.

---

## 18. Common Mistakes

One common mistake is writing all code in one file. This may work for tiny examples, but it becomes painful when the project grows.

Another mistake is forgetting that file paths depend on where the script is run from. If Python says it cannot find a file, I should check my current folder and the file path.

A third mistake is not using a virtual environment. This can create library conflicts later.

A fourth mistake is committing the `.venv` folder to GitHub. The virtual environment folder can be large and machine-specific. Usually, `.venv` should be added to `.gitignore`.

---

## 19. What Should Be in GitHub?

Good things to commit:

```text
.md lesson files
.py source code
.ipynb notebooks
requirements.txt
small example data
README.md
```

Things usually not to commit:

```text
.venv/
__pycache__/
large datasets
temporary outputs
private credentials
API keys
```

This is part of responsible project management.

---

## 20. What I Learned From This Lesson

Files help me save and load information. Modules help me reuse code. Virtual environments help me keep projects reproducible.

These topics may look less exciting than models, but they are what make ML projects stable. A model is not useful if I cannot organize its code, reproduce its environment, or save its results.

This lesson is about moving from "I wrote some Python" to "I am building a project."

---

## Mini Exercise

Create these files:

```text
code/ml_metrics.py
code/05-python-files-modules-example.py
requirements.txt
```

In `ml_metrics.py`, write reusable metric functions.

In `05-python-files-modules-example.py`, import the functions, calculate accuracy and MSE, then save the results into:

```text
outputs/experiment_summary.txt
```

Run the script:

```powershell
python code\05-python-files-modules-example.py
```

Then check whether the output file was created.

---

## Final Reflection

This lesson helped me understand that Machine Learning is not only about algorithms. It is also about project structure, reusable code, environments, and saved results.

A clean project is easier to read. A clean project is easier to debug. A clean project is easier to continue.

That is why files, modules, and virtual environments matter.
