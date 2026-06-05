# 00 — Terminal, Git, and Setup

## Why I Am Starting Here

When I first thought about learning Machine Learning, I wanted to jump directly into models.

I wanted to understand things like:

- linear regression

- neural networks

- transformers

- LLMs

- RAG systems

- computer vision

But then I realized something important.

Machine Learning is not only about models.

It is also about files, folders, datasets, notebooks, code, experiments, results, reports, and version control.

Before training a model, I need to know how to organize my work.

Before building an AI system, I need to know how to control my project.

That is why this learning journey starts with the terminal, Git, GitHub, and project structure.

This is not the most glamorous part of Machine Learning, but it is one of the most important parts.

---

## 1. What is the Terminal?

The terminal is a text-based way to communicate with the computer.

Usually, we use the computer visually.

We click folders.

We open files.

We drag things.

We rename files manually.

But in the terminal, we give commands.

For example, instead of opening a folder by clicking, we can write:

```powershell

cd my-project

```

Instead of creating a folder manually, we can write:

```powershell

mkdir notes

```

Instead of checking files by opening File Explorer, we can write:

```powershell

ls

```

At first, the terminal may look scary because it has no buttons.

But actually, it is just a direct conversation with the computer.

The computer waits.

I write a command.

The computer answers.

That is the basic idea.

---

## 2. Why Do Machine Learning Students Need the Terminal?

Machine Learning projects can become messy very quickly.

A typical ML project may contain:

- datasets

- notebooks

- Python scripts

- trained models

- plots

- reports

- configuration files

- experiment results

- README files

If these files are not organized, the project becomes confusing.

For example, imagine this situation:

```text

final_model.ipynb

final_model2.ipynb

real_final_model.ipynb

real_final_model_fixed.ipynb

new_final_model_last_version.ipynb

```

This is not a project.

This is panic.

A serious ML project needs structure.

The terminal helps me create that structure.

It helps me move around the project, create files, create folders, run scripts, install libraries, and use Git.

So even though the terminal is not Machine Learning itself, it is part of the Machine Learning workflow.

---

## 3. My Mental Model: A Project is Like a Small City

I like to imagine a project as a small city.

Each folder is like a building.

Each building has a purpose.

```text

The-Learning-Gradient/

│

├── 00-start-here/        -> setup, terminal, Git, learning guide

├── 01-python-for-ml/     -> Python basics for ML

├── 02-math-for-ml/       -> math foundations

├── 03-data-analysis/     -> NumPy, Pandas, visualization

├── 04-machine-learning/  -> classical ML algorithms

├── 05-deep-learning/     -> neural networks and PyTorch

├── 06-llms-and-rag/      -> transformers, LLMs, RAG

├── 07-mlops/             -> deployment and monitoring

├── 08-projects/          -> practical projects

├── assets/               -> images and diagrams

├── notebooks/            -> experiments

├── code/                 -> reusable code

├── resources/            -> books, links, papers

└── README.md             -> map of the repository

```

The `README.md` file is like the entrance of the city.

It explains:

- what the project is

- why it exists

- how to use it

- what the learning path looks like

The folders are like streets.

The files are like rooms.

If everything has a place, learning becomes easier.

---

## 4. Basic Terminal Commands

### Show Current Location

```powershell

pwd

```

This command shows where I am in the computer.

Example output:

```text

C:\\Users\\Milana\\Desktop\\The-Learning-Gradient

```

This means I am inside the project folder.

---

### List Files and Folders

```powershell

ls

```

This shows the files and folders in the current location.

For example:

```text

README.md

00-start-here

01-python-for-ml

assets

notebooks

```

This helps me check whether I am in the right folder.

---

### Move Into a Folder

```powershell

cd 00-start-here

```

`cd` means "change directory".

A directory is just another word for folder.

---

### Move One Level Back

```powershell

cd ..

```

This moves me one folder back.

For example:

```text

The-Learning-Gradient/00-start-here

```

After writing:

```powershell

cd ..

```

I return to:

```text

The-Learning-Gradient

```

---

### Create a Folder

```powershell

mkdir 01-python-for-ml

```

This creates a new folder.

In ML projects, folders help separate different types of work.

For example:

```text

notebooks/  -> experiments

code/       -> reusable Python files

assets/     -> images

```

---

### Create a File

```powershell

New-Item notes.md -ItemType File

```

This creates a new file.

The `.md` extension means Markdown.

Markdown is useful because GitHub can display it beautifully.

That is why this repository uses `.md` files for lessons.

---

## 5. What is Markdown?

Markdown is a simple writing format.

It allows me to write clean notes with headings, lists, code blocks, formulas, and links.

For example:

```md

# Big Title

## Section Title

This is normal text.

- item 1

- item 2

- item 3

```

On GitHub, this becomes a clean formatted page.

Markdown is perfect for learning notes because it is simple, readable, and version-control friendly.

---

## 6. What is Git?

Git is a version control system.

That means Git saves the history of a project.

Without Git, I may change something and later forget what I changed.

With Git, I can track my work step by step.

A simple way to think about Git:

```text

Git = save points for a project

```

Like in a game, when I reach an important point, I save.

In Git, that save point is called a commit.

---

## 7. What is GitHub?

Git and GitHub are not the same thing.

Git is the tool that tracks project history.

GitHub is the online platform where I store and share my Git project.

Simple difference:

```text

Git    -> local version control tool

GitHub -> online place to host repositories

```

If Git is my notebook, GitHub is the library where I put that notebook so others can see it.

---

## 8. The Basic Git Workflow

The basic Git workflow is:

```text

change -> check -> stage -> commit -> push

```

In commands:

```powershell

git status

git add .

git commit -m "Add terminal and setup lesson"

git push

```

Now let me break this down.

---

### Step 1: Check Status

```powershell

git status

```

This shows what changed.

It tells me:

- which files are new

- which files are modified

- which files are ready to commit

- which files are not tracked yet

I should use `git status` often.

It is like asking:

> What is happening in my project right now?

---

### Step 2: Stage Changes

```powershell

git add .

```

This prepares all changed files for saving.

The dot `.` means:

```text

add everything that changed in this folder

```

Sometimes, instead of adding everything, I can add one specific file:

```powershell

git add README.md

```

---

### Step 3: Commit Changes

```powershell

git commit -m "Add terminal and setup lesson"

```

A commit is a saved checkpoint.

The message should explain what changed.

Good commit messages:

```text

Add terminal and setup lesson

Create ML learning folder structure

Update README with learning roadmap

```

Bad commit messages:

```text

update

final

changes

stuff

```

A good commit message helps future me understand the history of the project.

---

### Step 4: Push to GitHub

```powershell

git push

```

This sends my local commits to GitHub.

After pushing, the changes become visible online.

This is important because GitHub becomes my public learning portfolio.

---

## 9. Why Git Matters in Machine Learning

Machine Learning is experimental.

I may try one model today and another model tomorrow.

For example:

```text

Experiment 1: KNN with k = 3

Experiment 2: KNN with k = 5

Experiment 3: Logistic Regression

Experiment 4: Random Forest

Experiment 5: Neural Network

```

Each experiment may produce different results.

If I do not track my changes, I may forget:

- which model worked better

- which dataset version I used

- which preprocessing step changed

- which metric improved

- which code broke the project

Git helps me keep a history of my thinking.

That is why Git is not just a software engineering tool.

It is also a learning tool.

---

## 10. My First Repository Structure

For this learning journey, I want the repository to grow step by step.

The structure will be:

```text

The-Learning-Gradient/

│

├── 00-start-here/

│   ├── 00-terminal-git-and-setup.md

│   └── 01-how-to-learn-from-this-repo.md

│

├── 01-python-for-ml/

├── 02-math-for-ml/

├── 03-data-analysis/

├── 04-machine-learning/

├── 05-deep-learning/

├── 06-llms-and-rag/

├── 07-mlops/

├── 08-projects/

│

├── assets/

│   └── images/

│

├── notebooks/

├── code/

├── resources/

└── README.md

```

This structure follows the idea that I should not learn Machine Learning randomly.

I should learn it as a path.

First, I build the foundation.

Then I move to Python.

Then math.

Then data.

Then classical Machine Learning.

Then Deep Learning.

Then LLMs and AI Engineering.

---

## 11. A Small Example: Creating a Learning File

Suppose I want to create a new lesson about Python variables.

I can write:

```powershell

cd 01-python-for-ml

New-Item 01-python-variables.md -ItemType File

notepad 01-python-variables.md

```

Then I can write the lesson.

After saving, I return to the main folder:

```powershell

cd ..

```

Then I check Git:

```powershell

git status

```

Then I save my work:

```powershell

git add .

git commit -m "Add Python variables lesson"

git push

```

This is the learning cycle.

Write.

Check.

Save.

Push.

Repeat.

---

## 12. Common Mistakes I Should Avoid

### Mistake 1: Working in the Wrong Folder

If I am not inside the repository folder, Git commands may not work.

That is why I should check:

```powershell

pwd

```

---

### Mistake 2: Forgetting to Save the File

If I edit a file in Notepad but do not press `Ctrl + S`, Git may not detect the change.

Save first.

Then run:

```powershell

git status

```

---

### Mistake 3: Writing Bad Commit Messages

A commit message should explain the change.

Instead of:

```text

update

```

I should write:

```text

Add explanation of Git workflow

```

---

### Mistake 4: Being Afraid of the Terminal

The terminal is not magic.

It is just a place where I write commands.

At first, I may not remember everything.

That is normal.

The goal is not to memorize every command.

The goal is to understand the workflow.

---

## 13. What I Learned From This Lesson

In this lesson, I learned that before studying Machine Learning models, I need to prepare my working environment.

The terminal helps me control files and folders.

Git helps me save the history of my project.

GitHub helps me share my learning journey publicly.

A clean repository is not just about aesthetics.

It shows clear thinking.

And clear thinking is important in Machine Learning.

---

## 14. Mini Exercise

To practice this lesson, I should be able to do the following:

1. Open PowerShell.

2. Move into the repository folder.

3. Create a new folder.

4. Create a Markdown file.

5. Write something inside the file.

6. Save it.

7. Check Git status.

8. Commit the change.

9. Push it to GitHub.

Practice commands:

```powershell

pwd

ls

mkdir practice

cd practice

New-Item hello.md -ItemType File

notepad hello.md

cd ..

git status

git add .

git commit -m "Practice terminal and Git workflow"

git push

```

---

## Final Reflection

This lesson may look simple, but it is the beginning of the whole journey.

Every advanced AI system starts as a folder.

Every research project starts as a file.

Every strong portfolio starts with one clear commit.

So this is my first step:

I am not only learning Machine Learning.

I am learning how to think, organize, experiment, and build like an AI engineer.
