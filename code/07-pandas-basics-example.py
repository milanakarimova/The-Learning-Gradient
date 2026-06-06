import pandas as pd


students = pd.DataFrame({
    "name": ["Aylin", "Rauf", "Leyla", "Murad", "Nigar", "Kamran"],
    "study_hours": [5, 2, 4, None, 6, 1],
    "attendance": [90, 60, 85, 70, 95, 50],
    "passed": [1, 0, 1, 0, 1, 0],
})

print("First rows:")
print(students.head())

print()
print("Shape:")
print(students.shape)

print()
print("Columns:")
print(students.columns)

print()
print("Data types:")
print(students.dtypes)

print()
print("Info:")
students.info()

print()
print("Summary statistics:")
print(students.describe())

print()
print("Missing values:")
print(students.isna().sum())

mean_study_hours = students["study_hours"].mean()
students["study_hours"] = students["study_hours"].fillna(mean_study_hours)

students["study_attendance_score"] = (
    students["study_hours"] * students["attendance"]
)

print()
print("After filling missing values and creating new feature:")
print(students)

print()
print("Target distribution:")
print(students["passed"].value_counts())

print()
print("Average study hours by target:")
print(students.groupby("passed")["study_hours"].mean())

X = students[["study_hours", "attendance", "study_attendance_score"]]
y = students["passed"]

X_np = X.to_numpy()
y_np = y.to_numpy()

print()
print("Feature matrix X:")
print(X)

print()
print("Target y:")
print(y)

print()
print("X as NumPy array:")
print(X_np)

print()
print("y as NumPy array:")
print(y_np)
