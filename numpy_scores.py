import numpy as np

# Task 1 — Generate and Inspect the Data
np.random.seed(42)

# Generate 5 students × 4 subjects, integers from 50 to 100 inclusive
# randint(low, high) → high is exclusive → use 101 for inclusive 100
scores = np.random.randint(50, 101, size=(5, 4))

print("Generated scores array (5 students × 4 subjects):")
print(scores)
print()

# Extractions / slicing
print("1. Score of the 3rd student (index 2) in the 2nd subject (index 1):")
print(scores[2, 1])
print()

print("2. All scores of the last 2 students (rows 3 and 4):")
print(scores[-2:, :])          # or scores[3:5, :]
print()

print("3. All scores for the first 3 students (rows 0–2) in subjects 2 and 3 (columns 1–2):")
print(scores[:3, 1:3])
print()

# Task 2 — Analysis with Broadcasting (no loops)
# Column-wise mean (average per subject), rounded to 2 decimal places
col_means = np.mean(scores, axis=0).round(2)
print("Column-wise means (per subject):", col_means)

# Add curve using broadcasting: different bonus per subject
curve = np.array([5, 3, 7, 2])          # shape (4,) → will broadcast to (5,4)
curved_scores = scores + curve

# Cap at 100 (element-wise minimum)
curved_scores = np.minimum(curved_scores, 100)

print("\nCurved scores (after adding curve and capping at 100):")
print(curved_scores)

# Row-wise max (best subject score per student)
row_maxes = np.max(curved_scores, axis=1)
print("\nBest subject score per student:", row_maxes)

# Task 3 — Normalize and Identify
# Min-max normalization per row (student)
row_mins  = curved_scores.min(axis=1, keepdims=True)   # shape (5,1)
row_maxs  = curved_scores.max(axis=1, keepdims=True)   # shape (5,1)
normalized = (curved_scores - row_mins) / (row_maxs - row_mins)

print("\nMin-max normalized scores (per student, 0–1 scale):")
print(normalized.round(4))   # rounded for readability

# Find the single highest value across the entire normalized array
max_value = normalized.max()
max_location = np.where(normalized == max_value)

# np.where returns tuples of arrays → take first occurrence
student_idx = max_location[0][0]   # row
subject_idx = max_location[1][0]   # column

print(f"\nHighest normalized value ({max_value:.4f}) found at:")
print(f"  Student index: {student_idx}  (0-based)")
print(f"  Subject index: {subject_idx}  (0-based)")

# Extract all scores > 90 from curved_scores using boolean masking
above_90 = curved_scores[curved_scores > 90]

print("\nAll curved scores strictly above 90 (1D array):")
print(above_90)