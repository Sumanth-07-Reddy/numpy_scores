import numpy as np

np.random.seed(42)

scores = np.random.randint(50, 101, size=(5, 4))

print("Generated scores array (5 students × 4 subjects):")
print(scores)
print()

print("1. Score of the 3rd student (index 2) in the 2nd subject (index 1):")
print(scores[2, 1])
print()

print("2. All scores of the last 2 students (rows 3 and 4):")
print(scores[-2:, :])          
print()

print("3. All scores for the first 3 students (rows 0–2) in subjects 2 and 3 (columns 1–2):")
print(scores[:3, 1:3])
print()

col_means = np.mean(scores, axis=0).round(2)
print("Column-wise means (per subject):", col_means)

curve = np.array([5, 3, 7, 2])         
curved_scores = scores + curve

curved_scores = np.minimum(curved_scores, 100)

print("\nCurved scores (after adding curve and capping at 100):")
print(curved_scores)

row_maxes = np.max(curved_scores, axis=1)
print("\nBest subject score per student:", row_maxes)

row_mins  = curved_scores.min(axis=1, keepdims=True)  
row_maxs  = curved_scores.max(axis=1, keepdims=True)   
normalized = (curved_scores - row_mins) / (row_maxs - row_mins)

print("\nMin-max normalized scores (per student, 0–1 scale):")
print(normalized.round(4))

max_value = normalized.max()
max_location = np.where(normalized == max_value)

student_idx = max_location[0][0]   # row
subject_idx = max_location[1][0]   # column

print(f"\nHighest normalized value ({max_value:.4f}) found at:")
print(f"  Student index: {student_idx}  (0-based)")
print(f"  Subject index: {subject_idx}  (0-based)")

above_90 = curved_scores[curved_scores > 90]

print("\nAll curved scores strictly above 90 (1D array):")

print(above_90)
