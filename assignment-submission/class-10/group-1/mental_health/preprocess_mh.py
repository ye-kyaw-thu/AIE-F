import pandas as pd

# 1. Load dataset (adjust path if needed)
df = pd.read_csv("./data/mental_health.csv")

# 2. Stratified sampling (balanced)
dep1 = df[df["Depression"] == 1].sample(n=35, random_state=42)
dep0 = df[df["Depression"] == 0].sample(n=35, random_state=42)

# 3. Create train (50 rows: 25 + 25)
train = pd.concat([
    dep1.head(25),
    dep0.head(25)
]).sample(frac=1, random_state=42).reset_index(drop=True)

# 4. Create test (20 rows: 10 + 10)
test = pd.concat([
    dep1.tail(10),
    dep0.tail(10)
]).sample(frac=1, random_state=42).reset_index(drop=True)

# 5. Save
train.to_csv("./data/train.csv", index=False)
test.to_csv("./data/test.csv", index=False)

# 6. Verification
overlap = set(train["Person_ID"]) & set(test["Person_ID"])

print("=== DATA SPLIT SUMMARY ===")
print("Train shape:", train.shape)
print("Test shape: ", test.shape)
print("Overlap Person_IDs:", overlap if overlap else "None — clean split")

print("\nTrain Depression counts:")
print(train["Depression"].value_counts().sort_index())

print("\nTest Depression counts:")
print(test["Depression"].value_counts().sort_index())