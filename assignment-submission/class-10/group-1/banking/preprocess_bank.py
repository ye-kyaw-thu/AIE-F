import pandas as pd

df = pd.read_csv("./data/bank.csv")

# 1. Drop leakage column
df = df.drop(columns=["duration"])

# 2. Convert target
df["deposit"] = df["deposit"].map({"yes": 1, "no": 0})

# 3. Encode categorical
from sklearn.preprocessing import LabelEncoder

for col in df.select_dtypes(include="object").columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# 4. Sample (50 train, 20 test)
train = df.sample(n=50, random_state=42)
remaining = df.drop(train.index)

test = remaining.sample(n=20, random_state=42)

# 5. Save
train.to_csv("./data/train.csv", index=False)
test.to_csv("./data/test.csv", index=False)