import os
import re
import random
import pandas as pd
from myanmartools import ZawgyiDetector
from rabbit import Rabbit

detector = ZawgyiDetector()

def clean_myanmar_text(text):

    # Unicode/Zawgyi Normalize
    score = detector.get_zawgyi_probability(text)
    if score > 0.99:
        text = Rabbit.zg2uni(text)
    
    # English letters and numbers removal
    text = re.sub(r'[a-zA-Z0-9]', ' ', text)
    
    # Special Characters removal
    text = re.sub(r'[^\u1000-\u109F\s]', '', text)
    
    # Extra spaces removal
    text = " ".join(text.split())
    
    return text

def prepare_myanmar_dataset(folder_path):
    dataset = []
    label_map = {
        "sad": 0,
        "joy": 1,
        "love": 2,
        "anger": 3,
        "fear": 4,
        "surprise": 5
    }
    
    print("Processing files...")
    for filename in os.listdir(folder_path):
        category = filename.split('.')[0].lower()
        
        if category in label_map:
            label = label_map[category]
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                # cleaning linebreak
                lines = f.read().splitlines()
                
                for line in lines:
                    # Clean text
                    cleaned_text = clean_myanmar_text(line)
                    
                    # only add if text is not empty
                    if cleaned_text:
                        # add to dataset in format: text ||| label
                        formatted_entry = f"{cleaned_text}|||{label}"
                        dataset.append(formatted_entry)
            
            print(f"Done: {filename} (Label: {label})")

    # Shuffling
    random.seed(42)
    random.shuffle(dataset)
    
    return dataset

# Usage
folder_name = "rawdata" # folder name
final_dataset = prepare_myanmar_dataset(folder_name)

# Checking Resultss
print(f"\nTotal Data Count: {len(final_dataset)}")
print("Sample Data:", final_dataset[:3])

# List to DataFrame
df_list = [item.split("|||") for item in final_dataset]
df = pd.DataFrame(df_list, columns=['text', 'label'])

# Save to CSV
df.to_csv("myanmar_emotion_dataset.csv", index=False, encoding='utf-8-sig')
print("Saved to myanmar_emotion_dataset.csv")
