"""This module builds the vocabulary mapping from tokenized text and label id mappings."""

from collections import Counter


# function to build vocabulary from tokenized texts
def build_vocab(tokenized_texts, max_vocab=5000):

    # count the frequency of each word in the tokenized texts
    counter = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)

    # initialize word2id dictionary with <PAD> and <UNK> tokens
    word2id = {"<PAD>": 0, "<UNK>": 1}

    # add the most common words to the word2id dictionary
    for i, (word, _) in enumerate(counter.most_common(max_vocab), start=2):
        word2id[word] = i

    # return the dictionary
    return word2id


# display Burmese names for class ids
DEFAULT_LABEL_ORDER = ("ဝမ်းနည်းမှု", "ပျော်ရွှင်မှု", "ချစ်ခင်မှု", "ဒေါသ", "ကြောက်ရွံ့မှု", "အံ့အားသင့်မှု")

# function to build label map from labels
def build_label_map(fixed_unique_labels=None):
    """
    EXAMPLE: IF LABELS ARE STORED AS STRING, THEN:
        Inputs:
        - labels = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]
        
        Outputs:
        - label2id = {
            "Sadness": 0,
            "Joy": 1,
            "Love": 2,
            "Anger": 3,
            "Fear": 4,
            "Surprise": 5
        }
        - id2label = {
            0: "Sadness",
            1: "Joy",
            2: "Love",
            3: "Anger",
            4: "Fear",
            5: "Surprise"
        }
    """

    # if no fixed unique labels are provided, use the default label order
    if fixed_unique_labels is None:
        fixed_unique_labels = DEFAULT_LABEL_ORDER

    # create label2id dictionary
    label2id = {label: idx for idx, label in enumerate(fixed_unique_labels)}
    
    # create id2label dictionary
    id2label = {idx: label for label, idx in label2id.items()}

    # return the dictionaries
    return label2id, id2label


# function to encode labels to ids
def encode_labels(labels, label2id):
    """
    Use dictionary created above to encode labels to ids
    """
    return [label2id[label] for label in labels]