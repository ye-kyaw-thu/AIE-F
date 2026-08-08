import re
import os

# Original Syllable Break  
def create_break_pattern():
    my_consonant = r"က-အ"
    en_char = r"a-zA-Z0-9"
    other_char = r"ဣဤဥဦဧဩဪဿ၌၍၏၀-၉၊။!-/:-@[-`{-~\s"
    subscript_symbol = r'္'
    a_that = r'်'
    return re.compile(
        r"((?<!" + subscript_symbol + r")[" + my_consonant + r"]"
        r"(?![" + a_that + subscript_symbol + r"])"
        + r"|[" + en_char + other_char + r"])"
    )

def break_syllables(line, break_pattern, separator='|'):
    line = re.sub(r'\s+', ' ', line.strip())
    segmented_line = break_pattern.sub(separator + r"\1", line)
    if segmented_line.startswith(separator):
        segmented_line = segmented_line[len(separator):]
    double_delimiter = separator + " " + separator
    segmented_line = segmented_line.replace(double_delimiter, " ")
    return segmented_line

# Keep English words whole instead of letter-by-letter
# Only myanmar syllable break
break_pattern = create_break_pattern()
en_word_pattern = re.compile(r'[a-zA-Z0-9]+')

def tokenize_line(line):
    tokens = []
    pos = 0
    for m in en_word_pattern.finditer(line):
        pre = line[pos:m.start()]
        if pre.strip():
            seg = break_syllables(pre, break_pattern, '|')
            tokens.extend([t for t in seg.split('|') if t.strip()])
        tokens.append(m.group())  # keep English word whole, untouched
        pos = m.end()
    tail = line[pos:]
    if tail.strip():
        seg = break_syllables(tail, break_pattern, '|')
        tokens.extend([t for t in seg.split('|') if t.strip()])
    return tokens

def tokenize_file(input_path, output_path):
    count = 0
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            tokens = tokenize_line(line)
            if tokens:
                fout.write(' '.join(tokens) + '\n')
                count += 1
    print(f"Tokenized {count} lines: {input_path} -> {output_path}")

os.makedirs("data/tokenized", exist_ok=True)

files = [
    "general_wikipedia.txt",
    "general_mypos.txt",
    "domain_facebook.txt",
    "domain_religious.txt",
    "domain_news.txt",
]

for f in files:
    tokenize_file(f"data/clean/{f}", f"data/tokenized/{f}")

print("All files tokenized.")
