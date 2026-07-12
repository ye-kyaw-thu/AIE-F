import argparse
import string

def remove_punctuation(input_path, output_path):
    # Define English punctuation from the string module
    english_punct = string.punctuation
    
    # Define Burmese specific punctuation
    burmese_punct = "၊။"
    
    # Combine them into one set of characters to remove
    all_to_remove = english_punct + burmese_punct
    
    # Create a translation table for fast processing
    # This maps every character in 'all_to_remove' to None
    table = str.maketrans('', '', all_to_remove)

    try:
        with open(input_path, 'r', encoding='utf-8') as infile:
            with open(output_path, 'w', encoding='utf-8') as outfile:
                for line in infile:
                    # Apply the translation table to each line
                    clean_line = line.translate(table)
                    outfile.write(clean_line)
        
        print(f"Success! Cleaned text saved to: {output_path}")
        
    except FileNotFoundError:
        print(f"Error: The file '{input_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove English and Burmese punctuation from a text file.")
    
    # Setting up --input/-i and --output/-o
    parser.add_argument('-i', '--input', required=True, help="Path to the source text file.")
    parser.add_argument('-o', '--output', required=True, help="Path where the cleaned file will be saved.")

    args = parser.parse_args()

    remove_punctuation(args.input, args.output)

