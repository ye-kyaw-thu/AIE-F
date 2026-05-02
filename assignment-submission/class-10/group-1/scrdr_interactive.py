## For understanding SCRDR to my AI Engineering (Fundamental) Class students
## Written by Ye Kyaw Thu, LU Lab., Myanmar
## Last Updated: 12 April 2026
##
## Reference: 
## The Alternative to Machine Learning
## by Paul Compton (The University of NSW, Sydney, Australia), 
## Byeong Ho Kang (The University of Tasmania, Hobart, Australia)


import pandas as pd
import json
import argparse
import os
import sys
from sklearn.metrics import classification_report

class RDRNode:
    def __init__(self, condition=None, conclusion=None):
        self.condition = condition # {'col':.., 'op':.., 'val':..}
        self.conclusion = str(conclusion) if conclusion is not None else None
        self.if_true = None   # Refinement (Exception)
        self.if_false = None  # Alternative

    def evaluate(self, row):
        # The root/default node always matches 
        if not self.condition or not self.condition['col']:
            return True
        col, op, val = self.condition['col'], self.condition['op'], self.condition['val']
        row_val = row[col]
        try:
            if op == '==': return str(row_val) == str(val)
            if op == '<':  return float(row_val) < float(val)
            if op == '>':  return float(row_val) > float(val)
        except:
            return str(row_val) == str(val)
        return False

    def to_dict(self):
        return {
            "condition": self.condition,
            "conclusion": self.conclusion,
            "if_true": self.if_true.to_dict() if self.if_true else None,
            "if_false": self.if_false.to_dict() if self.if_false else None
        }

    @staticmethod
    def from_dict(data):
        if not data: return None
        node = RDRNode(data['condition'], data['conclusion'])
        node.if_true = RDRNode.from_dict(data['if_true'])
        node.if_false = RDRNode.from_dict(data['if_false'])
        return node

class SCRDR_Engine:
    def __init__(self, target, default_conclusion="Unknown"):
        self.target = target
        self.root = RDRNode(conclusion=default_conclusion)

    def classify(self, row):
        curr = self.root
        last_match = self.root
        while curr:
            if curr.evaluate(row):
                last_match = curr
                curr = curr.if_true # Try refinements 
            else:
                curr = curr.if_false # Try alternatives 
        return last_match

    def add_rule(self, row, last_node):
        print(f"\n[KNOWLEDGE ACQUISITION]")
        print(f"System predicted: '{last_node.conclusion}' but Actual is: '{row[self.target]}'")
        
        def get_validated_input(prompt, options=None, check_cols=False):
            while True:
                val = input(prompt).strip()
                if val.lower() == 'exit':
                    return 'EXIT_COMMAND'
                if not val:
                    print("Input cannot be empty.")
                    continue
                if options and val not in options:
                    print(f"Invalid input. Choose from: {options}")
                    continue
                if check_cols and val not in row.index:
                    print(f"Invalid column. Available: {list(row.index)}")
                    continue
                return val

        # 1. Feature Name
        col = get_validated_input(f"Enter feature name (or 'exit'): ", check_cols=True)
        if col == 'EXIT_COMMAND': return False
        
        # 2. Operator
        op = get_validated_input("Enter operator (==, <, >): ", options=['==', '<', '>'])
        if op == 'EXIT_COMMAND': return False
        
        # 3. Value
        val = get_validated_input("Enter threshold value: ")
        if val == 'EXIT_COMMAND': return False
        
        new_node = RDRNode(condition={'col': col, 'op': op, 'val': val}, conclusion=row[self.target])
        
        # SCRDR Attachment Logic 
        if last_node.evaluate(row):
            # Predicted node matched (but was wrong) -> This is a REFINEMENT
            if last_node.if_true is None:
                last_node.if_true = new_node
            else:
                curr = last_node.if_true
                while curr.if_false: curr = curr.if_false
                curr.if_false = new_node
        else:
            # Predicted node did not match (it was an alternative) -> This is a new ALTERNATIVE
            curr = last_node
            while curr.if_false: curr = curr.if_false
            curr.if_false = new_node
        return True

def main():
    help_epilog = """
Interactive Workflow:
1. Use --mode build to teach the model. The script only asks for rules when it is wrong.
2. Type 'exit' at any prompt to save the current tree and stop.
3. Use --mode test to evaluate a built model without being prompted for rules.
    """
    parser = argparse.ArgumentParser(
        description="Interactive RDR Knowledge Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=help_epilog
    )
    parser.add_argument("--input", required=True, help="Dataset CSV file")
    parser.add_argument("--target", required=True, help="Target column to predict")
    parser.add_argument("--tree", default="rdr_model.json", help="Output JSON model file")
    parser.add_argument("--exclude", nargs='*', help="Columns to ignore (IDs, Names)")
    parser.add_argument("--mode", choices=['build', 'test'], default='build', 
                        help="build: Interactive training | test: Run model on data")
    
    args = parser.parse_args()

    # Load Data
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    if args.exclude:
        df = df.drop(columns=[c for c in args.exclude if c in df.columns])

    engine = SCRDR_Engine(args.target)
    
    # Load existing model if it exists
    if os.path.exists(args.tree):
        with open(args.tree, 'r') as f:
            engine.root = RDRNode.from_dict(json.load(f))
        print(f"Loaded existing model: {args.tree}")

    y_true, y_pred = [], []
    total_rows = len(df)
    stopped_early = False

    print(f"--- Running in {args.mode.upper()} mode ---")
    
    for idx, row in df.iterrows():
        pred_node = engine.classify(row)
        actual = str(row[args.target])
        predicted = str(pred_node.conclusion)
        
        if args.mode == 'build' and predicted != actual:
            print(f"\n[Row {idx+1}/{total_rows}] ERROR Found!")
            print(row.to_dict())
            
            success = engine.add_rule(row, pred_node)
            if not success:
                print("\nExiting and saving model...")
                stopped_early = True
                break
                
            # Save progress immediately
            with open(args.tree, 'w') as f:
                json.dump(engine.root.to_dict(), f, indent=2)
            
            # Update prediction after rule addition
            predicted = str(engine.classify(row).conclusion)

        y_true.append(actual)
        y_pred.append(predicted)

    print("\n--- FINAL PERFORMANCE SUMMARY ---")
    if stopped_early:
        print(f"(Partial results based on first {len(y_true)} rows)")
    print(classification_report(y_true, y_pred, zero_division=0))

if __name__ == "__main__":
    main()

