import json

class RDRNode:
    def __init__(self, condition=None, conclusion=None):
        self.condition = condition
        self.conclusion = conclusion
        self.if_true = None
        self.if_false = None

    @staticmethod
    def from_dict(data):
        if not data: return None
        print(f"Processing data keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
        node = RDRNode(data['condition'], data['conclusion'])
        node.if_true = RDRNode.from_dict(data['if_true'])
        node.if_false = RDRNode.from_dict(data['if_false'])
        return node

with open('snake_rules_demo.json', 'r') as f:
    data = json.load(f)
    RDRNode.from_dict(data)
