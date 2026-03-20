import re

MYANMAR_CHAR_RE = re.compile(r"[\u1000-\u109F\uAA60-\uAA7F]")
MYANMAR_TOKEN_RE = re.compile(r"[\u1000-\u109F\uAA60-\uAA7F]+|[a-zA-Z0-9]+")

SCRIPTS = {
    "en": {
        "initials": ["How do you do. Please tell me your problem.", "Is something troubling you?"],
        "finals": ["Goodbye. It was nice talking to you.", "Take care."],
        "quits": ["bye", "quit", "exit"],
        "pres": {"don't": "dont", "i'm": "i am", "recollect": "remember", "machine": "computer"},
        "posts": {"am": "are", "i": "you", "my": "your", "me": "you", "your": "my"},
        "keywords": [
            [r"(.*) die (.*)", ["Please don't talk like that. Tell me more about your feelings."], 10],
            [r"i need (.*)", ["Why do you need {0}?", "Would it help you to get {0}?"], 5],
            [r"i am (.*)", ["Is it because you are {0} that you came to me?", "How long have you been {0}?"], 5],
            [r"(.*) problem (.*)", ["Tell me more about this problem.", "How does it make you feel?"], 8],
            [r"(.*)", ["Please tell me more.", "I see.", "Can you elaborate?"], 0],
        ],
    },
    "mm": {
        "initials": [
            "မင်္ဂလာပါ။ သင့်စိတ်ထဲမှာရှိတာကို ပြောပြပါ။",
            "ဘာက သင့်ကို စိတ်အနှောင့်အယှက်ဖြစ်စေတာလဲ။",
        ],
        "finals": [
            "နှုတ်ဆက်ပါတယ်။ ပြောပြပေးတာ ကျေးဇူးတင်ပါတယ်။",
            "ဒီနေ့အတွက် ဒီလောက်နဲ့ရပ်မယ်နော်။",
        ],
        "quits": ["တာ့တာ", "ထွက်မယ်", "ပြီးပြီ", "bye", "quit", "exit"],
        "pres": {
            "ကျွန်တော်": "ကျွန်တော်",
            "ကျွန်မ": "ကျွန်မ",
            "ငါ": "ငါ",
            "မသိဘူး": "မသိ",
            "မပျော်ဘူး": "မပျော်",
        },
        "posts": {
            "ကျွန်တော်": "သင်",
            "ကျွန်မ": "သင်",
            "ကျွန်ုပ်": "သင်",
            "ငါ": "သင်",
            "ကျွန်တော့်": "သင့်",
            "ကျွန်မရဲ့": "သင့်",
            "ငါ့": "သင့်",
            "ငါ့ရဲ့": "သင့်",
            "သင်": "ကျွန်တော်",
            "သင့်": "ကျွန်တော့်",
        },
        "keywords": [
            [r"(.*)(သေချင်|မနေချင်တော့|ကိုယ့်ကိုယ်ကို သတ်ချင်)(.*)", ["အဲဒီလို ခံစားနေရတာကို ပိုပြောပြပါ။ အခု ဘာဖြစ်နေတယ်လို့ ထင်သလဲ။"], 10],
            [r"(?:ကျွန်တော်|ကျွန်မ|ငါ)\s?(.*)လိုအပ်(?:တယ်|ပါတယ်)", ["ဘာကြောင့် {0} လိုအပ်တာလဲ။", "{0} ရရင် သက်သာမယ်လို့ ထင်သလား။"], 6],
            [r"(?:ကျွန်တော်|ကျွန်မ|ငါ)\s?(.*)ခံစားရ(?:တယ်|ပါတယ်)", ["{0} လို့ ခံစားရတာ ဘယ်အချိန်က စတာလဲ။", "{0} လို့ ခံစားရတာကို ပိုရှင်းပြပါ။"], 6],
            [r"(?:ကျွန်တော်|ကျွန်မ|ငါ)\s?(.*)ဖြစ်နေ(?:တယ်|ပါတယ်)", ["{0} ဖြစ်နေတာ ဘာကြောင့်လို့ ထင်သလဲ။", "{0} ဖြစ်နေတာကို ပိုပြောပြပါ။"], 5],
            [r"(.*)(ပြဿနာ|အခက်အခဲ)(.*)", ["ဒီပြဿနာအကြောင်း ပိုပြောပြပါ။", "ဒီအရာက သင့်ကို ဘယ်လို ခံစားရစေလဲ။"], 8],
            [r"(.*)", ["ဆက်ပြောပြပါ။", "နားလည်ပါတယ်။", "အဲဒါကို နည်းနည်းပိုရှင်းပြပါ။"], 0],
        ],
    },
}



class Eliza:
    def __init__(self, language="mm"):
        if language not in SCRIPTS:
            raise ValueError(f"Unsupported language: {language}")
        self.script = SCRIPTS[language]
        self.language = language

    def normalize_text(self, text):
        text = str(text).strip().lower()
        text = re.sub(r"[၊။!?,;:\"'()\[\]{}]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def respond(self, text):
        text = self.normalize_text(text)
        for pattern, responses, priority in self.script["keywords"]:
            match = re.match(pattern, text)
            if match:
                response = responses[0].format(*match.groups())
                return response
        return "I'm not sure how to respond to that."