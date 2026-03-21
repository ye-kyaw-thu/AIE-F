"""Rule data for ELIZA-style responses (keywords, pres/posts, quits). Imported by src/eliza.py."""

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
            "မင်္ဂလာပါ။ သင့်စိတ်ထဲမှာ ရှိတာကို ပြောပြပါ။",
            "ဘာက သင့်ကို စိတ်အနှောင့်အယှက် ပေးနေသလဲ။",
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
            [
                # crisis / self-harm cues: Burmese + English lowercased
                r"(.*)((?:ကိုယ့်ကိုယ်ကို\s*သတ်ချင်|မနေချင်တော့|အဆုံးစီရင်ချင်|သေချင်|suicide|die|သတ်|သေ))(.*)",
                [
                    "ရင်ထဲမှာ တော်တော် မွန်းကြပ်နေမှာပဲနော်။ ကြားရတာ ဝမ်းနည်းပါတယ်။ ဘာများဖြစ်လို့လဲ?",
                    "စာနာမိပါတယ်။ ဒါပေမယ့် စိတ်လိုက်မာန်ပါ မလုပ်ပါနဲ့ဦး။ ရင်ထဲမှာ ဘယ်လိုတောင် ခံစားနေရလဲ?",
                    "ကြားရတာ စိတ်မကောင်းပါဘူး။ ဘယ်လို အကြောင်းရင်းတွေ ရှိနေလို့လဲ? ရင်ဖွင့်မယ်ဆို နားထောင်ပေးဖို့ အသင့်ပါပဲ။",
                ],
                10,
            ],
            [
                # need cues: Burmese + English lowercased
                r"(?:ကျွန်တော်|ကျွန်မ|ငါ)\s?(.*)လိုအပ်(?:ပါတယ်|တယ်)",
                [
                    "နားလည်ပါတယ်။ {0} က သင့်အတွက် အတော်အရေးကြီးမှာပဲ။ ဘာကြောင့် {0} လိုအပ်တာလဲ?",
                    "{0} ရရင် ပိုပြီး စိတ်ထဲမှာ ပိုပြီးသက်သာရာရသွားမယ်လို့ ထင်လား?",
                ],
                6,
            ],
            [
                # feelings cues: Burmese + English lowercased
                r"(?:ကျွန်တော်|ကျွန်မ|ငါ)\s?(.*)ခံစားရ(?:ပါတယ်|တယ်)",
                [
                    "{0} လို့ ခံစားရတာ ဘယ်အချိန်တည်းကလဲ?",
                    "{0} လို့ ဘာလို့ခံစားရတာလဲ ပြောပြပါဦး။",
                ],
                6,
            ],
            [
                # state cues: Burmese + English lowercased
                r"(?:ကျွန်တော်|ကျွန်မ|ငါ)\s?(.*)ဖြစ်နေ(?:ပါတယ်|တယ်)",
                [
                    "{0} ဖြစ်နေတာ ဘာကြောင့်လို့ ထင်သလဲ?",
                    "ဘာတွေက သင့်ကို {0} ဖြစ်စေတာလဲ?",
                ],
                5,
            ],
            [
                # sadness cues: Burmese + English lowercased
                r"(.*)(depression|depressed|sadness|unhappy|စိတ်မကောင်း|ဝမ်းနည်း|စိတ်ညစ်|မပျော်|sad)(.*)",
                [
                    "စိတ်မကောင်းမဖြစ်ပါနဲ့ဦး။ အကြောင်းရင်းကို ရင်ဖွင့်လို့ရပါတယ်။",
                    "အဲဒီခံစားချက်က အတော်ဆိုးမှာပဲ။ ဘာတွေကြောင့် အခုလို ခံစားနေရတာလဲ?",
                ],
                7,
            ],
            [
                # anger cues: Burmese + English lowercased
                r"(.*)(ဒေါသထွက်|စိတ်ဆိုး|စိတ်တို|ဒေါသ|anger|angry)(.*)",
                [
                    "စိတ်ဆိုးတာဟာ သဘာဝပါပဲ။ ဘာလို့ အခုလို စိတ်ဆိုးနေရတာလဲ?",
                    "ဒေါသက သင့်ကို ဘယ်လို ထိခိုက်စေတာလဲ?",
                ],
                7,
            ],
            [
                # fear cues: Burmese + English lowercased
                r"(.*)(panic attack|anxiety attack|ကြောက်ရွံ့|ထိတ်လန့်|စိုးရိမ်|anxiety|panic|fear|ကြောက်|လန့်)(.*)",
                [
                    "အသက်ဝဝရှူပါ။ သင်ဘာကို အများဆုံး စိုးရိမ်နေတာလဲ?",
                    "သင့်ခံစားချက်ကို နားလည်ပါတယ်။ ဒီကြောက်ရွံ့မှုရဲ့ အကြောင်းရင်းကို ရှင်းပြပါဦး။",
                ],
                7,
            ],
            [
                # joy cues: Burmese + English lowercased
                r"(.*)(satisfied|pleased|စိတ်ချမ်းသာ|ဝမ်းသာ|happy|ပျော်ရွှင်|ပျော်|joy)(.*)",
                [
                    "သင့်ကို စိတ်ချမ်းသာစေတဲ့ အကြောင်းရင်းကို ပိုပြောပြပါဦး။",
                    "ဘာကြောင့် အခုလို ပျော်ရွှင်နေတာလဲ?",
                ],
                6,
            ],
            [
                # love cues: Burmese + English lowercased
                r"(.*)(affection|respect|admire|သဘောကျ|လေးစား|အားကျ|like|love|လွမ်း|ချစ်)(.*)",
                [
                    "သင့်ကို ဒီလိုခံစားရစေတဲ့ အကြောင်းရင်းက ဘာများလဲ?",
                    "ဘယ်သူ သို့မဟုတ် ဘာအရာကြောင့် အခုလို ခံစားရတာလဲ?",
                ],
                6,
            ],
            [
                # problem cues: Burmese + English lowercased
                r"(.*)(problem|အခက်အခဲ|ပြဿနာ|issue)(.*)",
                [
                    "ဒီပြဿနာအကြောင်း ပိုပြောပြပါလား။",
                    "ဒီအရာက သင့်ကို ဘယ်လို ခံစားရစေလဲ?",
                ],
                8,
            ],
            [
                # default fallback: Burmese + English lowercased
                r"(.*)",
                ["ဆက်ပြောပြပါ။", "နားလည်ပါတယ်။", "အဲဒါကို နည်းနည်းပိုရှင်းပြပါဦး။"],
                0,
            ],
        ],
    },
}
