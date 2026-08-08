## ရည်ရွယ်ချက်

ဒီ assignment ရဲ့ အဓိကရည်ရွယ်ချက်က Marian NMT framework ကို အသုံးပြုပြီး phoneme-to-grapheme translation ကို လုပ်ကြည့်ပြီး၊ အချင်းချင်း မတူတဲ့ မော်ဒယ်နှစ်ခုကို နှိုင်းယှဉ်ကြည့်တာပါ။

- Sequence-to-Sequence (Seq2Seq) မော်ဒယ်
- Transformer မော်ဒယ်

ဒီ repository ထဲမှာ notebook၊ shell script၊ training data၊ checkpoint၊ log ဖိုင်များပါဝင်တဲ့အတွက် အခြား external folder မလိုဘဲ review လုပ်နိုင်ပါတယ်။

## ဒေတာနှင့် preprocessing

ဒီ project မှာ g2p-par corpus ကို အသုံးပြုထားပါတယ်။

```text
g2p-par/train.ph  g2p-par/train.my
g2p-par/dev.ph    g2p-par/dev.my
g2p-par/test.ph   g2p-par/test.my
```

Training နှင့် development data ကို ပေါင်းပြီး vocabulary ဖိုင်များကို ဆောက်ထားပါတယ်။

```bash
mkdir -p g2p-par/preprocessing g2p-par/vocab
cat g2p-par/train.my g2p-par/dev.my > g2p-par/preprocessing/train-dev.my
cat g2p-par/train.ph g2p-par/dev.ph > g2p-par/preprocessing/train-dev.ph
marian-vocab < g2p-par/preprocessing/train-dev.my > g2p-par/vocab/vocab.my.yml
marian-vocab < g2p-par/preprocessing/train-dev.ph > g2p-par/vocab/vocab.ph.yml
```

Notebook မှာ ဖော်ပြထားသလို combined file တွေမှာ တိုင်းတာချက်အရ ပိုင်းခြားထားသော line အရေအတွက် 22,000 နီးပါးနှင့် token အရေအတွက် 63,000 နီးပါးရှိပါတယ်။

## မော်ဒယ် training setup

ထိုမော်ဒယ်နှစ်ခုအတွက် training pipeline နှစ်ခုကို ရေးဆွဲထားပါတယ်။

### Seq2Seq မော်ဒယ်

seq2seq.phmy.sh script က LSTM layer ပါဝင်သော recurrent encoder-decoder architecture ကို အသုံးပြုပါတယ်။ အဓိက parameter တွေက အောက်ပါအတိုင်းဖြစ်ပါတယ်။

- encoder နှင့် decoder depth ကို 2 သတ်မှတ်ထားခြင်း
- tied embeddings နှင့် layer normalization အသုံးပြုခြင်း
- validation အတွက် BLEU၊ perplexity၊ cross-entropy ကို အသုံးပြုခြင်း
- checkpoint ကို every 5,000 updates တစ်ကြိမ်စီ သိမ်းခြင်း

### Transformer မော်ဒယ်

transformer.phmy.sh script က တူညီတဲ့ data နှင့် vocabulary ကို အသုံးပြုပြီး Transformer architecture နဲ့ training လုပ်ပါတယ်။

- encoder နှင့် decoder depth ကို 2 သတ်မှတ်ထားခြင်း
- attention heads နှင့် feed-forward size ကို script ထဲမှာ သတ်မှတ်ထားခြင်း
- dropout နှင့် label smoothing အသုံးပြုခြင်း
- validation အတွက် BLEU metric ကို အသုံးပြုခြင်း

## Evaluation နှင့် report လုပ်ခြင်း

မော်ဒယ်နှစ်ခုလုံးကို marian-decoder နဲ့ test set ပေါ်မှာ evaluate လုပ်ထားပြီး၊ BLEU score ကို multi-bleu.perl နဲ့ တွက်ထားပါတယ်။ Output အချက်အလက်များကို logs နှင့် root folder တွေမှာ သိမ်းထားပါတယ်။

## BLEU score ထဲမှာပါတဲ့ BP ဆိုတာဘာလဲ?

BLEU output မှာ ပုံမှန်အားဖြင့် BP ဟု တွေ့ရတတ်ပါတယ်။ ဒီ BP က brevity penalty ကို ရည်ညွှန်းပြီး၊ ဘာကြောင့်လဲဆိုတော့ translation က 너무 short ဖြစ်နေပါက score ကို လျှော့ချပေးတဲ့ factor ဖြစ်ပါတယ်။

ဥပမာအားဖြင့်:

```text
BLEU = 74.20, 86.3/76.4/70.4/65.6 (BP=0.999, ratio=0.999, hyp_len=8041, ref_len=8047)
```

ဒီ output မှာ BP=0.999 ဆိုတာက translation length နှင့် reference length က နီးစပ်ကြောင်းကို ပြနေပြီး၊ translation က အလွန်တိုချိုးမနေကြောင်းကို အနည်းငယ်ပဲ ဆုံးဖြတ်ပေးတဲ့ အချက်ပါ။

## ဒီ repository မှာ ပြင်ဆင်ရမယ့် အပိုင်းများ

Scripts တွေကို အသုံးပြုနိုင်သော်လည်း လက်ရှိအခြေအနေမှာ ပိုမို portable ဖြစ်အောင် ပြင်ရမယ့် အပိုင်းအနည်းငယ်ရှိပါတယ်။

- Marian binary path ကို ~/marian/build/marian မှလွဲပြီး အခြားနေရာမှာ install လုပ်ထားရင် update လုပ်ရမယ်
- project folder ကို ရွှေ့လိုက်ရင် absolute data_path ကို ပြန်ပြင်ရမယ်
- multi-bleu.perl script ရှိမရှိနှင့် path ကို ပြင်ရမယ်
- GPU အခြေအနေအလိုက် --devices 0 ကို CPU သို့မဟုတ် အခြား GPU setting နဲ့ ပြင်ရမယ်

## ရလဒ်များနှင့် discussion

ဒီ workspace ထဲမှာ Marian ကို install လုပ်ပြီး full training run များကို ပြီးမြောက်အောင် လုပ်ထားပြီးဖြစ်တာကို log files နှင့် checkpoint files က သက်သေပြနေပါတယ်။ Validation log မှာ Seq2Seq မော်ဒယ်အတွက် best BLEU score ကို 73.56 အဖြစ် တွေ့ရပြီး Transformer မော်ဒယ်အတွက်က 76.10 အဖြစ် တွေ့ရပါတယ်။

Notebook မှာ test set ပေါ်မှာ evaluate လုပ်တဲ့အခါ Seq2Seq အတွက် BLEU = 74.20၊ Transformer အတွက် BLEU = 76.45 ရခဲ့ပါတယ်။

| Model | Best validation BLEU from training logs | Test BLEU from evaluation output |
| --- | ---: | ---: |
| Seq2Seq | 73.56 | 74.20 |
| Transformer | 76.10 | 76.45 |

ဤရလဒ်အရ Transformer မော်ဒယ်က Seq2Seq ထက် ပိုကောင်းသော BLEU score ကို ရရှိခဲ့ပြီး၊ ဒီ dataset ပေါ်မှာ Transformer က ပိုကောင်းစွာ generalize လုပ်နိုင်ကြောင်းကို ပြသပါတယ်။ အဓိကအချက်က training run တွေကို ဒီ repository ထဲမှာ ရှိပြီးသား checkpoints နှင့် logs တို့အပေါ် မူတည်ပြီး အမှန်တကယ်တင်ထားခဲ့တဲ့ score များကို အသုံးပြုထားခြင်းဖြစ်ပါတယ်။

## အဆုံးသတ်

ဒီ repository ကို Marian-based training နှင့် evaluation pipeline အဖြစ် နားလည်လွယ်အောင် documentation ပြုလုပ်ထားပြီး၊ လက်ရှိ workspace ထဲမှာ Marian install လုပ်ပြီး full training run များကို ပြီးမြောက်စေခဲ့တဲ့ checkpoints၊ logs နှင့် evaluation outputs များပါဝင်နေပါတယ်။ ထို့ကြောင့် ဤ repository ကို အခြားသူတစ်ဦးက review လုပ်မယ်ဆိုရင် အမှန်တကယ် run ပြီးသား ရလဒ်များကို အခြေခံပြီး သုံးသပ်နိုင်ပါတယ်။
