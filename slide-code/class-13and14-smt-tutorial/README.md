# SMT Tutorial

Class 13 နဲ့ 14 မှာ SMT ကို presentation slide နဲ့ ရှင်းပြတာကော၊ လက်တွေ့ run ပြတာကော ဆရာ လုပ်ပြခဲ့ပါတယ်။  
အတန်းထဲမှာ သုံးခဲ့တဲ့ Jupyter notebook တွေအပြင် ဒေတာအားလုံး (ထိုင်း-မြန်မာ parallel data ခြောက်သောင်းကျော်) ကို သုံးပြီးတော့ လက်တွေ့ ထိုင်း-မြန်မာ၊ မြန်မာ-ထိုင်း phrase-based SMT (PBSMT) လုပ်ထားတဲ့ notebook တွေကိုလည်း အသေးစိတ် လေ့လာလို့ ရအောင် တင်ပေးထားပါတယ်။ လက်တွေ့ SMT လုပ်တဲ့အခါမှာ အရေးကြီးတဲ့ data cleaning အပိုင်းကိုလည်း syl_normalizer.py (ver 0.6) ကို သုံးပြီး လုပ်ပြထားပါတယ်။  

## Assignment No. 6
### (@Class-13 and 14)

Automatic Speech Recognition (ASR), Text to Speech (TTS) အလုပ်တွေအတွက် အရေးကြီးတဲ့ grapheme2phoneme ပြောင်းတဲ့ အပိုင်းကို အတန်းထဲမှာ သင်ထားတဲ့ Statistical Machine Translation ကို သုံးပြီး ရလဒ်ကို ကောင်းအောင် ကြိုးစားကြည့်ပါ။  

### Data Information

Data က myG2P corpus ကနေ ယူထားတာပါ။  
Link က [https://github.com/ye-kyaw-thu/myG2P](https://github.com/ye-kyaw-thu/myG2P)  

Original corpus format က အောက်ပါအတိုင်း ရှိတယ်။ အဲဒီကနေ field 3 နဲ့ 4 ကို ဆွဲထုတ်ယူလိုက်ပြီး parallel corpus ဆောက်ပေးထားတယ်။  

```
$ head ./myg2p.ver2.0.txt
1       ...ဖြစ်စေ...ဖြစ်စေ      ... ဖြစ် စေ ... ဖြစ် စေ ... hpji' sei ... hpji' sei     ... pʰjɪʔ sè ... pʰjɪʔ sè
2       ...ရိုး...စဉ်   ... ရိုး ... စဉ်        ... jou: ... sin        ... jó ... sɪ̀ɴ
3       ...ရိုး...စဉ်   ... ရိုး ... စဉ်        ... jou: ... zin        ... jó ... zɪ̀ɴ
4       ...လို...ငြား   ... လို ... ငြား        ... lou ... nja:        ... lò ... ɲá
5       ကကတစ်   က က တစ် ka. ga- di'     ka̰ gə dɪʔ
6       ကကတိုး  က က တိုး        ka. ga- dou:    ka̰ gə dó
7       ကကုသန်  က ကု သန်        ka. ku. than    ka̰ kṵ θàɴ
8       ကကုသန်  က ကု သန်        kau' ka- than   kaʊʔ kə θàɴ
9       ကကူရံ   က ကူ ရံ ka. ku jan      ka̰ kù jàɴ
10      ကကြိုး  က ကြိုး ka. gyou:       ka̰ dʑó
```

### Source  

Machine Translation လုပ်ဖို့အတွက် Source နဲ့ target ဒေတာတွေက အတွဲလိုက် shuffle လုပ်ထားပြီးသားပါ။   

```
$ wc *.my
  2000   5687  59222 dev.my
  2802   8047  83959 test.my
 20000  57336 594183 train.my
 24802  71070 737364 total
```

```
$ head -n 5 *.my
==> dev.my <==
ရပ် မှု ရွာ မှု
ပြီး ဆေး
မြင်း စား ဂျုံ
အ ငံ့ ပေး
အ နေ အ ထား

==> test.my <==
တက် တက် ပြောင်
ကပ် ပိ
ရှုံ့ မဲ့
ညှဉ်း ပန်း
မွမ်း မံ

==> train.my <==
ပြာ သာဒ် ဆောင်
ကိုယ် ပိုင် စာ ကြည့် တိုက်
သိန် ဓော ဆား
စား သောက် ဆိုင်
ကြိုး စင်
```

### Target  

```
ye@lst-hpc3090:/mnt/disk1/ye/data/zg-un/myg2p/uni/src-tgt/g2p-par$ wc *.ph
  2000   5688  25849 dev.ph
  2802   8048  36532 test.ph
 20000  57346 260356 train.ph
 24802  71082 322737 total
ye@lst-hpc3090:/mnt/disk1/ye/data/zg-un/myg2p/uni/src-tgt/g2p-par$
```

```
$ head -n 5 *.ph
==> dev.ph <==
ja' mhu. jwa mhu.
pi: zei:
mjin: za: gyoun
a- ngan pei:
a- nei a- hta:

==> test.ph <==
te' te' pjaun
ka' pi.
shoun. me.
njhin: ban:
mun: man

==> train.ph <==
pja' tha' hsaun
kou bain sa kyi. dai'
thein: do: hsa:
sa: thau' hsain
kyou: zin
```

အခု ဆရာပြင်ပေးထားတဲ့ ဒေတာကို သုံးပြီး Phrase-based SMT ကို ကိုယ့်စက်ထဲမှာ run ကြည့်ပါ။ ထွက်လာတဲ့ baseline ရလဒ်ကိုမှ ပိုကောင်းအောင် စဉ်းစားပြီး BLEU score ကို တင်ကြည့်ပါ။  
အနည်းဆုံးတော့ SMT ရဲ့ baseline ကိုတော့ ကိုယ့်ဖာသာကိုယ် ရအောင်ထုတ်ကြည့်ပါ။  
Assignment တင်တဲ့အခါမှာ perl script တွေ ပြင်တာ၊ test data ကို SGM format ပြောင်းတာ၊ configuration file ကို my-ph, ph-my အတွက် ထုတ်တာ၊ SMT training/tuning/evaluation စတာတွေ လုပ်ထားတဲ့ Jupyter notebook ကို တင်ကြပါ။  

