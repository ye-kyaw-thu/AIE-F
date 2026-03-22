## Hybrid-ELIZA (BiLSTM + Rule-based Engine)
### Project Summary & Model Evaluation

#### ၁။ Project အကျဉ်းချုပ် (Summary)
**Hybrid ELIZA** သည် Rule-based Engine (ELIZA Pattern Matching) ၏ တိကျမြန်ဆန်မှု နှင့် Deep Learning (BiLSTM) ၏ ဘာသာစကား နားလည်နိုင်စွမ်းတို့ကို ပေါင်းစပ်ထားသော မြန်မာဘာသာစကား Emotion Classification စနစ် တစ်ခုဖြစ်ပါသည်။ 

---

#### ၂။ Model စွမ်းဆောင်ရည် စမ်းသပ်ခြင်း (Evaluation Report)
AI Model ကြီးအနေဖြင့် တစ်ခါမှ မမြင်ဖူးသော **Test Data (၅၈၆) ကြောင်း** ပေါ်တွင် လုံးဝ အသစ်စမ်းသပ် (Evaluate) ခဲ့ရာ အောက်ပါအတိုင်း အလွန် ကောင်းမွန်သော ရလဒ်များကို ရရှိခဲ့ပါသည် -

```text
--- BiLSTM Model Evaluation Results ---
              precision    recall  f1-score   support

    ဝမ်းနည်း       0.76      0.74      0.75       100
  ပျော်ရွှင်       0.86      0.73      0.79       100
   ချစ်ခြင်း       0.82      0.91      0.86       100
        ဒေါသ       0.78      0.84      0.81        95
  ကြောက်ရွံ့       0.94      0.95      0.95       100
        အံ့ဩ       0.91      0.90      0.91        91

    accuracy                           0.84       586
   macro avg       0.85      0.85      0.84       586
weighted avg       0.85      0.84      0.84       586
```
