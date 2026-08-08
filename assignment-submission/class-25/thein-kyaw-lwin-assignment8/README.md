# Myanmar Sign Language Recognition

**Student:** Thein Kyaw Lwin  
**Assignment:** Myanmar Sign Language Recognition
**Instructor:** Dr. Ye Kyaw Thu, Language Understanding Lab., Myanmar  
**Platform:** Kaggle (2× Tesla T4 GPU, Ubuntu 22.04, CUDA 13.0)  
---

> assignment 8 အတွက် experiment run တာကို kaggle gpu နဲ့ run ခဲ့ပါတယ် ဆရာ။ column 1 ရော 2 ရော run လိုက်ပါတယ်ဆရာ။ file size တွေ ကြီးနေလို့ run ခဲ့တဲ့ notebook 2 ခုကိုပဲ တင်လိုက်ပါတယ် ဆရာ။ output တွေကို ကြည့်လို့ အဆင်ပြေအောင် kaggle notebook link တွေလည်း ဒီ Readme ထဲမှာ ထည့်ထားလိုက်ပါတယ် ဆရာ။

**Reference Notebooks:**

msl-col1-run-mm-text-labels
[![kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/theinkyawlwin/msl-col1-run-mm-text-labels)

### MSL Recognition Results — Label: Myanmar Text (Col 1)

| Model | Val Top-1 | Test Top-1 | Test F1 |
|---|---:|---:|---:|
| BiLSTM + Attention | 95.16% | 100.00% | 100.00% |
| Transformer Encoder | 96.77% | 100.00% | 100.00% |
| ST-GCN | 94.27% | 100.00% | 100.00% |

---

msl-col2-run-msl-gloss-labels
[![kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/theinkyawlwin/msl-col2-run-msl-gloss-labels)

### MSL Recognition Results — Label: MSL Gloss (Col 2)

| Model | Val Top-1 | Test Top-1 | Test F1 |
|---|---:|---:|---:|
| BiLSTM + Attention | 95.16% | 100.00% | 100.00% |
| Transformer Encoder | 96.95% | 100.00% | 100.00% |
| ST-GCN | 94.27% | 100.00% | 100.00% |

---
