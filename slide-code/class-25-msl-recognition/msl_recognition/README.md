# MSL Recognition Tutorial

## Jupyter Notebooks

1. [MSL-Video-Recognition-Tutorial.ipynb](https://github.com/ye-kyaw-thu/AIE-F/blob/main/slide-code/class-25/msl_recognition/MSL-Video-Recognition-Tutorial.ipynb)
   ဒီ notebook က အတန်းထဲမှာ ဆရာ အစအဆုံး ၂နာရီကျော် အချိန်ယူပြီး run ပြခဲ့တဲ့ notebook ပါ။   
2. [Single-Video-Inference.ipynb](https://github.com/ye-kyaw-thu/AIE-F/blob/main/slide-code/class-25/msl_recognition/Single-Video-Inference.ipynb)
   ဒီ notebook ကတော့ ဆောက်ထားပြီးသား မော်ဒယ် သုံးမျိုးကို သုံးပြီး ဗီဒီယိုဖိုင် တစ်ဖိုင်ချင်းစီရော၊ ကိုယ် test လုပ်ချင်တဲ့ ဗီဒီယိုဖိုင်တွေကို ဖိုလ်ဒါတစ်ခုအောက်မှာ စုသိမ်းထားပြီး batch အလိုက် recognition လုပ်တာကိုကော ဒီမို လုပ်ပြထားတဲ့ notebook ပါ။
3. [Python_Codes_for_MSL_Recognition.ipynb](https://github.com/ye-kyaw-thu/AIE-F/blob/main/slide-code/class-25/msl_recognition/Python_Codes_for_MSL_Recognition.ipynb)
   ဒီ notebook ကတော့ python code အကုန်လုံးနီးပါးကို တစုတစည်းတည်း notebook ပေါ်ကို တင်ပေးထားတာပါ။
4. [Export-to-ONNX-and-Infer-Experiment.ipynb](https://github.com/ye-kyaw-thu/AIE-F/blob/main/slide-code/class-25/msl_recognition/Export-to-ONNX-and-Infer-Experiment.ipynb)
   ဒီ နံပါတ် ၄ notebook ကတော့ Linux ဆာဗာပေါ်မှာ ဆောက်ထားတဲ့ မော်ဒယ်တွေကို Windows OS ပေါ်မှာ ခေါ်သုံးနိုင်အောင်လို့ ONNX export လုပ်ပေးတဲ့ code နဲ့ infer လုပ်တဲ့ code တွေကိုရေးနေတဲ့ အချိန်မှာ debugging log လုပ်ခဲ့တာတွေကို မှတ်တမ်းတင်ထားတာပါ။ infer_onnx.py က ဆရာတို့ debugging လုပ်ဖို့ ကျန်နေပါသေးတယ်။ အရင်ရေးထားတဲ့ code တွေကို Claude ကိုသုံးပြီး ပြန် update/debugg လုပ်တဲ့အခါမှာ ဘယ်လိုမျိုး အမှားတွေ ရှိတယ်ဆိုတဲ့ လက်တွေ့အင်ဂျင်နီယာအပိုင်းကိုလည်း မြင်အောင်လို့ပါ။

## Scripts

scripts ဖိုလ်ဒါအောက်မှာက experiment လုပ်တဲ့အခါမှာ တစ်ခါထက်မက run ကြရတာမို့ အဆင်ပြေအောင် ပြင်ဆင်ထားတဲ့ shell script တွေပါ။ ဒေတာကို ပြင်ဆင်ပြီး၊ နံပါတ် အစီအစဉ်လိုက် အဆင့်ဆင့် run တာကို ဆရာ အတန်းထဲမှာ ပြပြီးသားပါ။  

## Src

src ဖိုလ်ဒါအောက်မှာကတော့ Python code တွေကို သိမ်းထားတာပါ။ လက်ရှိဗားရှင်းတွေကို တင်ပေးထားလိုက်ပါတယ်။ infer_onnx.py က ဆရာ့ Windows OS မှာ run လို့ အဆင်မပြေသေးပါဘူး။ debug လုပ်နေတုန်းပါ။ အဲဒီအပိုင်းကို debug လုပ်နိုင်တဲ့သူတွေကလည်း လုပ်ကြည့်ကြပါ။  

## Config

config ဖိုလ်ဒါအောက်မှာက experiment အတွက် သုံးခဲ့တဲ့ configuration ဖိုင်ကို သိမ်းထားပါတယ်။  

## tools

tools ဖိုလ်ဒါထဲက ဖိုင်တွေက ဆရာ အတန်းထဲမှာ run မပြခဲ့ပါဘူး။ Server အသစ်ပေါ်ကို code တွေရွှေ့လာပြီး အရင်ရေးထားတဲ့ ဗားရှင်းတွေကို ပြန်ပြင်တဲ့အခါမှာ mediapipe library import path ကို စစ်ဆေးဖို့ ရေးခဲ့တာပါ။ lookup_video_label.py ကတော့ data preparation လုပ်တဲ့အခါမှာ လေဘယ်နဲ့ ဗီဒီယိုနဲ့ ကိုက်ရဲ့လား ဆိုတဲ့ အပိုင်းနဲ့ ဆိုင်တဲ့ ကုဒ်ပါ။  

## Requirements

requirements.txt ဖိုင်ကတော့ လိုအပ်တဲ့ python library တွေကို installation လုပ်တဲ့အခါမှာ အသုံးဝင်ပါလိမ့်မယ်။  

## Data

data ဖိုလ်ဒါအောက်မှာ videos ဆိုတဲ့ ဖိုလ်ဒါအသစ်ဆောက်ပြီး MSL4Emergency ဗီဒီယိုဖိုင်တွေကို ဒေါင်းလုဒ်လုပ်ထားပါ။ annotations.txt ဖိုင်ကိုတော့ ဆရာ ပြင်ပေးထားပါတယ်။  

## Assignment Information

- Video recognition ကို ကိုယ့်စက်ထဲမှာ ဖြစ်ဖြစ်၊ Colab မှာဖြစ်ဖြစ် run ကြည့်ပါ။ 
- အရင်ဆုံး requirements.txt ဖိုင်ကို အခြေခံပြီး လိုအပ်တဲ့ Python environment ကို ဆောက်ပါ။
- ဆရာ တင်ပေးထားတဲ့ Python နဲ့ shell script တွေကို src/ နဲ့ scripts/ ဖိုလ်ဒါအောက်မှာ သိမ်းပါ။
- data/ ဖိုလ်ဒါ ကိုပြင်ဆင်ပြီး [MSL4Emergency](https://github.com/ye-kyaw-thu/MSL4Emergency) repository ကနေ ဗီဒီယိုဖိုင်တွေကို download လုပ်ယူပြီး ပြင်ဆင်ပါ။
- config/ ဖိုလ်ဒါအောက်မှာ config.yaml ဖိုင်ကိုလည်း ပြင်ဆင်ထားဖို့ လိုအပ်ပါလိမ့်မယ်။
- ပြီးရင်တော့ script/ ဖိုလ်ဒါအောက်ထဲက shell script တွေကို နံပါတ်စဉ်အတိုင်း အဆင့်ဆင့် run သွားရင် အဆင်ပြေပါလိမ့်မယ်။
- ဒီ Assignment မှာ အဓိက အပြောင်းအလဲ လုပ်ကြည့်စေချင်တာက annotations.txt ဖိုင်ထဲက ကော်လံနံပါတ် (၂) MSL gloss ကို သုံးပြီး video recognition လုပ်ကြည့်စေချင်တာပါ။ ရလဒ်တွေနဲ့တကွ run ထားတဲ့ Jupyter Notebook ကို ဆရာ့ဆီကို အီးမေးလ်နဲ့ ပို့ပေးတာ ဖြစ်ဖြစ်၊ ဒါမှမဟုတ် ဆရာ ပြင်ဆင်ပေးထားတဲ့ GitHub Repository [assignment-submission/class-25/](https://github.com/ye-kyaw-thu/AIE-F/tree/main/assignment-submission/class-25) အောက်မှာ Pull request လုပ်ထားပေးကြပါ။    

### pdf

- လေ့လာရတဲ့အခါမှာ လွယ်ကူအောင်လို့ Jupyter notebook တွေကို PDF ဖိုင်အဖြစ်လည်း ပြောင်းပေးထားပါတယ်။  

