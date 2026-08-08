# Assignment No. 6: Myanmar G2P & P2G SMT Pipeline
### (@Class-13 and 14)

>Prof: Dr. Ye Kyaw Thu (LU Lab)

>Student: Thein Kyaw Lwin

>Date: 17, June, 2026

---

## Final SMT Evaluation Results

Both grapheme->phoneme and phoneme->grapheme pipelines were executed successfully inside the customized Docker container environment, achieving the following BLEU scores:

| Task Direction | Source $\to$ Target | BLEU Score | n-gram Precisions (1 / 2 / 3 / 4-gram) | Length Ratio (BP / hyp / ref) |
| :--- | :---: | :---: | :---: | :---: |
| **G2P** | `graphemes (my)` $\to$ `phonemes (ph)` | **69.51** | `85.3 / 72.7 / 64.5 / 58.4` | `BP=1.000 (ratio=1.000, 8049/8048)` |
| **P2G** | `phonemes (ph)` $\to$ `graphemes (my)` | **78.09** | `87.8 / 79.3 / 75.0 / 71.3` | `BP=1.000 (ratio=1.000, 8046/8047)` |


---

## Quick Replication Guide

To replicate these results using the pre-compiled binaries and dynamic execution scripts, follow these steps from your host terminal:

### 1. Build the Docker Image
Build the customized environment containing all runtime libraries:
```bash
docker build -f moses-smt.Dockerfile -t moses-smt:latest .
```

### 2. Start the Docker Container
Launch the container by mounting your current directory (`$(pwd)`) to `/workspace`:
```bash
docker run -it --rm --name moses-smt -v "$(pwd)":/workspace moses-smt:latest bash
```
*(ဘာ lib မှ ထပ် install စရာ မလိုတော့ပါဘူး။ compiled ပြီးသား binary file တွေက mounted workspace directory ထဲမှာ ရှိနေပြီးသားမို့ တိုက်ရိုက်ခေါ်သုံးပြီး လုပ်သွားလို့ရပါပြီ)*

### 3. Run the SMT Pipelines (inside the Docker container)

* **Grapheme $\to$ Phoneme (G2P)**:
  ```bash
  /workspace/tools/run_my-ph.sh
  ```
* **Phoneme $\to$ Grapheme (P2G)**:
  ```bash
  /workspace/tools/run_ph-my.sh
  ```

---

## SMT Pipeline Step-by-Step Execution Flow
`run_my-ph.sh` နှင့် `run_ph-my.sh` master shell script တွေထဲမှာ အောက်ကအတိုင်း အဆင့်ဆင့် လုပ်သွားမှာဖြစ်ပါတယ် -

1. **Step 1: Workspace & SGM Setup**:
   Re-initializes clean directories, copies raw data, and filters the corpus. Runs SGM scripts inside `exp/scripts/` to generate `test.<lang>.ref.sgm` and `test.<lang>.src.sgm` test structures.
2. **Step 2: Target-Side Language Modeling (KenLM)**:
   * **G2P**: Targets phonemes (`clean-train.ph`).
   * **P2G**: Targets graphemes (`clean-train.my`).
   * Binarization: Runs `build_binary` to convert the `.arpa` language model into a fast loading `.blm` binary format.
3. **Step 3: Word Alignment & SMT Translation Training**:
   Runs Moses `train-model.perl` with `-alignment grow-diag-final-and`, `-reordering msd-bidirectional-fe`, and uses multi-threaded **MGIZA** (`-mgiza -mgiza-cpus 4`) to extract grapheme-to-phoneme alignments.
4. **Step 4: MERT Tuning (Minimum Error Rate Training)**:
   Runs `mert-moses.pl` to optimize log-linear feature weights iteratively on the development set.
5. **Step 5: Decoding & BLEU Evaluation**:
   Translates the test dataset using the tuned weights and evaluates translation quality using `multi-bleu.perl`.

---

## Detailed Build & Compilation History
*(ဒီ assignment အတွက် linux-based moses environment တစ်ခုလုံးကို ဘယ်လိုလုပ်ခဲ့တယ်ဆိုတာ မှတ်တမ်းတင်ထားပါတယ်)*
*Google Antigravity IDE ကို သုံးပြီး Gemini 3.5 Flash ရဲ့ အကူအညီနဲ့ မေးရင်းမြန်းရင်း စမ်းကြည့်ရင်း လုပ်ခဲ့တာဖြစ်ပါတယ်*
>စမ်းလို့ အဆင်ပြေသွားတော့မှ Dockerfile သေချာရေးပြီးတော့ reproducible ဖြစ်အောင် scripts တွေကို ပြန် update လုပ်ပေးထားတာကြောင့် `Quick Replication Guide` ကို ကြည့်ပြီး အဆင်သင့်လိုက်လုပ်လို့ရသွားမှာပါ။

### 1. Compile Moses Decoder
အရင်ဆုံး linux container ပေါ်မှာ moses ကို compile ဖို့ ကြိုးစားပါတယ်
```bash
docker run -it --rm --name moses-smt -v "$(pwd)":/workspace my-ubuntu:latest bash
```
container ထဲမှာ moses ရဲ့ documentation ထဲက အတိုင်း လိုအပ်တဲ့ libraries များ install ပါတယ်-
```bash
apt-get update && apt-get install -y \
  build-essential git subversion pkg-config automake autoconf libtool \
  wget cmake make zlib1g-dev libbz2-dev liblzma-dev libboost-all-dev \
  libicu-dev libgoogle-perftools-dev python3-dev doxygen perl \
  libxml-twig-perl libsort-naturally-perl
```
နောက်ပိုင်း အဆင်သင့်သုံးလို့ရအောင် `/workspace/tools/` ထဲမှာ binary file တွေ သိမ်းခဲ့ပါတယ်
```bash
mkdir -p /workspace/tools
cd /workspace/tools
git clone https://github.com/moses-smt/mosesdecoder.git
```
ပြီးရင်တော့ compile လုပ်ခဲ့ပါတယ် (၁ နာရီ နီးပါး ကြာပါတယ်)
```bash
cd /workspace/tools/mosesdecoder
./bjam -j4
```
ပြီးရင် moses version စစ်လို့ရပါပြီ
```bash
/workspace/tools/mosesdecoder/bin/moses --version
# Moses code version (git tag or commit hash): mmt-mvp-v0.12.1-3050-g08e782040
# Libraries used: Boost version 1.74.0
```

### 2. Compile MGIZA (Word Aligner)
နောက်ပြီး MGIZA ကို compile ပါတယ်
```bash
git clone https://github.com/moses-smt/mgiza.git /workspace/tools/mgiza
cd /workspace/tools/mgiza/mgizapp
cmake .
make -j4
```
ပြီးတော့ mgiza binary file တွေကို training-tools directory ထဲကို ရွှေ့လိုက်ပါတယ်။ အဲ့အတွက် shell script ရေးပြီး run ခဲ့ပါတယ်။
```bash
/workspace/tools/setup_mgiza.sh
```

### 3. Space & Size Optimization (Stripping Down)
အိမ်စာတင်ရင် file size ကြီးပြီး GitHub push မရတာမျိုးမဖြစ်အောင် binary runtime များအတွက် မလိုအပ်တဲ့ source code တွေနဲ့ intermediate `.o` file များကို safe cleanup လုပ်ခဲ့ပါတယ်
* **Removed**: `tools/mgiza/` source, `tools/mosesdecoder/.git` history, and all compiler folders (`moses/`, `mert/`, `lm/`, `phrase-extract/`, `lib/`, `util/`, etc.).
* **Retained**: Only compiled executable files in `tools/mosesdecoder/bin/` and perl/python running scripts in `tools/mosesdecoder/scripts/`.
* **Result**: Repository size was reduced from **953 MB** to under **200 MB** (79% space saved) while remaining fully functional.

အဲ့လို stripped down လုပ်ပြီးတော့မှ pipeline တစ်ခုလုံး အစ-အဆုံး ပြန် run ကြည့်တာလည်း အလုပ်လုပ်ပါတယ်။

---

## Container Spec Verification (`fastfetch`)
```
demo@container:/workspace$ fastfetch
                             ....              root@91ed942fe2ba
              .',:clooo:  .:looooo:.           -----------------
           .;looooooooc  .oooooooooo'          OS: Ubuntu 22.04.5 LTS (Jammy Jellyfish) aarch64
        .;looooool:,''.  :ooooooooooc          Host: Apple Virtualization Generic Platform (1)
       ;looool;.         'oooooooooo,          Kernel: Linux 6.8.0-100-generic
      ;clool'             .cooooooc.  ,,       Uptime: 7 hours, 33 mins
         ...                ......  .:oo,      Packages: 568 (dpkg)
  .;clol:,.                        .loooo'     Shell: bash 5.1.16
 :ooooooooo,                        'ooool     Terminal: xterm
'ooooooooooo.                        loooo.    Terminal Font: fixed (8.0pt)
'ooooooooool                         coooo.    CPU: Virtualized Apple Silicon (4)
 ,loooooooc.                        .loooo.    Memory: 407.89 MiB / 3.81 GiB (10%)
   .,;;;'.                          ;ooooc     Swap: Disabled
       ...                         ,ooool.     Disk (/): 2.53 GiB / 97.87 GiB (3%) - overlay
    .cooooc.              ..',,'.  .cooo.      Local IP (eth0): 172.17.0.2/16
      ;ooooo:.           ;oooooooc.  :l.       Locale: C
       .coooooc,..      coooooooooo.
         .:ooooooolc:. .ooooooooooo'
           .':loooooo;  ,oooooooooc
               ..';::c'  .;loooo:'
```