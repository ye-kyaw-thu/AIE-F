## What is inside this project?

- g2p-par/: training, development, and test data for the source and target languages
- notebooks/: Jupyter notebooks for the Seq2Seq and Transformer experiments
- seq2seq.phmy.sh and transformer.phmy.sh: ready-to-run training scripts
- model.seq2seq.phmy/ and notebooks/model.transformer.phmy/: model checkpoints and config files
- logs/: training and validation logs for both systems

## How the workflow works

1. Prepare the vocabulary files from the combined training and development data.
2. Train a recurrent sequence-to-sequence model.
3. Decode the test set and compute BLEU score.
4. Train a Transformer model with the same data.
5. Compare the outputs and monitor the validation logs.

## Prerequisites

Before running the experiments, make sure the following are available:

- Marian NMT binaries such as marian, marian-vocab, and marian-decoder
- Perl for running BLEU evaluation
- A working multi-bleu script, usually named multi-bleu.perl

You can verify the Marian tools with:

```bash
marian --help
marian-vocab --help
marian-decoder --help
```

## Running the project

From the project root, run the shell scripts:

```bash
chmod +x seq2seq.phmy.sh transformer.phmy.sh
./seq2seq.phmy.sh
./transformer.phmy.sh
```

If you prefer to run the commands manually, the same steps are documented in the report and in the notebooks.

## What may need to be changed

The scripts are ready to use, but they currently contain a few machine-specific paths and settings:

- The Marian binary path is set to ~/marian/build/marian. If Marian is installed elsewhere, update that line.
- The data path is hard-coded to the current workspace location. If the folder is moved, update data_path in the shell scripts.
- The BLEU evaluation command uses a path to multi-bleu.perl. Update it to the actual location on your machine.
- The training commands use --devices 0. If you are on CPU or a different GPU, change this option accordingly.

## Expected outputs

After a successful run, you should see:

- vocabulary files under g2p-par/vocab/
- model checkpoints under model.seq2seq.phmy/ and notebooks/model.transformer.phmy/
- training logs under logs/
- decoded predictions such as seq2seq.phmy.hyp.txt and transformer.phmy.hyp.txt

## Results from the completed local runs

The workspace already contains completed training runs for both models. The validation logs and notebook evaluation output show the following results:

| Model | Best validation BLEU from training logs | Test BLEU from evaluation output |
| --- | ---: | ---: |
| Seq2Seq | 73.56 | 74.20 |
| Transformer | 76.10 | 76.45 |

These values come from the actual files in this repository, including [model.seq2seq.phmy/train.log](model.seq2seq.phmy/train.log), [model.seq2seq.phmy/valid.log](model.seq2seq.phmy/valid.log), [notebooks/model.transformer.phmy/valid.log](notebooks/model.transformer.phmy/valid.log), and the notebook evaluation cells that produced the BLEU outputs.
