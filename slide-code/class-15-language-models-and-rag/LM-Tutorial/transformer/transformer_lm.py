#!/usr/bin/env python3
"""
Transformer-based Language Model using Hugging Face.
Defaults to fine-tuning XGLM (a GPT-style autoregressive model). 
Note: Standard GPT-2 is NOT used by default because its tokenizer lacks Myanmar 
subwords, forcing it to split Myanmar text into inefficient UTF-8 bytes. 
XGLM uses a 256k SentencePiece vocabulary that natively includes Myanmar script.
"""

import os
import math
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

# ==========================================
# 1. DATA UTILITIES
# ==========================================
def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def tokenize_dataset(lines, tokenizer, max_length):
    """Safely tokenizes and chunks text using HuggingFace best practices."""
    from datasets import Dataset
    
    # 1. Create a raw dataset
    raw_dataset = Dataset.from_dict({"text": lines})
    
    # 2. Tokenize without truncation first
    def tokenize_function(examples):
        return tokenizer(examples["text"])
    
    tokenized_dataset = raw_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"],
        desc="Tokenizing"
    )
    
    # 3. Group into fixed-length blocks (prevents the massive tensor warning)
    def group_texts(examples):
        # Concatenate all token lists
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = (len(concatenated_examples["input_ids"]) // max_length) * max_length
        
        # Split by max_length
        result = {
            k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
            for k, t in concatenated_examples.items()  # <--- FIX IS HERE
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        desc="Grouping into chunks"
    )
    return lm_dataset

# ==========================================
# 2. STRIDED PERPLEXITY EVALUATOR
# ==========================================
def evaluate_ppl(args, model, tokenizer, test_lines):
    """Calculates exact strided Perplexity, identical to HF advanced documentation."""
    print(f"Tokenizing test set ({len(test_lines)} lines)...")
    encodings = tokenizer("\n\n".join(test_lines), return_tensors="pt").to(model.device)
    
    max_length = args.seq_len
    stride = args.stride
    seq_len = encodings.input_ids.size(1)
    
    nll_sum, n_tokens = 0.0, 0
    prev_end_loc = 0
    
    print(f"Evaluating with Stride={stride}, Context={max_length}...")
    with torch.no_grad():
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc

            input_ids = encodings.input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100 # Ignore context tokens in loss

            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss.item()

            num_valid_tokens = (target_ids != -100).sum().item()
            num_loss_tokens = num_valid_tokens - target_ids.size(0) # HF shifts labels internally

            nll_sum += neg_log_likelihood * num_loss_tokens
            n_tokens += num_loss_tokens
            prev_end_loc = end_loc
            
            if end_loc == seq_len: break

    avg_nll = nll_sum / n_tokens
    ppl = math.exp(avg_nll)
    return ppl, avg_nll

# ==========================================
# 3. MODES
# ==========================================
def train_model(args):
    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load model in bfloat16 for RTX 3090 efficiency
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, 
        torch_dtype=torch.bfloat16
    )
    
    print(f"Loading training data: {args.train_file}")
    train_lines = load_text_file(args.train_file)
    train_dataset = tokenize_dataset(train_lines, tokenizer, args.seq_len)
    
    #data_collator = DataCollatorForLanguageModeling(mlm=False)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)  

    # Setup HuggingFace Trainer
    training_args = TrainingArguments(
        output_dir=args.model_path,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_strategy="epoch",
        logging_steps=10,
        bf16=True, # RTX 3090 Ampere architecture supports this natively
        gradient_accumulation_steps=2, # Helps with stability on small datasets
        learning_rate=args.lr,
        weight_decay=0.01,
        report_to="none", # Disable wandb/tensorboard for clean terminal output
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    print("Starting Transformer Fine-Tuning...")
    trainer.train()
    
    # Save final model
    trainer.save_model(args.model_path)
    tokenizer.save_pretrained(args.model_path)
    print(f"Model saved to {args.model_path}")

def test_model(args):
    print(f"Loading fine-tuned model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    
    test_lines = load_text_file(args.test_file)
    ppl, nll = evaluate_ppl(args, model, tokenizer, test_lines)
    print(f"\nTest Results -> PPL: {ppl:.2f}, Avg NLL: {nll:.4f} nats")

def generate_text(args):
    print(f"Loading fine-tuned model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    
    prompt = args.prompt if args.prompt else ""
    print(f"\nPrompt: {prompt}")
    print("Generating: ")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # HF Generate function handles sampling natively
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.gen_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)

# ==========================================
# 4. MAIN
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer LM (Hugging Face)")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test", "generate"])
    parser.add_argument("--train_file", type=str, help="Path to train text")
    parser.add_argument("--test_file", type=str, help="Path to test text")
    parser.add_argument("--model_path", type=str, default="./xglm_finetuned")
    parser.add_argument(
    "--base_model", 
    type=str, 
    default="facebook/xglm-564M", 
    help="Base model to fine-tune. Default is XGLM-564M (strong multilingual tokenizer). "
         "Do NOT use standard 'gpt2' for Myanmar as its tokenizer will shatter the text into bytes."
    )
    
    # Data/Model Args
    parser.add_argument("--seq_len", type=int, default=128, help="Max sequence length for training/chunking")
    parser.add_argument("--stride", type=int, default=64, help="Stride for PPL evaluation")
    
    # Training Args
    parser.add_argument("--epochs", type=int, default=3) # Transformers need fewer epochs than LSTMs!
    parser.add_argument("--batch_size", type=int, default=8) # Larger than LSTM due to efficiency
    parser.add_argument("--lr", type=float, default=5e-5) # Standard HF learning rate
    
    # Generation Args
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--gen_length", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.mode == "train" and not args.train_file: parser.error("--train_file required")
    if args.mode == "test" and not args.test_file: parser.error("--test_file required")
    
    if args.mode == "train": train_model(args)
    elif args.mode == "test": test_model(args)
    elif args.mode == "generate": generate_text(args)

