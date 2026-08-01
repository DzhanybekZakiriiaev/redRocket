"""LoRA supervised fine-tune -- distill the Bayes filter into a small student LLM.

    python tools/distill_train.py \
        --train out/distill_train.jsonl \
        --val   out/distill_val.jsonl \
        --base  google/gemma-2-2b-it \
        --out   out/distill_adapter

This trains a small open base model to imitate the optimal Bayes filter using the
chat-format JSONL emitted by `tools/distill_dataset.py`. Each row is a two-turn
conversation: a natural-language evidence prompt (identical to what
`sim/advisor/gemma.py` sends at inference time) and the JSON label the Bayes
filter concluded. The model learns the prompt -> label mapping; at deploy time it
emits the same JSON gemma.py already knows how to parse.

WHY LoRA. We only need to teach a narrow input->output mapping, not repair the
base model's general knowledge. LoRA freezes the base weights and trains a few
million adapter parameters, so this fits on a single consumer/Colab GPU (a T4 is
enough for a 2B base in 4-bit) and produces a tiny (~10-50 MB) adapter you can
ship or merge.

DELIBERATELY IMPORTS torch / transformers / peft / trl INSIDE __main__ so this
file imports cleanly on the dataset-generation box (which has neither). It does
NOT need to run in the local repo venv -- run it on the GPU box per
tools/DISTILL_README.md. Heavily commented so it is auditable, not magic.
"""

from __future__ import annotations

import argparse


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LoRA SFT: distill the Bayes filter into a small LLM.")
    # --- data ---
    p.add_argument("--train", default="out/distill_train.jsonl",
                   help="training JSONL (chat format from distill_dataset.py)")
    p.add_argument("--val", default="out/distill_val.jsonl",
                   help="validation JSONL (optional but recommended)")
    # --- model ---
    # Any small instruction-tuned base with a chat template works. Good picks:
    #   google/gemma-2-2b-it   (matches the 'gemma' framing; needs HF license accept)
    #   Qwen/Qwen2.5-1.5B-Instruct   (permissive, no gate, very small)
    #   meta-llama/Llama-3.2-1B-Instruct
    p.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="HF model id of the base to fine-tune")
    p.add_argument("--out", default="out/distill_adapter",
                   help="directory to write the LoRA adapter + tokenizer")
    # --- training schedule ---
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=8,
                   help="per-device train batch size (lower if you hit OOM)")
    p.add_argument("--grad-accum", type=int, default=2,
                   help="gradient accumulation steps (effective batch = bs*accum)")
    p.add_argument("--lr", type=float, default=2e-4,
                   help="learning rate -- LoRA tolerates a higher LR than full FT")
    p.add_argument("--max-seq-len", type=int, default=1024,
                   help="the evidence prompt is ~350 tokens; 1024 is ample")
    # --- LoRA hyperparameters ---
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    # --- quantization ---
    p.add_argument("--load-4bit", action="store_true",
                   help="QLoRA: load the base in 4-bit (fits a 2B model on a T4)")
    p.add_argument("--merge", action="store_true",
                   help="also write a merged full-precision model next to the adapter")
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    # -- Heavy deps imported HERE so the file stays importable without them. --
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    # ------------------------------------------------------------------ data
    # trl's SFTTrainer understands the {"messages": [...]} chat schema natively:
    # it applies the tokenizer's chat template to each row, so the model trains
    # on exactly the wire format it will see at inference.
    data_files = {"train": args.train}
    if args.val:
        data_files["validation"] = args.val
    ds = load_dataset("json", data_files=data_files)
    print(f"[train] loaded {len(ds['train'])} train rows"
          + (f", {len(ds['validation'])} val rows" if args.val else ""))

    # -------------------------------------------------------------- tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        # Causal LMs often ship without a pad token; reuse EOS so batching works.
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------ model
    quant_config = None
    if args.load_4bit:
        # QLoRA: 4-bit NF4 base weights + a bf16 compute dtype. This is what lets
        # a 2B (or even 7B) base fine-tune inside a single 16 GB GPU / free Colab.
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing

    # -------------------------------------------------------------- LoRA spec
    # Target the attention + MLP projection matrices -- the standard, portable
    # choice that works across Llama/Qwen/Gemma-style architectures. If a given
    # base names its modules differently, transformers/peft will raise and tell
    # you which names to use.
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    # ------------------------------------------------------------- trainer cfg
    sft_config = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_seq_len,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if args.val else "no",
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        seed=args.seed,
        # We only want the loss computed on the assistant's JSON answer, not on
        # the (long, fixed) evidence prompt -- otherwise the model spends its
        # gradient budget memorizing the cause menu instead of the mapping.
        assistant_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    # ------------------------------------------------------------------ train
    trainer.train()

    # The adapter (a few MB) + tokenizer is the deployable artifact.
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"[train] LoRA adapter written to {args.out}")

    # -------------------------------------------------- optional merged model
    if args.merge:
        # Fold the adapter into the base weights for a standalone model that any
        # OpenAI-compatible server (vLLM, TGI) can load without knowing about
        # PEFT. Skip when using 4-bit -- merge into a fresh fp16 base instead.
        from peft import PeftModel
        merged_dir = args.out.rstrip("/") + "_merged"
        base = AutoModelForCausalLM.from_pretrained(
            args.base, torch_dtype=torch.bfloat16, device_map="auto")
        merged = PeftModel.from_pretrained(base, args.out).merge_and_unload()
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"[train] merged full model written to {merged_dir}")


if __name__ == "__main__":
    main()
