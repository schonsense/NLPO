"""Train a compatible native HF causal model on a local JSON/JSONL dataset.

For custom architectures or preconfigured adapters, use the Python API with
your already-loaded model. This generic example does not modify architectures.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from token_preference import (
    ChosenDataCollator, PreferenceConfig, PreferenceTrainer, prepare_dataset,
)


def read_rows(path):
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, list):
        raise ValueError("JSON input must be a list of records; JSONL is also supported.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local HF directory or Hub model ID")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--probability-floor", type=float, default=.05)
    parser.add_argument("--alternative-ratio", type=float, default=3.)
    parser.add_argument("--negative-weight", type=float, default=1.)
    parser.add_argument("--floor-weight", type=float, default=1.)
    parser.add_argument("--loss-chunk-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--epochs", type=float, default=1.)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attn-implementation", choices=["sdpa", "flash_attention_2"], default="sdpa")
    parser.add_argument("--resume", default=None, help="A checkpoint directory created by this trainer")
    options = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("This model-loading example requires CUDA; use tiny_cpu.py for a CPU test.")
    if not torch.cuda.is_bf16_supported():
        parser.error("This example requires BF16 support. Use the Python API for another precision.")
    # This setting covers the forward and checkpoint recomputation during
    # backward. Unsupported fused configurations fail instead of using math SDPA.
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(True)
    tokenizer = AutoTokenizer.from_pretrained(
        options.model, use_fast=True, trust_remote_code=options.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Set a valid tokenizer pad token before using this loader.")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        options.model,
        torch_dtype=torch.bfloat16,
        attn_implementation=options.attn_implementation,
        trust_remote_code=options.trust_remote_code,
    )
    dataset = prepare_dataset(read_rows(options.data), tokenizer, max_length=options.max_length)
    trainer = PreferenceTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        data_collator=ChosenDataCollator(tokenizer=tokenizer),
        preference_config=PreferenceConfig(
            probability_floor=options.probability_floor,
            alternative_ratio=options.alternative_ratio,
            negative_weight=options.negative_weight,
            floor_weight=options.floor_weight,
            chunk_size=options.loss_chunk_size,
        ),
        args=TrainingArguments(
            output_dir=options.output_dir,
            per_device_train_batch_size=options.batch_size,
            gradient_accumulation_steps=options.gradient_accumulation_steps,
            num_train_epochs=options.epochs,
            learning_rate=options.learning_rate,
            bf16=True,
            gradient_checkpointing=options.gradient_checkpointing,
            average_tokens_across_devices=True,
            report_to="none",
            logging_steps=10,
            save_strategy="epoch",
        ),
    )
    trainer.train(resume_from_checkpoint=options.resume)
    trainer.save_model(options.output_dir)


if __name__ == "__main__":
    main()
