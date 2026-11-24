"""Exercise the complete Trainer on a tiny random model without downloads."""

import argparse
import json
import tempfile

import torch
from transformers import GPT2Config, GPT2LMHeadModel, TrainingArguments

from token_preference import ChosenDataCollator, PreferenceConfig, PreferenceTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--output-dir", default=None)
    options = parser.parse_args()
    if options.steps < 1:
        parser.error("--steps must be positive")
    torch.manual_seed(7)
    torch.set_num_threads(1)
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=64, n_positions=32, n_embd=32, n_layer=1, n_head=4,
        resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
        bos_token_id=1, eos_token_id=2, pad_token_id=0, use_cache=False,
    ))
    rows = []
    for index in range(8):
        ids = [1, 4 + index, 20, 21, 22, 23, 2]
        rows.append({
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "labels": [-100, -100, 20, 21, 22, 23, -100],
        })
    collator = ChosenDataCollator(pad_token_id=0)
    output = options.output_dir or tempfile.mkdtemp(prefix="token-preference-demo-")
    trainer = PreferenceTrainer(
        model=model,
        preference_config=PreferenceConfig(),
        args=TrainingArguments(
            output_dir=output, use_cpu=True, max_steps=options.steps,
            per_device_train_batch_size=2, gradient_accumulation_steps=2,
            learning_rate=1e-3, logging_steps=1, save_strategy="no",
            report_to="none", disable_tqdm=True, seed=7,
        ),
        train_dataset=rows,
        eval_dataset=rows,
        data_collator=collator,
    )
    batch = collator(rows[:2])
    print("Before:", json.dumps(trainer.inspect_batch(batch), sort_keys=True))
    trainer.train()
    print("After:", json.dumps(trainer.inspect_batch(batch), sort_keys=True))
    print("Evaluation:", json.dumps(trainer.evaluate(), sort_keys=True))
    trainer.save_model(output)
    print("Saved tiny test model and objective configuration to", output)
    print("This checks execution; it is not a model-quality benchmark.")


if __name__ == "__main__":
    main()
