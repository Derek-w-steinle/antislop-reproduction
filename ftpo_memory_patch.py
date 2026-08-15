"""Two memory fixes for auto-antislop's FTPO trainer.

Diagnosis (verified by reading the code, 2026-08-15): fine-tuning gemma-3-12b
at LoRA rank 256 OOM'd on a 141GB H200 using 139.6 GiB. Halving the rank to 128
did NOT help, which ruled out the adapters as the cause. Two things dominate:

FIX 1 - gradient checkpointing is silently disabled on the transformers path.
  `finetune_gradient_checkpointing` is read in exactly one place,
  core/finetuning.py:497, which sits INSIDE the `if use_unsloth:` branch. With
  finetune_use_unsloth: false (required, since unsloth breaks the install) the
  setting is dead code and checkpointing never turns on. For a 48-layer 12B at
  batch 3 x seq 2500 that is roughly 76 GiB of activations held for no reason.

FIX 2 - the forward computes 2500x more logits than the loss uses.
  ftpo_trainer.py builds logits for every position, then reads only the last:
      logits_last = outputs.logits[:, -1, :]
  `[:, -1, :]` is a view, so the whole [B, L, V] tensor stays alive. At
  3 x 2500 x 262208 in bf16 that is 3.66 GiB per tensor, and there are several
  live copies across the policy and reference passes. Passing logits_to_keep=1
  is mathematically identical (the slice still selects the final position) and
  removes nearly all of it.

Run from the auto-antislop repo root:  python3 ftpo_memory_patch.py
Idempotent; re-running is safe.
"""
import re
import sys

TRAINER = "core/ftpo_trainer.py"
FINETUNING = "core/finetuning.py"


def patch_logits():
    src = open(TRAINER).read()
    if "logits_to_keep" in src:
        print("  fix 2: already applied")
        return False
    n = src.count("return_dict=True,")
    if n != 3:
        sys.exit(f"  fix 2 ABORT: expected 3 forward calls, found {n}. "
                 "Repo has changed; re-check by hand.")
    # All three occurrences are model forward calls in compute_loss.
    src = src.replace("return_dict=True,", "return_dict=True, logits_to_keep=1,")
    open(TRAINER, "w").write(src)
    print(f"  fix 2: patched {n} forward calls with logits_to_keep=1")
    return True


def patch_checkpointing():
    src = open(FINETUNING).read()
    if "GRADIENT CHECKPOINTING PATCH" in src:
        print("  fix 1: already applied")
        return False
    anchor = "        model = get_peft_model(model, lora_cfg)\n"
    if anchor not in src:
        sys.exit("  fix 1 ABORT: could not find the transformers-path "
                 "get_peft_model call. Repo has changed.")
    add = anchor + (
        "        # --- GRADIENT CHECKPOINTING PATCH ---\n"
        "        # finetune_gradient_checkpointing is only honoured in the\n"
        "        # unsloth branch, so on this path it never turns on. Without\n"
        "        # it a 48-layer 12B holds every layer's activations.\n"
        "        if config.get('finetune_gradient_checkpointing', True):\n"
        "            model.gradient_checkpointing_enable(\n"
        "                gradient_checkpointing_kwargs={'use_reentrant': False})\n"
        "            model.enable_input_require_grads()\n"
        "            model.config.use_cache = False\n"
        "            print('[patch] gradient checkpointing ENABLED "
        "(transformers path)')\n"
    )
    open(FINETUNING, "w").write(src.replace(anchor, add))
    print("  fix 1: gradient checkpointing enabled on the transformers path")
    return True


if __name__ == "__main__":
    print("Applying FTPO memory fixes:")
    patch_checkpointing()
    patch_logits()
    import py_compile
    for f in (TRAINER, FINETUNING):
        py_compile.compile(f, doraise=True)
    print("Both files compile. Done.")
