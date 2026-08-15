"""Two memory fixes for auto-antislop's FTPO trainer.

Diagnosis (verified live on rented GPUs, 2026-08-15): fine-tuning gemma-3-12b
at LoRA rank 256 (batch 1, grad accum 16, max seq 4000, 4-bit base) OOM'd on a
141GB H200 using 139.6 GiB. A fallback attempt at rank 128 with 6 unfrozen
layers instead of 10 still OOM'd at ~138.9 GiB, which ruled out the adapters as
the cause. Two things dominate:

FIX 1 - gradient checkpointing is silently disabled on the transformers path.
  `finetune_gradient_checkpointing` is read in exactly one place,
  core/finetuning.py:497, which sits INSIDE the `if use_unsloth:` branch. With
  finetune_use_unsloth: false (required, since unsloth breaks the install) the
  setting is dead code and checkpointing never turns on. A 48-layer 12B at
  seq 4000 with eager attention then keeps every layer's activations, including
  the seq x seq attention matrices, for the whole backward pass.

FIX 2 - the forward computes 4000x more logits than the loss uses.
  ftpo_trainer.py builds logits for every position, then reads only the last:
      logits_last = outputs.logits[:, -1, :]
  `[:, -1, :]` is a view, so the whole [B, L, V] tensor stays alive. At
  1 x 4000 x 262208 in bf16 that is ~1.95 GiB per forward, three forwards per
  step across the policy and reference passes, before counting the float32
  copies the loss math makes. Passing logits_to_keep=1 is mathematically
  identical (the slice still selects the final position) and removes nearly
  all of it.

With both fixes the identical rank-256 configuration trained in 28.3 GiB on an
A100 80GB and converged (chosen_win 0.8566).

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
        "        # use_reentrant MUST be True here. False crashes at step 0\n"
        "        # with a checkpoint metadata mismatch (saved [1, 4000, 3840]\n"
        "        # float32 vs recomputed [4000, 3840] bfloat16), verified live.\n"
        "        if config.get('finetune_gradient_checkpointing', True):\n"
        "            model.gradient_checkpointing_enable(\n"
        "                gradient_checkpointing_kwargs={'use_reentrant': True})\n"
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
