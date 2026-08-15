# Reproducing Antislop, and a memory fix for its fine-tuner

A reproduction of [**Antislop: A Comprehensive Framework for Identifying and
Eliminating Repetitive Patterns in Language Models**](https://arxiv.org/abs/2510.15061)
(Paech, Roush, Goldfeder, Shwartz-Ziv, ICLR 2026), plus a fix for a memory
problem I hit in the reference implementation.

The paper's method works. I measured it on two model sizes and reproduced the
core result. The fix below is about the released code, not the research.

---

## Headline results

**The sampler removes AI-cliché language, measured on 2.5 million words.**

| Model | Baseline | With sampler | Reduction |
|---|---|---|---|
| gemma-3-4b-it | 12.77 per 10k words | 3.36 per 10k words | **73.7%** |
| gemma-3-12b-it | 11.96 per 10k words | 3.77 per 10k words | **68.5%** |

Scored against a fixed list of well-known AI clichés (`tapestry`, `delve`,
`shimmered`, `palpable`, `testament to`, `barely above a whisper` and similar)
chosen **independently of the ban lists the pipeline generates**, so the
measurement is not grading its own homework. Two model sizes, consistent result.

**A two-line fix cuts fine-tuning memory about 5x.**

| Configuration | GPU | Peak VRAM | Result |
|---|---|---|---|
| As shipped, LoRA rank 256 | H200 141GB | 139.6 GiB | out of memory |
| As shipped, LoRA rank 128 | H200 141GB | ~138.9 GiB | out of memory |
| **Patched, LoRA rank 256** | **A100 80GB** | **28.3 GiB** | **converged** |

Same model, same rank, same eight target modules, same ten unfrozen layers.
The patched run reached the project's own convergence threshold
(`chosen_win 0.8566`, early-stopped at step 35/69, loss 3.9119 to 1.0184) on a
card with **57% of the memory and about a third of the hourly cost**.

### What the difference looks like

Both from the same gemma-3-12b run, same prompt set. First the model as shipped:

> The scent of lilies and ozone still clung to **Elara**. She'd been pulling
> them from her hair for days, but the floral phantom persisted, a constant
> reminder of the moment Kratos, god of storms [...]

That sample also contains `testament to`, `shimmering` and `kaleidoscope`. Note
the character name: **Elara** is the paper's own headline example, measured at
85,513 times more frequent in this model family's output than in human writing.
It appeared unprompted.

The same model with the sampler active:

> The vet's hands were gentle, too gentle. They smelled of antiseptic and a
> strange, quiet sadness, and I didn't want to breathe them in. I didn't want
> to acknowledge the weight in my lap, the warm, shuddering bulk of him, the
> steady thump-thump-thump of his heart slowing, slowing, slowing until it was
> just a ghost echo.

Concrete detail instead of stock fantasy vocabulary. Worth saying plainly:
suppression is not total. That second passage still contains one yardstick
term. The measured reduction is roughly 70%, not 100%.

---

## How I found it

The fine-tune ran out of memory on a 141GB card. The obvious response is to
rent a bigger one. Before doing that I halved the LoRA rank from 256 to 128,
expecting a large drop.

**Memory barely moved.** 139.6 GiB to roughly 138.9 GiB.

That non-result is the whole finding. If halving the adapter size changes
almost nothing, the adapters are not what is consuming the memory, so no amount
of shrinking them will help and a bigger GPU only buys headroom for waste.
Reading the source from that starting point turned up two causes.

### Cause 1: gradient checkpointing never turns on

`finetune_gradient_checkpointing` is read in exactly one place,
`core/finetuning.py:497`, and that line sits inside the `if use_unsloth:`
branch:

```python
model = FastLanguageModel.get_peft_model(
    ...
    use_gradient_checkpointing=config['finetune_gradient_checkpointing'],
```

The transformers/TRL branch never calls `gradient_checkpointing_enable()`, and
the trainer config built further down never passes
`gradient_checkpointing=True`. So with `finetune_use_unsloth: false` the
setting is read from the YAML and then never applied.

This matters because installing unsloth pulls its own torch build and breaks a
pinned vllm/trl environment, so the transformers path is the one you land on
when the other path fails. A 48-layer 12B without checkpointing keeps every
layer's activations.

### Cause 2: full-vocabulary logits computed for every position, one used

All three forward passes in `core/ftpo_trainer.py` compute logits for the whole
sequence and then read a single position:

```python
outputs = model(ids, attention_mask=attn, position_ids=pos_full,
                use_cache=False, return_dict=True)
logits_last = outputs.logits[:, -1, :]   # [B, V]
```

`[:, -1, :]` returns a view, so the full `[B, L, V]` tensor stays alive for the
rest of the loss. FTPO only ever needs the final token, which is the point of
the method. For gemma-3-12b (vocab 262,208) at batch 3 and sequence 2500 in
bf16 that is about 3.66 GiB per tensor, several times over across the policy
and reference passes.

`logits_to_keep=1` produces the same values through the same slice without
building the rest.

---

## The fix

[`ftpo_memory_patch.py`](ftpo_memory_patch.py) applies both changes to a clone
of the upstream repo. It is idempotent and verifies the files compile.

```bash
git clone --recurse-submodules https://github.com/sam-paech/auto-antislop.git
cd auto-antislop
python3 ../ftpo_memory_patch.py
```

**Use `use_reentrant: True`.** With `False`, training dies at step 0 on a
checkpoint metadata mismatch (saved `[1, 4000, 3840] float32` against
recomputed `[4000, 3840] bfloat16`). I hit this and it cost a run to find.

---

## Reproducing the measurement

[`measure_slop_reduction.py`](measure_slop_reduction.py) computes the numbers in
the first table from a completed pipeline run. It compares iteration 0 (the
base model) against iteration 1 (the same model with the sampler active)
using the independent yardstick described above.

```bash
python3 measure_slop_reduction.py /path/to/results/auto_antislop_runs/run_YYYYMMDD_HHMMSS
```

---

## What this does and does not claim

**Does:** the released code loses gradient checkpointing on one of its
two supported paths, and its trainer builds far more logits than it reads.
Fixing both let a configuration that would not fit on a 141GB card converge on
an 80GB card.

**Does not:** say anything is wrong with the paper. The method works, the
results reproduced, and I measured a 74% and a 69% reduction myself. This is an
engineering issue in released research code, which is an ordinary thing to find
and an easy thing to fix.

Everything above was run on rented GPUs (RunPod A100 80GB and H200 141GB) with
gemma-3-4b-it and gemma-3-12b-it via the ungated `unsloth/` mirrors.

## Credit

All credit for the method to the authors. Paper:
[arXiv:2510.15061](https://arxiv.org/abs/2510.15061). Code:
[sam-paech/auto-antislop](https://github.com/sam-paech/auto-antislop) and
[sam-paech/antislop-sampler](https://github.com/sam-paech/antislop-sampler),
MIT licensed.

## License

MIT, for the contents of this repository.
