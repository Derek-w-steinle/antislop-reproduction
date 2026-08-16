# Reproducing Antislop, and a memory fix for its fine-tuner

A reproduction of [**Antislop: A Comprehensive Framework for Identifying and
Eliminating Repetitive Patterns in Language Models**](https://arxiv.org/abs/2510.15061)
(Paech, Roush, Goldfeder, Shwartz-Ziv, ICLR 2026), plus a fix for a memory
problem I hit in the reference implementation.

The paper's method works. I measured it on two model sizes and reproduced the
core result. The fix below is about the released code, not the research.

**Status:** reported upstream as
[issue #3](https://github.com/sam-paech/auto-antislop/issues/3) on 2026-08-15.
The author confirmed the finds the same day and requested a pull request;
[PR #4](https://github.com/sam-paech/auto-antislop/pull/4) is open.

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
| As shipped, rank 256, 10 unfrozen layers | H200 141GB | 139.6 GiB | out of memory |
| As shipped, rank 128, 6 unfrozen layers | H200 141GB | 138.9 GiB | out of memory |
| **Patched, rank 256, 10 unfrozen layers** | **A100 80GB** | **28.3 GiB** | **converged** |

The first and third rows are the same configuration in every memory-relevant
setting. Same model, LoRA rank 256 with alpha 256 across the same eight target
modules, ten unfrozen layers, batch size 1 with gradient accumulation 16,
4,000-token max sequence, 4-bit base, eager attention. I verified that after
the fact against the session's full command record rather than trusting memory,
and batch and sequence length are also confirmed at runtime, since a crash
message from an early patched attempt names the tensor shape `[1, 4000, 3840]`,
which is batch 1, sequence 4000, and gemma-3-12b's hidden size.

The patched run reached the project's own convergence threshold
(`chosen_win 0.8566`, early-stopped at step 35/69, loss 3.9119 to 1.0184) on a
card with **57% of the memory and about a third of the hourly cost**. Full
curve in [`TRAINING-LOG.md`](TRAINING-LOG.md), including where it is noisy.

One note on the 139.6 GiB figure. That is where the run died, not where it
would have topped out. A killed run only shows the memory it had reached when
it hit the wall, so the true unpatched requirement is something above 141 GB
and unknown. On a bigger card the left side of this comparison would likely
read higher, which makes the roughly 5x saving a floor rather than an estimate.

Two other differences between the runs, disclosed rather than buried. The A100
run built its FTPO dataset from 120 prompts against the H200's 1,000, which
changes how many optimizer steps there are, not how much memory one step takes
at batch size 1. The A100 run demonstrably hit the 4,000-token cap (that crash
tensor again); the H200's batch lengths are not recoverable, and if its batches
happened to be shorter it ran out of memory on an even lighter workload, which
would only make the contrast larger. And the two peak figures come from
different instruments. The
139.6 GiB is what the CUDA out-of-memory report said was in use when the H200
run died, while the 28.3 GiB is nvidia-smi polled repeatedly during training.

### What the difference looks like

Two samples from the same gemma-3-12b run. **These are different prompts, not a
matched pair.** Read them as flavour, not as evidence. The evidence is the table
above.

Sampler off, three yardstick terms in the passage:

> The scent of lilies and ozone still clung to **Elara**. She'd been pulling
> them from her hair for days, but the floral phantom persisted, a constant
> reminder of the moment Kratos, god of storms [...]

Sampler on, one yardstick term:

> The vet's hands were gentle, too gentle. They smelled of antiseptic and a
> strange, quiet sadness, and I didn't want to breathe them in. I didn't want
> to acknowledge the weight in my lap, the warm, shuddering bulk of him, the
> steady thump-thump-thump of his heart slowing, slowing, slowing until it was
> just a ghost echo.

Concrete detail instead of stock fantasy vocabulary. Suppression is not total.
That second passage still contains a yardstick term. The measured reduction is
roughly 70%, not 100%.

### The name problem, which this method handles worst

The first passage contains **Elara**, the paper's headline example, measured at
85,513 times more frequent in this model family's output than in human writing.
The base model reached for it unprompted. That is all I can claim. `Elara` is on
the ban lists, so the sampler was configured to catch it, but names were not in
my yardstick and I released the GPU before thinking to check. Whether it
survived into the sampled output is unmeasured.

Names are the weak case for banning in general. Of forty stock character names I
checked against the shipped 2,000-word list, seventeen are banned (`Elara`,
`Lyra`, `Kael`, `Seraphina`, `Aeliana`, `Thorne`, `Isolde`) and twenty-three are
not (`Cassian`, `Soren`, `Vesper`, `Rowan`, `Thalia`, `Lucian`, `Nyx`).

Ban a name and the model takes the next one down. A phrase like `testament to`
has no equally good substitute, so suppressing it forces an actual rewrite. A
name has thousands of substitutes and the swap costs the model nothing. This is
why the paper's design is iterative regeneration rather than one fixed list, and
it is the part of the approach I would want to measure properly before claiming
anything about it.

---

## How I found it

The fine-tune ran out of memory on a 141GB card. The obvious response is to
rent a bigger one. The overnight run had been scripted with a fallback chain,
smaller configuration after smaller configuration, so by morning there were
three data points instead of one.

| attempt | adapters | result |
|---|---|---|
| rank 256, all 8 modules, 10 layers | full size | OOM at 139.6 GiB |
| rank 128, all 8 modules, 6 layers | roughly half | OOM at 138.9 GiB |
| rank 64, lm_head only, 4 layers | a fraction | trained |

**Rows one and two are the finding.** Cutting the adapters roughly in half,
and the unfrozen layer count nearly in half with them, moved peak memory by
half of one percent. If shrinking the adapters changes almost nothing, the
adapters are not what is consuming the memory, so no amount of shrinking them
will help and a bigger GPU only buys headroom for waste.

Row three points the same way from the other side. It fit not because its
adapters were smaller but because backprop only had to reach into the top few
layers, so far fewer activations were kept alive. Useful as a diagnostic,
useless as a result, since it is nowhere near the paper's configuration.

Reading the source while looking for a large fixed cost that ignores adapter
size turned up two causes.

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

All three forward passes in `core/ftpo_trainer.py`, two on the policy model and
one on the reference model (lines 218, 260 and 265), compute logits for the
whole sequence and then read a single position:

```python
outputs = model(ids, attention_mask=attn, position_ids=pos_full,
                use_cache=False, return_dict=True)
logits_last = outputs.logits[:, -1, :]   # [B, V]
```

`[:, -1, :]` returns a view, so the full `[B, L, V]` tensor stays alive for the
rest of the loss. FTPO only ever needs the final token, which is the point of
the method. For gemma-3-12b (vocab 262,208) at batch 1 and sequence 4000 in
bf16 that is about 1.95 GiB per forward, three forwards per step, before
counting the float32 copies the loss math makes.

`logits_to_keep=1` produces the same values through the same slice without
building the rest.

Line numbers in both causes were re-verified against upstream `main` on the day
of writing (commit `da22315`, 2026-07-29). `logits_to_keep` appeared zero times
in the trainer at that point. Both fixes are now proposed upstream in
[PR #4](https://github.com/sam-paech/auto-antislop/pull/4), opened at the
author's request after he confirmed the finds on
[issue #3](https://github.com/sam-paech/auto-antislop/issues/3).

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

Worth saying as well. None of this would have been possible if the authors had
not released their code. Finding a fixable issue in a public implementation is
the system working, and releasing code that people can poke at is the
generous half of that exchange.

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
