# Training log, patched FTPO run

gemma-3-12b-it, LoRA rank 256, 8 target modules, 10 unfrozen layers,
`finetune_use_unsloth: false`, both memory fixes applied. A100 80GB.
Peak VRAM 28.3 GiB. Run `run_20260815_175654`, 2026-08-15.

Pulled from the run's TensorBoard event file before the pod was released.

## Convergence

Early stopping fired at step 35 of 69 when `chosen_win` crossed the project's
own 0.85 threshold.

| step | loss | pref_loss | chosen_win | margin_win | grad_norm |
|---:|---:|---:|---:|---:|---:|
| 5  | 3.9119 | 3.9003 | 0.1848 | 0.0470 | 25.91 |
| 10 | 2.8653 | 2.6262 | 0.3865 | 0.2065 | 18.88 |
| 15 | 2.0091 | 1.5295 | 0.6557 | 0.4356 | 25.72 |
| 20 | 1.6869 | 1.0711 | 0.7580 | 0.5786 | 19.72 |
| 25 | 1.7093 | 1.3338 | 0.7063 | 0.4906 | 18.73 |
| 30 | 1.8685 | 1.5469 | 0.6527 | 0.4430 | 18.76 |
| 35 | **1.0184** | 0.6045 | **0.8566** | 0.7182 | 12.00 |

Worth pointing out rather than hiding: the curve is not monotonic. `chosen_win`
peaked at 0.758 by step 20, fell back through steps 25 and 30, then jumped to
0.8566. Half an epoch is a short run and these are noisy per-batch numbers, not
held-out evaluation. Read it as "it converged", not as a smooth learning curve.

`active_weight` falling from 0.8971 to 0.2076 is the expected shape: fewer
examples still carry gradient as the model stops preferring the rejected tokens.

## Pipeline's own generation statistics

From `final_iteration_statistics.csv`, comparing the base model against the same
model with the sampler active over the same prompt set:

| | texts | chars | type-token ratio | repetition per 100k chars |
|---|---:|---:|---:|---:|
| iteration 0, base model | 68 | 300,286 | 0.2092 | 259.75 |
| iteration 1, sampler on | 64 | 290,189 | 0.2153 | **155.07** |

Repetition down about 40%, lexical diversity slightly up. This is the pipeline
grading itself, so it is weaker evidence than the independent yardstick in the
README. It points the same direction, which is the useful part.

## What was kept and what was not

The 6.3 GB LoRA adapters were left on the pod. They were the training output,
not the finding, and re-creating them costs about an hour of A100 time against
a patch file that is a few lines long. Everything needed to reproduce them is in
this repository.
