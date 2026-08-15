"""Measure how much the Antislop sampler reduces AI-cliché language.

Compares a completed auto-antislop run's two generation passes:

  iteration 0  the base model
  iteration 1  the same model, same prompts, sampler active

Scoring uses a fixed list of well-known AI clichés chosen INDEPENDENTLY of the
ban lists the pipeline generates for itself. That matters: scoring against the
pipeline's own ban list would only prove it can avoid words it was told to
avoid. This asks a harder question, whether general cliché density drops.

Usage:
    python3 measure_slop_reduction.py /path/to/run_YYYYMMDD_HHMMSS

Measured with this script:
    gemma-3-4b-it    12.77 -> 3.36 per 10k words   73.7% reduction
    gemma-3-12b-it   11.96 -> 3.77 per 10k words   68.5% reduction
"""
import glob
import json
import os
import re
import sys

# Fixed yardstick. Chosen from the wider discourse about AI writing tells, not
# from any ban list this pipeline produced.
YARDSTICK = [
    "tapestry", "testament to", "barely above a whisper", "delve", "delved",
    "shimmering", "shimmered", "a symphony of", "kaleidoscope", "palpable",
    "unwavering", "indelible", "maelstrom", "cacophony", "ministrations",
    "labyrinthine", "ethereal", "liminal",
]
PATTERNS = [(t, re.compile(r"\b" + re.escape(t) + r"\b", re.I)) for t in YARDSTICK]


def load(path):
    """Read generations out of one of the pipeline's jsonl files."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path, errors="ignore") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            for key in ("generation", "text", "output", "response"):
                if rec.get(key):
                    out.append(rec[key])
                    break
    return out


def score(docs):
    words = sum(len(d.split()) for d in docs)
    hits = {}
    total = 0
    for doc in docs:
        for term, pat in PATTERNS:
            n = len(pat.findall(doc))
            if n:
                hits[term] = hits.get(term, 0) + n
                total += n
    per10k = (total / words * 10000) if words else 0.0
    return len(docs), words, total, per10k, hits


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    run = sys.argv[1].rstrip("/")
    base = load(os.path.join(run, "iter_0_creative_writing_generations.jsonl"))
    slop = load(os.path.join(run, "iter_1_creative_writing_generations.jsonl"))
    if not base or not slop:
        sys.exit(f"Could not find both generation files under {run}")

    nb, wb, hb, rb, hits_b = score(base)
    ns, ws, hs, rs, hits_s = score(slop)

    print(f"{'':<12}{'docs':>7}{'words':>12}{'hits':>8}{'per 10k':>10}")
    print(f"{'baseline':<12}{nb:>7}{wb:>12,}{hb:>8}{rb:>10.2f}")
    print(f"{'antislop':<12}{ns:>7}{ws:>12,}{hs:>8}{rs:>10.2f}")
    if rb:
        print(f"\nreduction: {(1 - rs / rb) * 100:.1f}%")

    print("\nmost frequent in baseline:")
    for term, n in sorted(hits_b.items(), key=lambda kv: -kv[1])[:8]:
        after = hits_s.get(term, 0)
        print(f"  {term:<26} {n:>4} -> {after}")


if __name__ == "__main__":
    main()
