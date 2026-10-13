"""Shared harness: load a HF causal LM at a given precision on MPS, report perplexity, speed, memory."""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "mps"
RESULTS = Path(__file__).parents[1] / "results"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    p.add_argument("--ctx", type=int, default=1024, help="tokens per perplexity window")
    p.add_argument("--max-windows", type=int, default=40, help="cap on wikitext-2 windows (full set is ~280)")
    p.add_argument("--gen-tokens", type=int, default=128)
    return p.parse_args()


def load(model_id, dtype, transform=None):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).eval()
    if transform:  # runs on CPU, before the move, so it can use dtypes MPS lacks
        model = transform(model)
    return tok, model.to(DEVICE)


@torch.no_grad()
def perplexity(tok, model, ctx, max_windows):
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    n = min(len(ids) // ctx, max_windows)
    # equal-length windows, so mean of per-window losses == mean per-token loss
    nll = sum(model(x := ids[i * ctx:(i + 1) * ctx][None].to(DEVICE), labels=x).loss.item() for i in range(n))
    return math.exp(nll / n)


@torch.no_grad()
def throughput(tok, model, n_tokens):
    x = tok("The history of floating point arithmetic", return_tensors="pt").to(DEVICE)
    model.generate(**x, max_new_tokens=8, do_sample=False)  # warmup: MPS builds kernels on first call
    torch.mps.synchronize()
    t = time.perf_counter()
    out = model.generate(**x, max_new_tokens=n_tokens, min_new_tokens=n_tokens, do_sample=False)
    torch.mps.synchronize()
    return (out.shape[1] - x.input_ids.shape[1]) / (time.perf_counter() - t)


def run(precision, dtype, transform=None):
    a = parse_args()
    tok, model = load(a.model, dtype, transform)
    r = {
        "precision": precision,
        "model": a.model,
        "ctx": a.ctx,
        "windows": a.max_windows,
        "weights_gb": round(torch.mps.current_allocated_memory() / 1e9, 3),
        "perplexity": perplexity(tok, model, a.ctx, a.max_windows),
        "tokens_per_s": throughput(tok, model, a.gen_tokens),
    }
    print(json.dumps(r, indent=2))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{a.model.replace('/', '_')}_{precision}.json").write_text(json.dumps(r, indent=2))
