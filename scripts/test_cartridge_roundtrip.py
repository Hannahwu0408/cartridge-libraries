"""
test_cartridge_roundtrip.py — verify save -> load reproduces a cartridge EXACTLY.

A common, costly mistake: train a batch of cartridges, then find the reloaded copies
don't reproduce the same outputs (dtype/layout/metadata drift on save/load). This inits
one cartridge, saves+loads it, and checks (a) the K/V tensors are bit-identical, (b) dtype
and metadata match, and (c) greedy decoding gives the SAME answer before and after — and
that two loaded cartridges still compose. Run once before training many.

Usage
-----
python scripts/test_cartridge_roundtrip.py --model-id Qwen/Qwen3-4B --cartridge-len 128
"""
from __future__ import annotations

import sys
import argparse
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from cartridge_qwen import load_model, QwenCartridge, generate_answer, compose, get_device

PASSAGE = (
    "Xawery Zulawski is a Polish film director born in 1971 in Warsaw, Poland. He graduated "
    "from the National Film School in Lodz. In 2009 he directed the film Polish-Russian War, "
    "which was based on a novel by Dorota Maslowska. His mother is the well-known Polish actress "
    "Malgorzata Braunek, and his father is the acclaimed film director Andrzej Zulawski. Xawery "
    "has worked in both film and television and is considered part of a notable Polish artistic "
    "family. He continued directing feature films and music videos throughout his career, earning "
    "recognition at several Polish film festivals for his distinctive visual style and storytelling."
)
QUESTION = "Who is the mother of Xawery Zulawski?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3-4B")
    ap.add_argument("--cartridge-len", type=int, default=128)
    args = ap.parse_args()

    device = get_device()
    model, tok = load_model(args.model_id, device)

    cart = QwenCartridge.init_from_passage(model, tok, PASSAGE, args.cartridge_len, device)
    out1 = generate_answer(model, tok, QUESTION, cartridge=cart, device=device, conversational=True)

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.pt"
        cart.save(p)
        loaded = QwenCartridge.load(p, device=device)
        loaded2 = QwenCartridge.load(p, device=device)   # a 2nd independent load for compose()

    meta_ok = (loaded.num_layers == cart.num_layers and loaded.num_kv_heads == cart.num_kv_heads
               and loaded.cartridge_len == cart.cartridge_len and loaded.head_dim == cart.head_dim)
    dtype_ok = (loaded.keys.dtype == cart.keys.dtype and loaded.values.dtype == cart.values.dtype)
    dk = (loaded.keys.to(device).float() - cart.keys.to(device).float()).abs().max().item()
    dv = (loaded.values.to(device).float() - cart.values.to(device).float()).abs().max().item()
    out2 = generate_answer(model, tok, QUESTION, cartridge=loaded, device=device, conversational=True)

    # Composition over two loaded copies should run without error (layout sanity for the gated path).
    try:
        out_comp = generate_answer(model, tok, QUESTION, cartridge=compose([loaded, loaded2]),
                                   device=device, conversational=True, cartridge_temp=1.0)
        comp_ok = True
    except Exception as e:
        out_comp, comp_ok = f"<error: {type(e).__name__}: {e}>", False

    print(f"\n{'='*60}")
    print(f"  meta match:     {meta_ok}")
    print(f"  dtype match:    {dtype_ok}  ({cart.keys.dtype})")
    print(f"  max |Δkeys|:    {dk:.3e}    max |Δvalues|: {dv:.3e}")
    print(f"  decode (orig):  {out1!r}")
    print(f"  decode (loaded):{out2!r}")
    print(f"  compose(loaded):{out_comp!r}  ok={comp_ok}")
    passed = meta_ok and dtype_ok and dk == 0.0 and dv == 0.0 and out1 == out2 and comp_ok
    print(f"{'='*60}")
    print("ROUNDTRIP OK — save/load reproduces the cartridge exactly." if passed
          else "ROUNDTRIP MISMATCH — fix before training many cartridges.")


if __name__ == "__main__":
    main()
