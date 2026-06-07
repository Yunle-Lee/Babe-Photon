"""Photon-AMD demo: real Gemma-4 12B inference through Photon pipeline.

Demonstrates the Photon inference engine running real text generation
on AMD Instinct MI300 using llama-cpp-python + Gemma-4 12B GGUF.

Currently uses blocking mode (num_slots=1).  Pipelined mode (num_slots=2)
with llama-cpp requires save_state/load_state guards that are documented
in the adapter.py module; the core pipeline framework (ping-pong slots,
forward-now-sample-later, zombie lifecycle) is fully functional and
benchmarked in bench_real_model.py.

Usage:
    python examples/demo_real_model.py
"""

from __future__ import annotations

import time
import torch
from llama_cpp import Llama

from photon_amd import PhotonConfig, PhotonEngine, PipelineCallbacks
from photon_amd.decode_slot import DecodeSlot
from photon_amd.scheduler import Batch

MODEL_PATH = "/mnt/data/my_workspace/models/gemma-4-12b-it-UD-Q4_K_XL.gguf"

PROMPTS = [
    (
        "What is 2+2?",
        "<start_of_turn>user\nWhat is 2+2? Answer in one word.<end_of_turn>\n<start_of_turn>model\n",
    ),
    (
        "Who are you?",
        "<start_of_turn>user\nWho are you? Answer in one sentence.<end_of_turn>\n<start_of_turn>model\n",
    ),
    (
        "Haiku",
        "<start_of_turn>user\nWrite a haiku about programming.<end_of_turn>\n<start_of_turn>model\n",
    ),
]


def main():
    print("=" * 70)
    print("Photon-AMD + Gemma-4 12B: Real Model Inference Demo")
    print("=" * 70)
    print("GPU:  AMD Instinct MI300 (gfx942) @ 192 GB VRAM")
    print(f"ROCm: {torch.version.hip}")
    print("Model: Gemma-4 12B IT (Q4_K_XL GGUF)")
    print()

    # ---- Load model --------------------------------------------------------
    print("Loading model...", end=" ", flush=True)
    llm = Llama(model_path=MODEL_PATH, n_gpu_layers=99, n_ctx=512, verbose=False)
    eos = llm.token_eos() or 106
    vocab = llm.n_vocab()
    print(f"done.  vocab={vocab}, eos={eos}")
    print()

    # ---- do_decode closure -------------------------------------------------
    def make_do_decode(_llm):
        def do_decode(slot: DecodeSlot, batch: Batch, stream: torch.cuda.Stream):
            for i, seq in enumerate(batch.sequences):
                token = seq.next_token_id()
                _llm.eval([token])
                sampled = _llm.sample(temp=0.0)
                slot.sampled_ids[i] = sampled
                seq.pending_token = sampled
                slot.logits[i, 0] = float(sampled)

        return do_decode

    # ---- Run prompts -------------------------------------------------------
    for tag, prompt_text in PROMPTS:
        tokens = llm.tokenize(prompt_text.encode("utf-8"))
        llm.eval(tokens[:-1])  # prefill all but last token

        config = PhotonConfig(
            max_batch_size=1,
            num_slots=1,
            vocab_size=vocab,
            eos_token_id=eos,
            capture_graphs=False,
        )
        callbacks = PipelineCallbacks(
            do_decode=make_do_decode(llm),
            do_sample=lambda slot, batch, stream: None,
        )
        engine = PhotonEngine(config, callbacks)
        engine.submit(tokens, max_new_tokens=30)

        t0 = time.perf_counter()
        engine.run()
        elapsed = time.perf_counter() - t0

        results = engine.collect_results()
        for _, gen_tokens in results.items():
            text = llm.detokenize(gen_tokens).decode("utf-8", errors="replace")
            tps = len(gen_tokens) / elapsed if elapsed > 0 else 0
            print(f"[{tag}] {len(gen_tokens)} tokens, {elapsed:.2f}s, {tps:.1f} tok/s")
            for line in text.strip().split("\n"):
                if line.strip():
                    print(f"  {line.strip()}")
            print()

    print("=" * 70)


if __name__ == "__main__":
    main()
