"""Photon model adapter: llama-cpp-python bridge for GGUF models.

Provides a ``do_decode`` callback compatible with Photon's pipeline,
using llama-cpp-python's eval/sample API for token-by-token decoding.

Pipelining note
---------------
llama-cpp-python's ``eval()`` is synchronous and operates on a single
shared KV cache.  Photon's pipelined decode (mechanism 2: "forward now,
sample later") requires independent GPU-visible state per slot.

For pipelined mode with llama-cpp, two approaches are possible:

1. **save_state / load_state** — snapshot the KV cache before a zombie
   eval, restore it after.  The zombie forward still runs (filling the
   GPU bubble) but its KV changes are reverted.  Zombie output is
   marked with ``_ZOMBIE_SENTINEL`` so the commit phase skips it.

2. **Multi-instance** — one Llama object per ping-pong slot, each with
   its own KV cache.  Requires 2× VRAM and state synchronisation
   between instances at commit boundaries.

The blocking mode (num_slots=1) works directly and is production-ready.
Pipelined mode is demonstrated with the synthetic GPU benchmark in
bench_real_model.py, which shows +56 % throughput improvement.

Usage
-----
>>> from photon_amd.adapter import PhotonLlamaAdapter
>>> adapter = PhotonLlamaAdapter(model_path="model.gguf")
>>> adapter.generate("Hello, how are you?")
"""

from __future__ import annotations

from dataclasses import dataclass

from llama_cpp import Llama


@dataclass
class PhotonLlamaAdapter:
    """Wraps a llama-cpp LLM for use as Photon's decode backend.

    Parameters
    ----------
    model_path : str
        Path to the GGUF model file.
    n_gpu_layers : int
        Number of layers to offload to GPU (99 = all).
    n_ctx : int
        Maximum context length.
    verbose : bool
        Whether to print llama-cpp logs.
    """

    model_path: str
    n_gpu_layers: int = 99
    n_ctx: int = 512
    verbose: bool = False

    def __post_init__(self):
        self._llm = Llama(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=self.n_ctx,
            verbose=self.verbose,
        )
        self.vocab_size = self._llm.n_vocab()
        self.eos_token = self._llm.token_eos() or 1

    # ---- Photon do_decode callback ----------------------------------------

    def do_decode(
        self,
        slot,    # DecodeSlot
        batch,   # Batch
        stream,  # torch.cuda.Stream
    ) -> None:
        """Run one decode forward (eval + sample) for each sequence."""
        for i, seq in enumerate(batch.sequences):
            token = seq.next_token_id()
            self._llm.eval([token])
            sampled = self._llm.sample(temp=0.0)
            slot.sampled_ids[i] = sampled
            seq.pending_token = sampled
            slot.logits[i, 0] = float(sampled)

    # ---- Tokenization ------------------------------------------------------

    def tokenize(self, text: str) -> list[int]:
        return self._llm.tokenize(text.encode("utf-8"))

    def detokenize(self, tokens: list[int]) -> str:
        return self._llm.detokenize(tokens).decode("utf-8", errors="replace")

    # ---- Prefill -----------------------------------------------------------

    def prefill_prompt(self, prompt_tokens: list[int]) -> None:
        """Eval all prompt tokens except the last.

        The last prompt token is evaluated by the first do_decode call.
        """
        if len(prompt_tokens) <= 1:
            if prompt_tokens:
                self._llm.eval(prompt_tokens)
            return
        self._llm.eval(prompt_tokens[:-1])

    # ---- Convenience -------------------------------------------------------

    def generate(self, prompt: str, max_new_tokens: int = 32) -> str:
        """Generate text through the Photon pipeline (blocking mode)."""
        from photon_amd import PhotonConfig, PhotonEngine, PipelineCallbacks

        tokens = self.tokenize(prompt)
        self.prefill_prompt(tokens)

        config = PhotonConfig(
            max_batch_size=1,
            num_slots=1,
            vocab_size=self.vocab_size,
            eos_token_id=self.eos_token,
            capture_graphs=False,
        )
        callbacks = PipelineCallbacks(
            do_decode=self.do_decode,
            do_sample=lambda s, b, st: None,
        )
        engine = PhotonEngine(config, callbacks)
        engine.submit(tokens, max_new_tokens=max_new_tokens)
        engine.run()
        for _, gen in engine.collect_results().items():
            return self.detokenize(gen)
        return ""
