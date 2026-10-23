"""Correctness of the hand-written decoder, KV-cached greedy and speculative decoding.

Uses Qwen2.5-0.5B-Instruct in float32 on CPU (downloaded on first run, ~1 GB).
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from trialmatch.decoding import greedy_cached, greedy_uncached, speculative
from trialmatch.qwen_decoder import QwenDecoder

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPTS = {
    "short": "The capital of France is",
    "medium": "A 58-year-old woman presents to the ER with episodic chest pain. She has hypertension "
              "and obesity and takes no medications. The most likely next diagnostic step is",
    "long": " ".join(["Patient note: 64-year-old man with type 2 diabetes, chronic kidney disease stage 3, "
                      "and a prior myocardial infarction, currently on metformin and aspirin."] * 6)
            + " Summary of eligibility concerns:",
}


@pytest.fixture(scope="module")
def setup():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    return tok, model, QwenDecoder(model)


def ids_of(tok, text):
    return tok(text, return_tensors="pt")["input_ids"]


def hf_greedy(model, ids, n):
    # repetition_penalty=1.0 overrides the 1.1 in Qwen's generation_config.json;
    # eos_token_id=None generates exactly n tokens, like our decoders without eos_ids.
    out = model.generate(ids, max_new_tokens=n, do_sample=False, eos_token_id=None,
                         temperature=None, top_p=None, top_k=None, repetition_penalty=1.0)
    return out[0, ids.shape[1]:].tolist()


@pytest.mark.parametrize("attention", ["sdpa", "reference", "tiled"])
def test_decoder_matches_hf_logits(setup, attention):
    tok, model, _ = setup
    ids = ids_of(tok, PROMPTS["medium"])
    dec = QwenDecoder(model, attention=attention)
    ours = dec.forward(ids, dec.new_cache(1, ids.shape[1]), 0)
    with torch.no_grad():
        theirs = model(ids).logits
    assert (ours - theirs).abs().max().item() < 1e-3


def test_incremental_forward_matches_full_forward(setup):
    """Feeding the prompt in chunks through the cache must equal one full pass."""
    tok, _, dec = setup
    ids = ids_of(tok, PROMPTS["medium"])
    full = dec.forward(ids, dec.new_cache(1, ids.shape[1]), 0)
    cache, chunks, pos = dec.new_cache(1, ids.shape[1]), [], 0
    for size in [5, 1, 7, 3, ids.shape[1]]:
        part = ids[:, pos:pos + size]
        if part.shape[1] == 0:
            break
        chunks.append(dec.forward(part, cache, pos))
        pos += part.shape[1]
    assert (torch.cat(chunks, dim=1) - full).abs().max().item() < 1e-3


@pytest.mark.parametrize("prompt,n_new", [("short", 8), ("medium", 24), ("long", 40)])
def test_cached_equals_uncached_and_hf(setup, prompt, n_new):
    tok, model, dec = setup
    ids = ids_of(tok, PROMPTS[prompt])
    cached = greedy_cached(dec, ids, n_new).tokens
    uncached = greedy_uncached(dec, ids, n_new).tokens
    assert cached == uncached
    assert cached == hf_greedy(model, ids, n_new)


def test_cache_crop_rolls_back(setup):
    tok, _, dec = setup
    ids = ids_of(tok, PROMPTS["short"])
    cache = dec.new_cache(1, 64)
    dec.forward(ids, cache, 0)
    n = cache.length
    dec.forward(torch.tensor([[11, 12, 13]]), cache, n)  # tokens that get rejected
    cache.crop(n)
    after_rollback = dec.forward(torch.tensor([[42]]), cache, n)

    fresh = dec.new_cache(1, 64)
    dec.forward(ids, fresh, 0)
    expected = dec.forward(torch.tensor([[42]]), fresh, n)
    assert (after_rollback - expected).abs().max().item() < 1e-4


class NoisyDraft:
    """A deliberately unreliable draft: the real model with heavy logit noise."""

    def __init__(self, dec, noise, seed=0):
        self.dec, self.noise = dec, noise
        self.gen = torch.Generator().manual_seed(seed)
        self.device = dec.device

    def new_cache(self, batch, max_len):
        return self.dec.new_cache(batch, max_len)

    def forward(self, ids, cache, start):
        logits = self.dec.forward(ids, cache, start)
        return logits + self.noise * torch.randn(logits.shape, generator=self.gen)


@pytest.mark.parametrize("k", [1, 3, 5])
@pytest.mark.parametrize("noise", [0.0, 2.0, 50.0])
def test_speculative_equals_greedy(setup, k, noise):
    tok, _, dec = setup
    ids = ids_of(tok, PROMPTS["medium"])
    expected = greedy_cached(dec, ids, 30).tokens
    result = speculative(dec, NoisyDraft(dec, noise), ids, 30, k=k)
    assert result.tokens == expected
    if noise == 0.0:
        assert result.stats["acceptance_rate"] == 1.0
    if noise == 50.0:
        assert result.stats["acceptance_rate"] < 0.5  # rejection path exercised


def test_speculative_stops_at_eos(setup):
    tok, _, dec = setup
    ids = ids_of(tok, PROMPTS["short"])
    reference = greedy_cached(dec, ids, 20).tokens
    eos = reference[5]
    expected = greedy_cached(dec, ids, 20, eos_ids={eos}).tokens
    got = speculative(dec, NoisyDraft(dec, 1.0), ids, 20, k=4, eos_ids={eos}).tokens
    assert got == expected and got[-1] == eos
