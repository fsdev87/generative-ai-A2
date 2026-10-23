"""Model/tokenizer loading and batched greedy generation used for evaluation."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompting import build_messages

TARGET_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def default_dtype(device):
    # T4 GPUs have no native bf16, so fp16 on CUDA; fp32 on CPU for accuracy.
    return torch.float16 if device.type == "cuda" else torch.float32


def load_tokenizer(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"  # decoder-only batch generation needs left padding
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(model_name, adapter=None, device=None, dtype=None):
    device = device or pick_device()
    dtype = dtype or default_dtype(device)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    return model


def prompt_text(tok, record, order="evidence_first"):
    return tok.apply_chat_template(build_messages(record, order), add_generation_prompt=True,
                                   tokenize=False)


@torch.no_grad()
def generate_batch(model, tok, prompts, max_new_tokens=64):
    """Greedy generation for a batch of chat-formatted prompts; returns decoded continuations.

    Qwen's generation_config.json sets repetition_penalty=1.1, which HF applies even
    with do_sample=False; it is reset to 1.0 so this is plain argmax decoding.
    """
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        repetition_penalty=1.0,
        pad_token_id=tok.pad_token_id,
    )
    return tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
