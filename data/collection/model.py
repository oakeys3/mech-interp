"""HuggingFace generation backend for collection.

Kept in its own module so that ``collect.py``'s trajectory loop can be tested
without importing torch. Ported from the ``LM`` class in ``spike/spike.py``,
minus the base-model path — Pythia was eliminated in Phase 0, so only the
instruct chat interface remains.
"""

from __future__ import annotations


class HuggingFaceGenerator:
    """Generates chat completions from a local HuggingFace causal LM."""

    def __init__(
        self, model: str, temperature: float = 0.0, max_new_tokens: int = 512
    ) -> None:
        """Load the tokenizer and model onto the best available device.

        Args:
            model: HuggingFace model id.
            temperature: 0 means greedy decoding; above 0 enables sampling.
                Greedy is reproducible but yields identical repeat samples,
                so multi-sample collection requires a nonzero value.
            max_new_tokens: Generation length cap.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model, dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.model.eval()

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Generate a response to a chat message list.

        Args:
            messages: OpenAI-style ``{"role", "content"}`` dicts.

        Returns:
            The decoded completion, with the prompt tokens removed.
        """
        # transformers 5.x returns a dict here; it must be unpacked with **.
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)

        with self._torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        prompt_length = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)
