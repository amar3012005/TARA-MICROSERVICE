import time
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class QwenChatbot:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        max_new_tokens: int = 64,
        temperature: float = 0.3,
        top_p: float = 0.9,
        force_no_think: bool = True,
        stop_strings: Optional[List[str]] = None,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        self.flash_attn_available = self._flash_attention_available()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        model_kwargs = {
            "dtype": torch.float16 if self.device == "cuda" else torch.float32,
            "low_cpu_mem_usage": True,
            "attn_implementation": (
                "flash_attention_2"
                if self.flash_attn_available
                else ("sdpa" if self.device == "cuda" else None)
            ),
        }
        # Remove keys with None to avoid HF warnings
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if self.device != "cuda":
            self.model.to(self.device)

        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.force_no_think = force_no_think
        self.stop_strings = stop_strings or ["<|im_end|>"]
        self.temperature = temperature
        self.top_p = top_p
        self.history = []

    def _augment_user_input(self, user_input: str) -> str:
        cleaned = user_input.strip()
        if not self.force_no_think:
            return cleaned

        has_tag = any(tag in cleaned for tag in ("/think", "/no_think"))
        return cleaned if has_tag else f"{cleaned} /no_think"

    def generate_response(self, user_input: str) -> Tuple[str, dict]:
        controlled_input = self._augment_user_input(user_input)
        messages = self.history + [{"role": "user", "content": controlled_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        prompt_tokens = inputs["input_ids"].shape[-1]
        start = time.perf_counter()

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=pad_token_id,
                use_cache=True,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.temperature > 0,
            )

        response_ids = outputs[0][len(inputs["input_ids"][0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        latency_ms = (time.perf_counter() - start) * 1000
        completion_tokens = len(response_ids)
        throughput = (completion_tokens / (latency_ms / 1000)) if completion_tokens else 0

        if "</think>" in response:
            think_split = response.split("</think>", 1)
            response = think_split[1] if len(think_split) > 1 else think_split[0]

        for stop_string in self.stop_strings:
            stop_index = response.find(stop_string)
            if stop_index != -1:
                response = response[:stop_index]
                break

        response = response.strip()

        # Update history
        self.history.append({"role": "user", "content": controlled_input})
        self.history.append({"role": "assistant", "content": response})

        metadata = {
            "device": self.device,
            "attn_backend": "flash_attention_2" if self.flash_attn_available else ("sdpa" if self.device == "cuda" else "eager"),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": latency_ms,
            "tokens_per_second": throughput,
        }

        return response, metadata

    @staticmethod
    def _flash_attention_available() -> bool:
        if not torch.cuda.is_available():
            return False

        major, _ = torch.cuda.get_device_capability()
        if major < 7:
            return False

        try:
            import flash_attn  # type: ignore # noqa: F401
        except ModuleNotFoundError:
            return False

        return True

# Example Usage
if __name__ == "__main__":
    chatbot = QwenChatbot()

    # First input (without /think or /no_think tags, thinking mode is enabled by default)
    user_input_1 = "Explain me the concepts of quantum mechanics in analogy as if i am a 10 year old kid?"
    print(f"User: {user_input_1}")
    response_1, info_1 = chatbot.generate_response(user_input_1)
    print(f"Bot: {response_1}")
    print(
        f"Insights: device={info_1['device']} prompt_tokens={info_1['prompt_tokens']} "
        f"completion_tokens={info_1['completion_tokens']} latency_ms={info_1['latency_ms']:.2f} "
        f"tokens_per_s={info_1['tokens_per_second']:.2f}"
    )
    print("----------------------")

    # Second input with /no_think
    user_input_2 = "Then, how many r's in blueberries? /no_think"
    print(f"User: {user_input_2}")
    response_2, info_2 = chatbot.generate_response(user_input_2)
    print(f"Bot: {response_2}") 
    print(
        f"Insights: device={info_2['device']} prompt_tokens={info_2['prompt_tokens']} "
        f"completion_tokens={info_2['completion_tokens']} latency_ms={info_2['latency_ms']:.2f} "
        f"tokens_per_s={info_2['tokens_per_second']:.2f}"
    )
    print("----------------------")

    # Third input with /think
    user_input_3 = "Really? /think"
    print(f"User: {user_input_3}")
    response_3, info_3 = chatbot.generate_response(user_input_3)
    print(f"Bot: {response_3}")
    print(
        f"Insights: device={info_3['device']} prompt_tokens={info_3['prompt_tokens']} "
        f"completion_tokens={info_3['completion_tokens']} latency_ms={info_3['latency_ms']:.2f} "
        f"tokens_per_s={info_3['tokens_per_second']:.2f}"
    )
