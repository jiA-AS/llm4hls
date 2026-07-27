import logging
import os
from .base import ModelBackend, ALPACA_PROMPT

logger = logging.getLogger(__name__)

# Use HF mirror for users behind network restrictions (e.g., mainland China).
_HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
os.environ.setdefault("HF_ENDPOINT", _HF_ENDPOINT)


class HuggingFaceBackend(ModelBackend):
    """Run inference via Hugging Face model."""

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 30000,
        temperature: float = 1.0,
        use_4bit: bool = True,
        max_seq_length: int = 8192,
        hf_token: str | None = None,
        hf_endpoint: str | None = None,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.hf_token = hf_token
        if hf_endpoint:
            os.environ["HF_ENDPOINT"] = hf_endpoint
            logger.info("HF_ENDPOINT (config) = %s", hf_endpoint)
        else:
            logger.info("HF_ENDPOINT = %s", _HF_ENDPOINT)
        self._load(use_4bit, max_seq_length)

    def _load(self, use_4bit: bool, max_seq_length: int) -> None:
        try:
            self._load_unsloth(use_4bit, max_seq_length)
        except ImportError:
            logger.info("unsloth not found — using standard transformers")
            self._load_transformers(use_4bit)

    def _load_unsloth(self, use_4bit: bool, max_seq_length: int) -> None:
        from unsloth import FastLanguageModel  # type: ignore

        logger.info("Loading %s via unsloth (4bit=%s)", self.model_name, use_4bit)
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=use_4bit,
        )
        FastLanguageModel.for_inference(self.model)
        self._backend = "unsloth"
        logger.info("unsloth model loaded")

    def _load_transformers(self, use_4bit: bool) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore

        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            logger.warning("CUDA not available — loading model in fp32 on CPU (4-bit disabled)")
            use_4bit = False

        logger.info("Loading %s via transformers (4bit=%s, cuda=%s)", self.model_name, use_4bit, use_cuda)

        # Set padding token
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            token=self.hf_token,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant_cfg = None
        max_memory = None
        if use_4bit and use_cuda:
            try:
                quant_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                # Limit GPU to 5.5GB, offload excess layers to CPU
                max_memory = {0: "5.5GiB", "cpu": "32GiB"}
                logger.info("GPU memory limit: 5.5GiB, CPU offload enabled")
            except Exception as exc:
                logger.warning("bitsandbytes 4-bit config failed (%s) — loading in fp16", exc)

        dtype = torch.float16 if use_cuda else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quant_cfg,
            torch_dtype=dtype,
            device_map="auto" if use_cuda else "cpu",
            max_memory=max_memory,
            trust_remote_code=True,
            token=self.hf_token,
        )
        self.model.eval()
        self._backend = "transformers"

        # Log device allocation
        if hasattr(self.model, 'hf_device_map'):
            gpu_layers = sum(1 for d in self.model.hf_device_map.values() if d == 0)
            cpu_layers = sum(1 for d in self.model.hf_device_map.values() if d == 'cpu')
            logger.info("Model layers: GPU=%d, CPU=%d", gpu_layers, cpu_layers)
        logger.info("transformers model loaded (GPU mem: %.2f GB)", torch.cuda.memory_allocated() / 1e9 if use_cuda else 0)

    def generate(self, instruction: str) -> str:
        import torch
        import gc

        # Clean GPU memory before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # Use DeepSeek-Coder official format
        prompt = f"<|User|>: {instruction}\n<|Assistant|>:"
        
        try:
            # Tokenize with increased max_length to avoid truncating function prototypes
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            
            # CRITICAL FIX: When using device_map="auto" with CPU offload,
            # do NOT use self.model.device (it may return 'meta' or inconsistent device)
            # Instead, let accelerate handle device placement automatically
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # Ensure pad_token_id is set
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            # Reduce output length to minimize KV cache memory usage
            max_tokens = min(self.max_new_tokens, 512)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )

            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Remove the input prompt from the generated text
            if prompt in generated_text:
                generated_text = generated_text.split(prompt, 1)[1]
            elif "<|Assistant|>:" in generated_text:
                generated_text = generated_text.split("<|Assistant|>:", 1)[1]
                
            return generated_text.strip()

        except torch.cuda.OutOfMemoryError as e:
            logger.error("GPU OOM during generation: %s", e)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            return ""
        except Exception as e:
            logger.error("Generation error: %s", e)
            return ""
        finally:
            # Clean up GPU memory
            if 'inputs' in locals():
                del inputs
            if 'output_ids' in locals():
                del output_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

    def close(self) -> None:
        import torch, gc

        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
