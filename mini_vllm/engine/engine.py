# mini_vllm/engine/engine.py
import copy
import json
import threading
import torch
from pathlib import Path
from dataclasses import dataclass, field
from mini_vllm.engine.sequence import Sequence, Status
from mini_vllm.engine.scheduler import Scheduler
from mini_vllm.engine.kv_cache import KVCacheManager
from mini_vllm.engine.sampler import Sampler
from mini_vllm.utils.config import EngineConfig, SamplingParams
from mini_vllm.utils.logger import logger


@dataclass
class GenerationOutput:
    prompt_text: str = ""
    prompt_token_ids: list[int] = field(default_factory=list)
    output_token_ids: list[int] = field(default_factory=list)
    output_text: str = ""
    finished: bool = False


class LLMEngine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self._lock = threading.Lock()
        self.device = config.resolve_device()
        self.dtype = config.get_torch_dtype()

        # Load model
        from mini_vllm.models.loader import load_model
        self.model, self.model_config = load_model(
            config.model_path, device=self.device, dtype=self.dtype
        )

        # 自动检测模型权重精度，而非依赖配置
        self._torch_dtype = next(self.model.parameters()).dtype
        logger.info(f"Model dtype auto-detected: {self._torch_dtype}")

        # Calculate model memory footprint for KV cache budget
        model_memory = sum(p.numel() * p.element_size() for p in self.model.parameters())
        model_memory += sum(b.numel() * b.element_size() for b in self.model.buffers())
        logger.info(f"Model memory: {model_memory / 1024**2:.0f} MB")

        # Optional: compile model for faster decode (PyTorch 2.x+)
        if config.compile and self.device == "cuda":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead", dynamic=True)
                logger.info("Model compiled with torch.compile (reduce-overhead)")
            except Exception as e:
                logger.warning(f"torch.compile failed, using eager mode: {e}")

        # Free any cached memory from model loading before computing KV cache budget
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # Initialize KV Cache
        num_blocks = config.get_num_gpu_blocks(
            head_dim=self.model_config.get(
                "head_dim",
                self.model_config["hidden_size"] // self.model_config["num_attention_heads"],
            ),
            num_layers=self.model_config["num_hidden_layers"],
            num_heads=self.model_config.get(
                "num_key_value_heads", self.model_config["num_attention_heads"]
            ),
            model_memory_bytes=model_memory,
        )
        self.kv_cache = KVCacheManager(
            num_blocks=num_blocks,
            block_size=config.block_size,
            num_layers=self.model_config["num_hidden_layers"],
            num_heads=self.model_config.get(
                "num_key_value_heads", self.model_config["num_attention_heads"]
            ),
            head_dim=self.model_config.get(
                "head_dim",
                self.model_config["hidden_size"] // self.model_config["num_attention_heads"],
            ),
            device=self.device,
            dtype=self._torch_dtype,
        )

        # Initialize tokenizer
        from mini_vllm.tokenizer.tokenizer import TokenizerWrapper
        self.tokenizer = TokenizerWrapper(config.model_path)

        # Initialize scheduler and sampler
        self.scheduler = Scheduler(config)
        self.sampler = Sampler()

        # EOS token - auto-detect from model config, fall back to tokenizer
        self.eos_token_ids = self._detect_eos_tokens(self.model_config, config.model_path, self.tokenizer)

        logger.info("LLMEngine initialized successfully")

    @staticmethod
    def _detect_eos_tokens(model_config: dict, model_path: str, tokenizer) -> list[int]:
        """Auto-detect EOS tokens: generation_config.json > config.json > tokenizer.
        Returns a list of all valid EOS token IDs."""
        eos_list: list[int] = []

        # 1. Try generation_config.json
        gen_config_path = Path(model_path) / "generation_config.json"
        if gen_config_path.exists():
            try:
                with open(gen_config_path) as f:
                    gen_config = json.load(f)
                eos = gen_config.get("eos_token_id")
                if isinstance(eos, int):
                    eos_list.append(eos)
                elif isinstance(eos, list):
                    eos_list.extend(e for e in eos if isinstance(e, int))
            except (json.JSONDecodeError, KeyError):
                pass

        # 2. Try config.json
        if not eos_list:
            eos = model_config.get("eos_token_id")
            if isinstance(eos, int):
                eos_list.append(eos)
            elif isinstance(eos, list):
                eos_list.extend(e for e in eos if isinstance(e, int))

        # 3. Fall back to tokenizer
        if not eos_list and tokenizer.eos_token_id is not None:
            eos_list.append(tokenizer.eos_token_id)

        return eos_list

    def generate(
        self, prompts: list[str], sampling_params: SamplingParams
    ) -> list[GenerationOutput]:
        """Synchronous generation interface"""
        if not hasattr(self, "_lock"):
            self._lock = threading.Lock()
        with self._lock:
            return self._generate_locked(prompts, sampling_params)

    def _generate_locked(
        self, prompts: list[str], sampling_params: SamplingParams
    ) -> list[GenerationOutput]:
        """Internal generate — must hold self._lock."""
        # Reset linear attention cache for models that have it (e.g. Qwen3.5)
        if hasattr(self.model, 'reset_cache'):
            self.model.reset_cache()

        # 1. Create Sequences
        sequences = []
        for prompt in prompts:
            token_ids = self.tokenizer.encode(prompt)
            # Copy sampling_params per sequence to avoid shared mutation
            params = copy.deepcopy(sampling_params)
            if self.eos_token_ids:
                params.eos_token_id = self.eos_token_ids
            seq = Sequence(prompt_token_ids=token_ids, sampling_params=params)
            seq.set_stop_conditions(self.eos_token_ids, self.tokenizer)
            self.scheduler.add_sequence(seq)
            sequences.append(seq)

        try:
            # 2. Inference loop
            while self.scheduler.has_unfinished:
                self._step()
        finally:
            # 3. Free KV cache for all sequences (even unfinished, in case of exception)
            for seq in sequences:
                if self.kv_cache is not None:
                    self.kv_cache.free(seq.seq_id)
                self.scheduler.on_complete(seq.seq_id)

        # 4. Collect results
        outputs = []
        for i, seq in enumerate(sequences):
            output_token_ids = seq.get_output_token_ids()
            output_text = self.tokenizer.decode(output_token_ids)
            # Strip stop strings from the end of output
            for stop_str in seq._stop_strings:
                if output_text.endswith(stop_str):
                    output_text = output_text[:-len(stop_str)]
                    break
            outputs.append(
                GenerationOutput(
                    prompt_text=prompts[i],
                    prompt_token_ids=seq.get_prompt_token_ids(),
                    output_token_ids=output_token_ids,
                    output_text=output_text,
                    finished=seq.is_finished(),
                )
            )

        return outputs

    def generate_stream(
        self, prompts: list[str], sampling_params: SamplingParams
    ):
        """Streaming generation: yields (index, token_text) as tokens are generated.
        Caller must hold self._lock for the entire iteration."""
        # Reset linear attention cache for models that have it (e.g. Qwen3.5)
        if hasattr(self.model, 'reset_cache'):
            self.model.reset_cache()

        # 1. Create sequences
        sequences = []
        for prompt in prompts:
            token_ids = self.tokenizer.encode(prompt)
            params = copy.deepcopy(sampling_params)
            if self.eos_token_ids:
                params.eos_token_id = self.eos_token_ids
            seq = Sequence(prompt_token_ids=token_ids, sampling_params=params)
            seq.set_stop_conditions(self.eos_token_ids, self.tokenizer)
            self.scheduler.add_sequence(seq)
            sequences.append(seq)

        # 2. Stream tokens
        # Track cumulative decoded text per sequence to avoid garbled output
        # from decoding individual SentencePiece/BPE tokens in isolation.
        prev_output_len = [0] * len(sequences)
        decoded_text = [""] * len(sequences)
        try:
            while True:
                self._step()
                for i, seq in enumerate(sequences):
                    if seq.output_len > prev_output_len[i]:
                        full_text = self.tokenizer.decode(seq.get_output_token_ids())
                        token_text = full_text[len(decoded_text[i]):]
                        decoded_text[i] = full_text
                        prev_output_len[i] = seq.output_len
                        yield i, token_text, seq.is_finished()
                if not self.scheduler.has_unfinished:
                    break
        finally:
            # 3. Cleanup — free all allocated KV cache blocks (idempotent)
            for seq in sequences:
                if self.kv_cache is not None:
                    self.kv_cache.free(seq.seq_id)
                self.scheduler.on_complete(seq.seq_id)

    def _step(self) -> None:
        """Execute one step of inference"""
        scheduler_output = self.scheduler.schedule()

        # Prefill phase
        for seq in scheduler_output.prefill_seqs:
            self._run_prefill(seq)

        # Decode phase
        if scheduler_output.decode_seqs:
            self._run_decode(scheduler_output.decode_seqs)

    def _run_prefill(self, seq: Sequence) -> None:
        """Execute prefill: process full prompt"""
        # Reset linear attention cache before each new sequence (e.g. Qwen3.5 hybrid attention)
        if hasattr(self.model, 'reset_cache'):
            self.model.reset_cache()

        block_table = None
        kv_cache_tensor = None
        if self.kv_cache is not None:
            block_table = self.kv_cache.allocate(seq.seq_id, seq.prompt_len + 1)
            if block_table is None:
                logger.error(f"Failed to allocate KV cache for sequence {seq.seq_id} (prompt_len={seq.prompt_len})")
                seq.status = Status.FINISHED
                self.scheduler.on_complete(seq.seq_id)
                return
            seq.block_table = block_table
            kv_cache_tensor = self.kv_cache.kv_cache

        input_ids = torch.tensor(
            [seq.get_prompt_token_ids()], device=self.device, dtype=torch.long
        )
        positions = torch.arange(
            seq.prompt_len, device=self.device, dtype=torch.long
        )

        with torch.no_grad(), torch.autocast(device_type=self.device, dtype=self._torch_dtype):
            logits = self.model(
                input_ids, positions,
                kv_cache=kv_cache_tensor,
                block_tables=block_table,
                slot_idx=0 if block_table else None,
                use_cache=False,
            )

        # Sample first output token
        last_logits = logits[:, -1, :].float()  # sampler needs float32
        token_id = self.sampler.sample(last_logits, seq.sampling_params)[0]
        seq.append_token(token_id)

        if seq.is_finished():
            if self.kv_cache is not None:
                self.kv_cache.free(seq.seq_id)
            self.scheduler.on_complete(seq.seq_id)

    def _run_decode(self, sequences: list[Sequence]) -> None:
        """Execute decode: batch all sequences into one forward pass."""
        # 1. Expand KV cache for all sequences
        valid_seqs = []
        for seq in sequences:
            if self.kv_cache is not None:
                if not self.kv_cache.append_token(seq.seq_id):
                    logger.error(f"Failed to expand KV cache for sequence {seq.seq_id}")
                    seq.status = Status.FINISHED
                    self.kv_cache.free(seq.seq_id)
                    self.scheduler.on_complete(seq.seq_id)
                    continue
            valid_seqs.append(seq)

        if not valid_seqs:
            return

        # 2. Build batch inputs
        batch_ids = [[seq.get_last_token_id()] for seq in valid_seqs]
        batch_pos = [[seq.get_len() - 1] for seq in valid_seqs]
        input_ids = torch.tensor(batch_ids, device=self.device, dtype=torch.long)
        positions = torch.tensor(batch_pos, device=self.device, dtype=torch.long)

        # 3. Collect per-sequence metadata for batched KV cache access
        block_tables_list = None
        slot_idx_list = None
        num_cached_list = None
        if self.kv_cache is not None:
            block_tables_list = [self.kv_cache.get_block_table(seq.seq_id) for seq in valid_seqs]
            slot_idx_list = [seq.get_len() - 1 for seq in valid_seqs]
            num_cached_list = [seq.get_len() - 1 for seq in valid_seqs]

            # Build causal attention mask for batch with variable cache lengths
            # +1 to account for the token that will be written during the forward pass
            max_cached = max(nc + 1 for nc in num_cached_list)
            q_len = 1
            # Use float mask: 0.0 = attend, -inf = blocked
            attn_mask = torch.zeros(len(valid_seqs), 1, q_len, max_cached, device=self.device, dtype=self._torch_dtype)
            for i, nc in enumerate(num_cached_list):
                if nc + 1 < max_cached:
                    attn_mask[i, :, :, nc + 1:] = float('-inf')  # block padding positions

        # 4. Single batched forward pass
        with torch.no_grad(), torch.autocast(device_type=self.device, dtype=self._torch_dtype):
            logits = self.model(
                input_ids, positions,
                kv_cache=self.kv_cache.kv_cache if self.kv_cache else None,
                block_tables=block_tables_list,
                slot_idx=slot_idx_list,
                use_cache=block_tables_list is not None,
                num_cached_tokens_list=num_cached_list,
                attn_mask=attn_mask if self.kv_cache else None,
            )

        # 5. Sample and update sequences
        for i, seq in enumerate(valid_seqs):
            token_logits = logits[i, -1, :].float().unsqueeze(0)
            token_id = self.sampler.sample(token_logits, seq.sampling_params)[0]
            seq.append_token(token_id)

            if seq.is_finished():
                if self.kv_cache is not None:
                    self.kv_cache.free(seq.seq_id)
                self.scheduler.on_complete(seq.seq_id)
