import threading
from enum import Enum
from mini_vllm.utils.config import SamplingParams


class Status(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


class Sequence:
    _next_id = 0
    _id_lock = threading.Lock()

    def __init__(
        self,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams | None = None,
        seq_id: int | None = None,
    ):
        if seq_id is not None:
            self.seq_id = seq_id
        else:
            with Sequence._id_lock:
                self.seq_id = Sequence._next_id
                Sequence._next_id += 1

        self.token_ids: list[int] = list(prompt_token_ids)
        self.prompt_len: int = len(prompt_token_ids)
        self.output_len: int = 0
        self.status: Status = Status.WAITING
        self.sampling_params = sampling_params or SamplingParams()
        self.block_table: list[int] = []
        self._stop_token_ids: set[int] = set()
        self._stop_strings: list[str] = []
        # Incremental decoded output cache for stop string detection
        self._decoded_output: str = ""
        self._decoded_len: int = 0  # number of output tokens already decoded

    def set_stop_conditions(self, eos_token_ids: int | list[int], tokenizer=None) -> None:
        """Configure EOS token IDs and stop strings."""
        if isinstance(eos_token_ids, int):
            self._stop_token_ids = {eos_token_ids}
        else:
            self._stop_token_ids = set(eos_token_ids)
        self._stop_strings = list(self.sampling_params.stop) if self.sampling_params.stop else []
        self._tokenizer = tokenizer

    def get_len(self) -> int:
        return self.prompt_len + self.output_len

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)
        self.output_len += 1

    def _update_decoded_output(self) -> None:
        """Incrementally decode only the new tokens since last call."""
        if self._stop_strings and self._tokenizer is not None:
            new_tokens = self.token_ids[self.prompt_len + self._decoded_len:]
            if new_tokens:
                new_text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
                self._decoded_output += new_text
                self._decoded_len = self.output_len

    def is_finished(self) -> bool:
        if self.status == Status.FINISHED:
            return True
        if self.output_len >= self.sampling_params.max_tokens:
            return True
        if self.output_len > 0 and self.token_ids[-1] in self._stop_token_ids:
            return True
        if self._stop_strings:
            self._update_decoded_output()
            for stop_str in self._stop_strings:
                if stop_str in self._decoded_output:
                    return True
        return False

    def get_last_token_id(self) -> int:
        return self.token_ids[-1]

    def get_prompt_token_ids(self) -> list[int]:
        return self.token_ids[: self.prompt_len]

    def get_output_token_ids(self) -> list[int]:
        return self.token_ids[self.prompt_len :]
