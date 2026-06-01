# mini_vllm/engine/scheduler.py
from dataclasses import dataclass, field
from collections import deque
from mini_vllm.engine.sequence import Sequence, Status
from mini_vllm.utils.config import EngineConfig
from mini_vllm.utils.logger import logger


@dataclass
class SchedulerOutput:
    prefill_seqs: list[Sequence] = field(default_factory=list)
    decode_seqs: list[Sequence] = field(default_factory=list)
    num_new_tokens: int = 0


class Scheduler:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.waiting_queue: deque[Sequence] = deque()
        self.running_seqs: dict[int, Sequence] = {}
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens

    def add_sequence(self, seq: Sequence) -> None:
        self.waiting_queue.append(seq)

    def schedule(self) -> SchedulerOutput:
        output = SchedulerOutput()

        # 1. Evict finished sequences from running set to free slots
        finished_ids = [sid for sid, s in self.running_seqs.items() if s.is_finished()]
        for sid in finished_ids:
            self.running_seqs.pop(sid)

        # 2. Continue running sequences (decode)
        for seq in list(self.running_seqs.values()):
            if seq.is_finished():
                continue
            output.decode_seqs.append(seq)
            output.num_new_tokens += 1

        # 3. Take new sequences from waiting queue (prefill)
        num_running = len(output.decode_seqs)
        while self.waiting_queue and num_running < self.max_num_seqs:
            seq = self.waiting_queue[0]
            new_tokens = seq.prompt_len + 1  # prefill prompt + first decode token
            if output.num_new_tokens + new_tokens > self.max_num_batched_tokens:
                break

            self.waiting_queue.popleft()
            seq.status = Status.RUNNING
            self.running_seqs[seq.seq_id] = seq
            output.prefill_seqs.append(seq)
            output.num_new_tokens += new_tokens
            num_running += 1

        return output

    def on_complete(self, seq_id: int) -> None:
        if seq_id in self.running_seqs:
            seq = self.running_seqs.pop(seq_id)
            seq.status = Status.FINISHED
            logger.debug(f"Sequence {seq_id} completed")

    @property
    def has_unfinished(self) -> bool:
        return bool(self.waiting_queue) or any(
            not s.is_finished() for s in self.running_seqs.values()
        )
