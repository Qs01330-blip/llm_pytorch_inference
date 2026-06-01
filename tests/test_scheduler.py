# tests/test_scheduler.py
from mini_vllm.engine.scheduler import Scheduler, SchedulerOutput
from mini_vllm.engine.sequence import Sequence, Status
from mini_vllm.utils.config import EngineConfig, SamplingParams


def make_config(**kwargs) -> EngineConfig:
    defaults = {"model_path": "/tmp/model", "device": "cpu", "max_num_seqs": 4, "max_num_batched_tokens": 16}
    defaults.update(kwargs)
    return EngineConfig(**defaults)


def test_scheduler_add_and_schedule():
    config = make_config()
    scheduler = Scheduler(config)
    seq = Sequence(prompt_token_ids=[1, 2, 3], sampling_params=SamplingParams(max_tokens=10))
    scheduler.add_sequence(seq)

    output = scheduler.schedule()
    assert len(output.prefill_seqs) == 1
    assert seq.status == Status.RUNNING


def test_scheduler_respects_max_num_seqs():
    config = make_config(max_num_seqs=2)
    scheduler = Scheduler(config)

    for i in range(3):
        seq = Sequence(prompt_token_ids=[1, 2], sampling_params=SamplingParams(max_tokens=10))
        scheduler.add_sequence(seq)

    output = scheduler.schedule()
    assert len(output.prefill_seqs) <= 2


def test_scheduler_running_continue():
    config = make_config()
    scheduler = Scheduler(config)
    seq = Sequence(prompt_token_ids=[1, 2], sampling_params=SamplingParams(max_tokens=10))
    scheduler.add_sequence(seq)

    # Step 1: prefill
    output1 = scheduler.schedule()
    assert len(output1.prefill_seqs) == 1

    # Step 2: decode continues
    output2 = scheduler.schedule()
    assert seq in output2.decode_seqs


def test_scheduler_finish():
    config = make_config()
    scheduler = Scheduler(config)
    seq = Sequence(prompt_token_ids=[1], sampling_params=SamplingParams(max_tokens=1))
    scheduler.add_sequence(seq)

    output = scheduler.schedule()
    assert len(output.prefill_seqs) == 1

    # Simulate completion
    seq.append_token(99)
    seq.status = Status.FINISHED
    scheduler.on_complete(seq.seq_id)
    assert seq.seq_id not in [s.seq_id for s in scheduler.running_seqs]
