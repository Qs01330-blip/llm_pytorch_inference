# tests/test_sequence.py
from mini_vllm.engine.sequence import Sequence, Status
from mini_vllm.utils.config import SamplingParams


def test_sequence_creation():
    seq = Sequence(seq_id=1, prompt_token_ids=[101, 102, 103])
    assert seq.seq_id == 1
    assert seq.status == Status.WAITING
    assert seq.prompt_len == 3
    assert seq.output_len == 0
    assert seq.get_len() == 3


def test_sequence_append_token():
    seq = Sequence(seq_id=1, prompt_token_ids=[101, 102])
    seq.append_token(200)
    assert seq.output_len == 1
    assert seq.get_len() == 3
    assert seq.token_ids[-1] == 200


def test_sequence_is_finished():
    params = SamplingParams(max_tokens=2)
    seq = Sequence(seq_id=1, prompt_token_ids=[101], sampling_params=params)
    seq.append_token(200)
    assert not seq.is_finished()
    seq.append_token(201)
    assert seq.is_finished()


def test_sequence_eos():
    params = SamplingParams()
    seq = Sequence(seq_id=1, prompt_token_ids=[101], sampling_params=params)
    seq.set_stop_conditions(params.eos_token_id)
    seq.append_token(params.eos_token_id)
    assert seq.is_finished()


def test_sequence_multi_eos():
    """Multiple EOS tokens should all trigger stop."""
    params = SamplingParams()
    seq = Sequence(seq_id=2, prompt_token_ids=[101], sampling_params=params)
    seq.set_stop_conditions([151645, 151643])
    seq.append_token(151643)
    assert seq.is_finished()


def test_sequence_stop_string():
    """Stop strings should trigger stop."""
    class MockTokenizer:
        def decode(self, token_ids, skip_special_tokens=True):
            return "".join(chr(t) for t in token_ids)

    params = SamplingParams(stop=["\nUser:"])
    seq = Sequence(seq_id=3, prompt_token_ids=[101], sampling_params=params)
    seq.set_stop_conditions([99], tokenizer=MockTokenizer())
    seq.append_token(ord("\n"))
    seq.append_token(ord("U"))
    seq.append_token(ord("s"))
    seq.append_token(ord("e"))
    seq.append_token(ord("r"))
    seq.append_token(ord(":"))
    assert seq.is_finished()
