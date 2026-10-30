import asyncio
from types import SimpleNamespace

import pytest

from slime.rollout.on_policy_distillation import post_process_rewards, reward_func, zero_reward_func
from slime.utils.types import Sample


def teacher_reward(tokens: list[int], log_probs: list[float | None]):
    assert len(tokens) == len(log_probs)
    return {
        "meta_info": {
            "input_token_logprobs": [
                [log_prob, token_id, f"token-{token_id}"]
                for token_id, log_prob in zip(tokens, log_probs, strict=True)
            ]
        }
    }


def test_post_process_extracts_response_suffix_and_returns_zero_task_reward():
    sample = Sample(
        tokens=[10, 11, 12, 13],
        response_length=2,
        reward=teacher_reward([10, 11, 12, 13], [None, -0.1, -0.4, -0.7]),
    )

    rewards, raw_rewards = post_process_rewards(SimpleNamespace(reward_key=None), [sample])

    assert rewards == [0.0]
    assert raw_rewards == [0.0]
    assert sample.teacher_log_probs == pytest.approx([-0.4, -0.7])


def test_post_process_accepts_response_only_teacher_scores():
    sample = Sample(
        tokens=[10, 11, 12, 13],
        response_length=2,
        reward=teacher_reward([12, 13], [-0.4, -0.7]),
    )

    rewards, raw_rewards = post_process_rewards(SimpleNamespace(reward_key=None), [sample])

    assert rewards == [0.0]
    assert raw_rewards == [0.0]
    assert sample.teacher_log_probs == pytest.approx([-0.4, -0.7])


def test_post_process_accepts_one_leading_context_token():
    sample = Sample(
        tokens=[10, 11, 12, 13],
        response_length=2,
        reward=teacher_reward([11, 12, 13], [None, -0.4, -0.7]),
    )

    post_process_rewards(SimpleNamespace(reward_key=None), [sample])

    assert sample.teacher_log_probs == pytest.approx([-0.4, -0.7])


def test_zero_length_response_stays_empty():
    sample = Sample(
        tokens=[10, 11, 12],
        response_length=0,
        reward=teacher_reward([10, 11, 12], [None, -0.1, -0.2]),
    )

    post_process_rewards(SimpleNamespace(reward_key=None), [sample])

    assert sample.teacher_log_probs == []


@pytest.mark.parametrize(
    ("reward", "message"),
    [
        (None, "must be a JSON object"),
        ({}, "missing object field 'meta_info'"),
        ({"meta_info": {}}, "missing list field 'meta_info.input_token_logprobs'"),
        ({"meta_info": {"input_token_logprobs": []}}, "contains 0 token entries"),
        (
            {"meta_info": {"input_token_logprobs": [[None, 10], [-0.1]]}},
            "malformed input_token_logprobs entry",
        ),
        (
            {"meta_info": {"input_token_logprobs": [[None, 10], ["bad", 11]]}},
            "non-numeric log-probability",
        ),
        (
            {"meta_info": {"input_token_logprobs": [[None, 10], [float("nan"), 11]]}},
            "non-finite log-probability",
        ),
        (
            {"meta_info": {"input_token_logprobs": [[None, 10], [-0.1, 99]]}},
            "token mismatch",
        ),
    ],
)
def test_malformed_teacher_response_is_rejected(reward, message):
    sample = Sample(tokens=[10, 11], response_length=1, reward=reward)

    with pytest.raises(ValueError, match=message):
        post_process_rewards(SimpleNamespace(reward_key=None), [sample])

    assert sample.teacher_log_probs is None


def test_batch_is_not_partially_mutated():
    valid = Sample(
        tokens=[10, 11],
        response_length=1,
        reward=teacher_reward([10, 11], [None, -0.1]),
    )
    invalid = Sample(
        tokens=[20, 21],
        response_length=1,
        reward=teacher_reward([20, 99], [None, -0.2]),
    )

    with pytest.raises(ValueError, match="sample 1 token mismatch"):
        post_process_rewards(SimpleNamespace(reward_key=None), [valid, invalid])

    assert valid.teacher_log_probs is None
    assert invalid.teacher_log_probs is None


def test_post_process_accepts_teacher_log_probs_attached_during_reward_call():
    sample = Sample(tokens=[10, 11], response_length=1, reward=0.0, teacher_log_probs=[-0.25])

    rewards, raw_rewards = post_process_rewards(SimpleNamespace(reward_key=None), [sample])

    assert rewards == [0.0]
    assert raw_rewards == [0.0]
    assert sample.teacher_log_probs == [-0.25]


@pytest.mark.parametrize("teacher_log_probs", [[], [float("nan")], [True]])
def test_post_process_rejects_invalid_attached_teacher_log_probs(teacher_log_probs):
    sample = Sample(tokens=[10, 11], response_length=1, reward=0.0, teacher_log_probs=teacher_log_probs)

    with pytest.raises(ValueError, match="teacher log-probabilities"):
        post_process_rewards(SimpleNamespace(reward_key=None), [sample])


def test_megatron_pure_opd_reward_is_zero():
    reward = asyncio.run(zero_reward_func(SimpleNamespace(), Sample()))
    assert reward == 0.0


def test_sglang_request_uses_only_sampled_token_log_probs(monkeypatch):
    posted = {}

    class FakeResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        def raise_for_status(self):
            return None

        async def json(self):
            return teacher_reward([10, 11], [None, -0.3])

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        def post(self, url, json):
            posted["url"] = url
            posted["payload"] = json
            return FakeResponse()

    monkeypatch.setattr("aiohttp.ClientSession", FakeSession)
    args = SimpleNamespace(rm_url="http://teacher.example/generate")
    sample = Sample(tokens=[10, 11], response_length=1)

    result = asyncio.run(reward_func(args, sample))

    assert result == 0.0
    assert sample.teacher_log_probs == [-0.3]
    assert posted["url"] == args.rm_url
    assert posted["payload"]["input_ids"] == [10, 11]
    assert posted["payload"]["sampling_params"]["temperature"] == 0
    assert posted["payload"]["logprob_start_len"] == 0


def test_sglang_teacher_scoring_retries_invalid_token_ids(monkeypatch):
    attempts = 0

    class FakeResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        def raise_for_status(self):
            return None

        async def json(self):
            nonlocal attempts
            attempts += 1
            token_id = 0 if attempts == 1 else 11
            return teacher_reward([10, token_id], [None, -0.3])

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        def post(self, url, json):
            return FakeResponse()

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr("aiohttp.ClientSession", FakeSession)
    monkeypatch.setattr("slime.rollout.on_policy_distillation.asyncio.sleep", no_sleep)
    sample = Sample(tokens=[10, 11], response_length=1, index=5382)

    result = asyncio.run(reward_func(SimpleNamespace(rm_url="http://teacher/generate"), sample))

    assert result == 0.0
    assert attempts == 2
    assert sample.teacher_log_probs == [-0.3]


def test_sglang_teacher_scoring_masks_persistent_token_mismatch(monkeypatch):
    attempts = 0

    class FakeResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        def raise_for_status(self):
            return None

        async def json(self):
            nonlocal attempts
            attempts += 1
            return teacher_reward([10, 11, 0, 13], [None, -0.1, -0.2, -0.3])

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        def post(self, url, json):
            return FakeResponse()

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr("aiohttp.ClientSession", FakeSession)
    monkeypatch.setattr("slime.rollout.on_policy_distillation.asyncio.sleep", no_sleep)
    sample = Sample(tokens=[10, 11, 12, 13], response_length=3, index=7284)

    result = asyncio.run(reward_func(SimpleNamespace(rm_url="http://teacher/generate"), sample))

    assert result == 0.0
    assert attempts == 3
    assert sample.teacher_log_probs == [-0.1, 0.0, -0.3]
    assert sample.loss_mask == [1, 0, 1]
    assert sample.metadata["teacher_score_token_mismatch_count"] == 1
    assert sample.metadata["teacher_score_first_mismatch_position"] == 1
