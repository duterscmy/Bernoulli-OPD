"""Pure sampled-token OPD reward helpers."""

import asyncio
import logging
import math

from slime.utils.types import Sample


TEACHER_SCORING_MAX_ATTEMPTS = 3


class TeacherTokenMismatchError(ValueError):
    """One or more teacher score entries do not describe the requested token."""


async def reward_func(args, sample, **kwargs):
    """Score a completed student sequence with an external SGLang teacher."""
    import aiohttp

    from slime.utils.processing_utils import encode_image_for_rollout_engine

    if sample.response_length == 0:
        sample.teacher_log_probs = []
        return 0.0

    response_start = len(sample.tokens) - sample.response_length
    payload = {
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        # Include the token immediately before the response. SGLang returns a
        # None log-probability at logprob_start_len itself; starting one token
        # earlier makes the first response token's conditional log-probability
        # available while still avoiding serialization of the full prompt.
        "logprob_start_len": max(0, response_start - 1),
    }

    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        image_data = sample.multimodal_inputs["images"]
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    # Rollout metrics run before ``post_process_rewards`` in current Slime and
    # expect ``sample.reward`` to be scalar.  Store the teacher signal on the
    # sample here, then return the pure-OPD task reward (zero) instead of the
    # structured SGLang response.
    sample_index = sample.index if sample.index is not None else -1
    for attempt in range(1, TEACHER_SCORING_MAX_ATTEMPTS + 1):
        async with aiohttp.ClientSession() as session:
            async with session.post(args.rm_url, json=payload) as resp:
                resp.raise_for_status()
                teacher_response = await resp.json()
        try:
            sample.teacher_log_probs = _extract_teacher_response_log_probs(
                sample, teacher_response, sample_index
            )
            break
        except ValueError as exc:
            if attempt == TEACHER_SCORING_MAX_ATTEMPTS:
                if isinstance(exc, TeacherTokenMismatchError):
                    sample.teacher_log_probs = _extract_teacher_response_log_probs(
                        sample,
                        teacher_response,
                        sample_index,
                        mask_token_mismatches=True,
                    )
                    break
                raise
            logging.warning(
                "Invalid teacher score for sample %s; retrying (%d/%d).",
                sample_index,
                attempt + 1,
                TEACHER_SCORING_MAX_ATTEMPTS,
                exc_info=True,
            )
            await asyncio.sleep(0.25 * attempt)
    return 0.0


async def zero_reward_func(args, sample, **kwargs):
    """Return no task reward for pure OPD with a Megatron teacher."""
    return 0.0


def _extract_teacher_response_log_probs(
    sample: Sample,
    reward: dict,
    sample_index: int,
    *,
    mask_token_mismatches: bool = False,
) -> list[float]:
    """Validate one SGLang response and return response-token log-probs."""
    context = f"teacher response for sample {sample_index}"

    if not isinstance(reward, dict):
        raise ValueError(f"{context} must be a JSON object, got {type(reward).__name__}.")
    meta_info = reward.get("meta_info")
    if not isinstance(meta_info, dict):
        raise ValueError(f"{context} is missing object field 'meta_info'.")
    token_log_probs = meta_info.get("input_token_logprobs")
    if not isinstance(token_log_probs, (list, tuple)):
        raise ValueError(f"{context} is missing list field 'meta_info.input_token_logprobs'.")

    response_length = sample.response_length
    if not isinstance(response_length, int) or isinstance(response_length, bool) or response_length < 0:
        raise ValueError(f"sample {sample_index} has invalid response_length={response_length!r}.")
    if response_length > len(sample.tokens):
        raise ValueError(
            f"sample {sample_index} has response_length={response_length}, "
            f"but only {len(sample.tokens)} total tokens."
        )
    allowed_lengths = {response_length, response_length + 1, len(sample.tokens)}
    if len(token_log_probs) in allowed_lengths:
        # Backward compatibility for saved/debug responses collected with
        # logprob_start_len=0, plus current responses containing one leading
        # context token.
        response_entries = token_log_probs[-response_length:] if response_length else []
    else:
        raise ValueError(
            f"{context} contains {len(token_log_probs)} token entries, "
            f"expected {response_length} response entries, {response_length + 1} "
            f"entries with one leading context token, or {len(sample.tokens)} "
            "full-sequence entries."
        )
    if response_length == 0:
        return []

    response_tokens = sample.tokens[-response_length:]
    values = []
    mismatched_positions = []
    for position, (entry, expected_token) in enumerate(zip(response_entries, response_tokens, strict=True)):
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            raise ValueError(
                f"{context} has malformed input_token_logprobs entry at response position {position}; "
                "expected [log_prob, token_id, ...]."
            )
        log_prob, returned_token = entry[:2]
        if isinstance(log_prob, bool) or not isinstance(log_prob, (int, float)):
            raise ValueError(
                f"{context} has a non-numeric log-probability at response position {position}: {log_prob!r}."
            )
        if not math.isfinite(log_prob):
            raise ValueError(
                f"{context} has a non-finite log-probability at response position {position}: {log_prob!r}."
            )
        if isinstance(returned_token, bool) or not isinstance(returned_token, int):
            raise ValueError(f"{context} has an invalid token id at response position {position}: {returned_token!r}.")
        if returned_token != expected_token:
            if not mask_token_mismatches:
                raise TeacherTokenMismatchError(
                    f"{context} token mismatch at response position {position}: "
                    f"expected {expected_token}, got {returned_token}."
                )
            mismatched_positions.append(position)
            # Keep tensor/list shapes aligned with the response. The zero is a
            # finite placeholder only; the corresponding loss-mask entry is
            # cleared below, so it cannot contribute to the OPD objective.
            values.append(0.0)
        else:
            values.append(float(log_prob))

    if mismatched_positions:
        if sample.loss_mask is None:
            sample.loss_mask = [1] * response_length
        elif len(sample.loss_mask) != response_length:
            raise ValueError(
                f"sample {sample_index} has loss_mask length {len(sample.loss_mask)}, "
                f"expected response_length={response_length}."
            )
        else:
            sample.loss_mask = list(sample.loss_mask)
        for position in mismatched_positions:
            sample.loss_mask[position] = 0

        if sample.metadata is None:
            sample.metadata = {}
        sample.metadata["teacher_score_token_mismatch_count"] = len(mismatched_positions)
        sample.metadata["teacher_score_first_mismatch_position"] = mismatched_positions[0]
        sample.metadata["teacher_score_last_mismatch_position"] = mismatched_positions[-1]
        logging.warning(
            "Masked %d persistent teacher token mismatch(es) for sample %s "
            "at response positions %d..%d after %d scoring attempts.",
            len(mismatched_positions),
            sample_index,
            mismatched_positions[0],
            mismatched_positions[-1],
            TEACHER_SCORING_MAX_ATTEMPTS,
        )

    return values


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Attach validated teacher log-probs and return zero task rewards."""
    teacher_log_probs = []
    for sample_index, sample in enumerate(samples):
        if sample.teacher_log_probs is None:
            # Backward compatibility for saved/debug rollouts produced by the
            # older hook, where the raw SGLang response lived in ``reward``.
            values = _extract_teacher_response_log_probs(sample, sample.get_reward_value(args), sample_index)
        else:
            values = sample.teacher_log_probs
            if len(values) != sample.response_length:
                raise ValueError(
                    f"sample {sample_index} contains {len(values)} teacher log-probabilities, "
                    f"expected {sample.response_length}."
                )
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in values
            ):
                raise ValueError(f"sample {sample_index} contains invalid attached teacher log-probabilities.")
            values = [float(value) for value in values]
        teacher_log_probs.append(values)

    for sample, token_log_probs in zip(samples, teacher_log_probs, strict=True):
        sample.teacher_log_probs = token_log_probs

    scalar_rewards = [0.0] * len(samples)
    return scalar_rewards, scalar_rewards
