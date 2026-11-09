"""Baseline BOPD token selection and advantages.

BOPD operates on the response tokens in one data-parallel worker's local
packed batch. It selects the lowest current-actor sampled-token log-probabilities
across that batch and assigns advantages only to those selected tokens.
"""

import math
from dataclasses import dataclass

import torch


def validate_bopd_args(args) -> None:
    """Validate baseline BOPD CLI values and derive entropy requirements."""
    if args.advantage_estimator != "bopd":
        return

    if args.use_opd:
        raise ValueError("--advantage-estimator=bopd and --use-opd are mutually exclusive.")
    if args.normalize_advantages:
        raise ValueError("--advantage-estimator=bopd and --normalize-advantages are mutually exclusive.")
    if args.use_rollout_logprobs:
        raise ValueError(
            "BOPD requires log-probs recomputed by the current Megatron actor; "
            "disable --use-rollout-logprobs."
        )
    if args.kl_coef != 0 or args.use_kl_loss:
        raise ValueError("BOPD is reference-free; --kl-coef must be 0 and --use-kl-loss must be disabled.")
    if not 0 < args.bopd_token_fraction <= 1:
        raise ValueError("--bopd-token-fraction must be in (0, 1].")

    if args.bopd_mode == "entropy":
        if not args.bopd_advantage_min < args.bopd_advantage_max < 0:
            raise ValueError("--bopd-advantage-min/max must satisfy min < max < 0.")
        if args.bopd_fixed_advantage is not None:
            raise ValueError("--bopd-fixed-advantage is only valid with --bopd-mode=fixed.")
        args.use_rollout_entropy = True
    elif args.bopd_mode == "fixed":
        if args.entropy_coef != 0:
            raise ValueError("--bopd-mode=fixed requires --entropy-coef=0 so entropy is not computed.")
        if args.bopd_fixed_advantage is None or args.bopd_fixed_advantage == 0:
            raise ValueError("--bopd-mode=fixed requires a non-zero --bopd-fixed-advantage.")
        if args.use_rollout_entropy:
            raise ValueError("--bopd-mode=fixed does not compute entropy; remove --use-rollout-entropy.")
        args.use_rollout_entropy = False
    else:
        raise ValueError(f"Unsupported baseline BOPD mode: {args.bopd_mode!r}.")


@dataclass(frozen=True)
class BOPDOutput:
    """Per-sample BOPD tensors plus batch-level diagnostic metrics."""

    advantages: list[torch.Tensor]
    loss_masks: list[torch.Tensor]
    metrics: dict[str, torch.Tensor]


def _validate_inputs(
    log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    entropies: list[torch.Tensor] | None,
) -> None:
    if len(log_probs) != len(loss_masks):
        raise ValueError(
            f"BOPD expected one loss mask per log-prob tensor, got "
            f"{len(loss_masks)} masks for {len(log_probs)} log-prob tensors."
        )
    if entropies is not None and len(entropies) != len(log_probs):
        raise ValueError(
            f"BOPD expected one entropy tensor per log-prob tensor, got "
            f"{len(entropies)} entropy tensors for {len(log_probs)} log-prob tensors."
        )

    for index, (log_prob, loss_mask) in enumerate(zip(log_probs, loss_masks, strict=True)):
        if log_prob.shape != loss_mask.shape:
            raise ValueError(
                f"BOPD log-prob/loss-mask shape mismatch for sample {index}: "
                f"{tuple(log_prob.shape)} vs {tuple(loss_mask.shape)}."
            )
        if entropies is not None and entropies[index].shape != log_prob.shape:
            raise ValueError(
                f"BOPD entropy/log-prob shape mismatch for sample {index}: "
                f"{tuple(entropies[index].shape)} vs {tuple(log_prob.shape)}."
            )


def compute_bopd(
    log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    *,
    token_fraction: float,
    mode: str,
    entropies: list[torch.Tensor] | None = None,
    advantage_min: float = -1.0,
    advantage_max: float = -0.5,
    fixed_advantage: float | None = None,
) -> BOPDOutput:
    """Compute baseline BOPD advantages for one DP-local packed response batch.

    All valid response tokens are ranked together by current actor log-probability.
    The lowest ``floor(token_fraction * N)`` are selected (at least one when
    ``N > 0``). Unselected tokens receive zero advantage and a zero BOPD mask.
    """
    if not 0 < token_fraction <= 1:
        raise ValueError(f"BOPD token_fraction must be in (0, 1], got {token_fraction}.")
    if mode not in {"entropy", "fixed"}:
        raise ValueError(f"Baseline BOPD mode must be 'entropy' or 'fixed', got {mode!r}.")
    if mode == "entropy":
        if entropies is None:
            raise ValueError("BOPD entropy mode requires actor entropies.")
        if not advantage_min < advantage_max < 0:
            raise ValueError(
                "BOPD entropy advantages must satisfy advantage_min < advantage_max < 0, "
                f"got {advantage_min} and {advantage_max}."
            )
    elif fixed_advantage is None or fixed_advantage == 0:
        raise ValueError("BOPD fixed mode requires a non-zero fixed_advantage.")

    _validate_inputs(log_probs, loss_masks, entropies)

    if not log_probs:
        empty = torch.tensor(0.0)
        return BOPDOutput(
            advantages=[],
            loss_masks=[],
            metrics={
                "bopd/selected_fraction": empty,
                "bopd/selected_tokens": empty,
                "bopd/valid_tokens": empty,
                "bopd/advantage_mean": empty,
            },
        )

    device = log_probs[0].device
    flat_log_probs = torch.cat([x.detach().to(device=device, dtype=torch.float32) for x in log_probs])
    flat_valid_mask = torch.cat([x.to(device=device).bool() for x in loss_masks])
    flat_advantages = torch.zeros_like(flat_log_probs, dtype=torch.float32)
    flat_bopd_mask = torch.zeros_like(flat_log_probs, dtype=torch.float32)

    valid_indices = torch.nonzero(flat_valid_mask, as_tuple=False).flatten()
    valid_count = valid_indices.numel()
    if valid_count > 0:
        selected_count = min(max(1, math.floor(token_fraction * valid_count)), valid_count)
        order = torch.argsort(flat_log_probs[valid_indices], stable=True)
        selected_indices = valid_indices[order[:selected_count]]
        flat_bopd_mask[selected_indices] = 1.0

        if mode == "fixed":
            flat_advantages[selected_indices] = fixed_advantage
        else:
            flat_entropies = torch.cat(
                [x.detach().to(device=device, dtype=torch.float32) for x in entropies]
            )
            selected_entropies = flat_entropies[selected_indices]
            entropy_range = selected_entropies.max() - selected_entropies.min()
            if entropy_range <= 1e-12:
                entropy_rank = torch.ones_like(selected_entropies)
            else:
                entropy_rank = (selected_entropies - selected_entropies.min()) / entropy_range
            flat_advantages[selected_indices] = (
                advantage_max + (advantage_min - advantage_max) * entropy_rank
            )

    split_sizes = [x.numel() for x in log_probs]
    advantages = [
        x.reshape_as(log_prob).to(dtype=log_prob.dtype)
        for x, log_prob in zip(flat_advantages.split(split_sizes), log_probs, strict=True)
    ]
    bopd_masks = [
        x.reshape_as(loss_mask)
        for x, loss_mask in zip(flat_bopd_mask.split(split_sizes), loss_masks, strict=True)
    ]

    selected_tokens = flat_bopd_mask.sum()
    valid_tokens = flat_valid_mask.to(dtype=torch.float32).sum()
    selected_advantages = flat_advantages[flat_bopd_mask.bool()]
    advantage_mean = (
        selected_advantages.mean()
        if selected_advantages.numel() > 0
        else torch.zeros((), dtype=torch.float32, device=device)
    )
    metrics = {
        "bopd/selected_fraction": selected_tokens / torch.clamp_min(valid_tokens, 1),
        "bopd/selected_tokens": selected_tokens,
        "bopd/valid_tokens": valid_tokens,
        "bopd/advantage_mean": advantage_mean,
    }
    return BOPDOutput(advantages=advantages, loss_masks=bopd_masks, metrics=metrics)
