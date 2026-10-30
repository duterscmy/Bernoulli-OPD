"""Token selection, objectives, and diagnostics for on-policy distillation."""

import math
from dataclasses import dataclass

import torch


_LOSS_TYPES = {
    "vanilla",
    "geometry_corrected",
    "bernoulli",
    "hybrid",
    "balanced_bernoulli",
    "weighted",
    "power_log",
    "power_opd",
    "sure",
}
_TOKEN_FILTERS = {"all", "high_conf", "bottom_percent", "top_percent"}


def uses_direct_opd_objective(args) -> bool:
    """Whether OPD must be evaluated separately from the legacy advantage path."""
    return args.use_opd and (args.opd_loss_type != "vanilla" or args.opd_token_filter != "all")


def validate_opd_args(args) -> None:
    """Validate the parameterized OPD objective without changing legacy defaults."""
    if args.opd_loss_type not in _LOSS_TYPES:
        raise ValueError(f"Unsupported --opd-loss-type: {args.opd_loss_type!r}.")
    if args.opd_token_filter not in _TOKEN_FILTERS:
        raise ValueError(f"Unsupported --opd-token-filter: {args.opd_token_filter!r}.")
    if not 0 < args.opd_high_conf_threshold < 1:
        raise ValueError("--opd-high-conf-threshold must be in (0, 1).")
    if not 0 < args.opd_hybrid_threshold < 1:
        raise ValueError("--opd-hybrid-threshold must be in (0, 1).")
    if not 0 < args.opd_bottom_fraction <= 1:
        raise ValueError("--opd-bottom-fraction must be in (0, 1].")
    if not 0 < args.opd_top_fraction <= 1:
        raise ValueError("--opd-top-fraction must be in (0, 1].")
    if not 0 <= args.opd_geometry_alpha <= 1:
        raise ValueError("--opd-geometry-alpha must be in [0, 1].")
    if not math.isfinite(args.opd_power_alpha) or args.opd_power_alpha < 0:
        raise ValueError("--opd-power-alpha must be finite and non-negative.")
    if args.opd_loss_type == "power_opd" and args.opd_power_alpha <= 0:
        raise ValueError("--opd-power-alpha must be positive with --opd-loss-type=power_opd.")
    if not math.isfinite(args.opd_sure_alpha) or args.opd_sure_alpha < 0:
        raise ValueError("--opd-sure-alpha must be finite and non-negative.")
    if not 0 <= args.opd_bernoulli_clip <= 1:
        raise ValueError("--opd-bernoulli-clip must be in [0, 1].")
    if args.opd_bernoulli_clip and args.opd_loss_type != "bernoulli":
        raise ValueError("--opd-bernoulli-clip is only valid with --opd-loss-type=bernoulli.")
    if not 0 <= args.opd_balance_alpha <= 1:
        raise ValueError("--opd-balance-alpha must be in [0, 1].")
    if args.opd_balance_bins < 2:
        raise ValueError("--opd-balance-bins must be at least 2.")
    if not math.isfinite(args.opd_balance_max_weight) or args.opd_balance_max_weight < 1:
        raise ValueError("--opd-balance-max-weight must be finite and at least 1.")
    if args.opd_diagnostics_interval < 0:
        raise ValueError("--opd-diagnostics-interval must be non-negative.")

    if not args.use_opd:
        if (
            args.opd_loss_type != "vanilla"
            or args.opd_token_filter != "all"
            or args.opd_filter_normalization != "selected"
            or args.opd_diagnostics_interval != 0
        ):
            raise ValueError("Non-default OPD objectives, filters, and diagnostics require --use-opd.")
        return

    if args.opd_kl_coef < 0:
        raise ValueError("--opd-kl-coef must be non-negative.")
    if args.opd_diagnostics_interval and args.use_routing_replay:
        raise ValueError("OPD post-update diagnostics do not currently support routing replay.")

    if uses_direct_opd_objective(args):
        if args.normalize_advantages:
            raise ValueError("Direct OPD objectives are incompatible with --normalize-advantages.")
        incompatible = {
            "--use-rollout-logprobs": args.use_rollout_logprobs,
            "--use-opsm": args.use_opsm,
            "--use-tis": args.use_tis,
            "--get-mismatch-metrics": args.get_mismatch_metrics,
            "--use-score-centering": getattr(args, "use_score_centering", False),
        }
        enabled = [name for name, value in incompatible.items() if value]
        if enabled:
            raise ValueError(
                "Direct OPD objectives currently require an on-policy, unmodified policy loss; "
                f"disable {', '.join(enabled)}."
            )
        if getattr(args, "custom_pg_loss_reducer_function_path", None) is not None:
            raise ValueError("Direct OPD objectives do not support a custom policy-loss reducer.")


@dataclass(frozen=True)
class OPDSelection:
    """Full-response selection masks and batch-level diagnostics."""

    loss_masks: list[torch.Tensor]
    metrics: dict[str, torch.Tensor]


def _balanced_bernoulli_weights(
    student_prob: torch.Tensor,
    teacher_prob: torch.Tensor,
    valid_mask: torch.Tensor | None,
    *,
    alpha: float,
    num_bins: int,
    max_weight: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return detached probability-bin weights with valid-token mean one.

    Let ``M_b = sum_{t in b} |pS_t - pT_t|`` be the ordinary Bernoulli
    sampled-logit gradient mass in probability bin ``b``.  Before bounding,
    each token in that bin receives ``(mean_b M_b / M_b) ** alpha``.  Thus
    ``alpha=0`` exactly recovers Bernoulli OPD, while ``alpha=1`` attempts to
    equalize total gradient mass across occupied bins.  The configured run
    uses the partial correction ``alpha=0.5``.

    The weights are normalized over valid response tokens.  A uniform shrink
    toward one enforces the symmetric ``[1/max_weight, max_weight]`` bound
    without changing their mean, avoiding a learning-rate change or extreme
    amplification from a sparsely populated bin.
    """
    if student_prob.shape != teacher_prob.shape:
        raise ValueError("Balanced Bernoulli requires aligned student and teacher probabilities.")
    if valid_mask is None:
        valid = torch.ones_like(student_prob, dtype=torch.bool)
    else:
        if valid_mask.shape != student_prob.shape:
            raise ValueError("Balanced Bernoulli valid_mask must match the probability tensors.")
        valid = valid_mask.to(device=student_prob.device, dtype=torch.bool)

    with torch.no_grad():
        weights = torch.ones_like(student_prob, dtype=torch.float32)
        if alpha == 0 or max_weight == 1 or not torch.any(valid):
            return weights

        p_s = student_prob.detach().to(torch.float32).clamp(0.0, 1.0)
        p_t = teacher_prob.detach().to(device=p_s.device, dtype=torch.float32).clamp(0.0, 1.0)
        gap = (p_s - p_t).abs()
        bin_index = torch.clamp((p_s * num_bins).to(torch.long), max=num_bins - 1)
        bin_mass = torch.zeros(num_bins, device=p_s.device, dtype=torch.float32)
        bin_mass.scatter_add_(0, bin_index[valid], gap[valid])
        active = bin_mass > eps
        if not torch.any(active):
            return weights

        mean_mass = bin_mass[active].mean()
        bin_weight = torch.ones_like(bin_mass)
        bin_weight[active] = (mean_mass / bin_mass[active]).pow(alpha)
        token_weight = bin_weight[bin_index]
        token_mean = token_weight[valid].mean().clamp_min(eps)
        normalized = token_weight / token_mean

        # w = 1 + shrink * (normalized - 1) retains E_valid[w] = 1.
        # Pick the largest shrink in [0, 1] satisfying both hard bounds.
        lower = 1.0 / max_weight
        upper = max_weight
        valid_normalized = normalized[valid]
        max_normalized = valid_normalized.max()
        min_normalized = valid_normalized.min()
        upper_shrink = torch.where(
            max_normalized > 1.0,
            (upper - 1.0) / (max_normalized - 1.0).clamp_min(eps),
            torch.ones_like(max_normalized),
        )
        lower_shrink = torch.where(
            min_normalized < 1.0,
            (1.0 - lower) / (1.0 - min_normalized).clamp_min(eps),
            torch.ones_like(min_normalized),
        )
        shrink = torch.minimum(torch.ones_like(upper_shrink), torch.minimum(upper_shrink, lower_shrink))
        balanced = 1.0 + shrink * (normalized - 1.0)
        weights[valid] = balanced[valid]
        return weights


def _sampled_logit_grad_abs(
    student_prob: torch.Tensor,
    teacher_prob: torch.Tensor,
    advantage: torch.Tensor,
    *,
    loss_type: str,
    geometry_alpha: float,
    power_alpha: float = 0.5,
    sure_alpha: float = 1.0,
    bernoulli_clip: float = 0.0,
    hybrid_threshold: float = 0.5,
    balance_alpha: float = 0.5,
    balance_bins: int = 10,
    balance_max_weight: float = 5.0,
    balance_valid_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Exact unreduced objective-gradient magnitude for the sampled token logit."""
    one_minus_p = (1.0 - student_prob).clamp_min(eps)
    if loss_type == "vanilla":
        return advantage.abs() * one_minus_p
    if loss_type == "geometry_corrected":
        return advantage.abs()
    if loss_type == "bernoulli":
        magnitude = (teacher_prob - student_prob).abs()
        return magnitude.clamp_max(bernoulli_clip) if bernoulli_clip else magnitude
    if loss_type == "hybrid":
        vanilla_magnitude = advantage.abs() * one_minus_p
        bernoulli_magnitude = (teacher_prob - student_prob).abs()
        return torch.where(student_prob > hybrid_threshold, bernoulli_magnitude, vanilla_magnitude)
    if loss_type == "balanced_bernoulli":
        magnitude = (teacher_prob - student_prob).abs()
        weights = _balanced_bernoulli_weights(
            student_prob,
            teacher_prob,
            balance_valid_mask,
            alpha=balance_alpha,
            num_bins=balance_bins,
            max_weight=balance_max_weight,
            eps=eps,
        )
        return magnitude * weights
    if loss_type == "weighted":
        return advantage.abs() * one_minus_p.pow(1.0 - geometry_alpha)
    if loss_type == "power_log":
        if power_alpha == 0:
            return advantage.abs()
        signal = (teacher_prob.pow(power_alpha) - student_prob.pow(power_alpha)) / power_alpha
        return signal.abs()
    if loss_type == "power_opd":
        signal = teacher_prob.pow(power_alpha) - student_prob.pow(power_alpha)
        return signal.abs() * one_minus_p
    if loss_type == "sure":
        # SuRe applies this detached, bounded multiplier to the vanilla K2
        # gradient without renormalizing the token-mean denominator.
        sure_weight = 1.0 + sure_alpha * one_minus_p
        return advantage.abs() * one_minus_p * sure_weight
    raise ValueError(f"Unsupported OPD loss type: {loss_type!r}.")


def _validate_inputs(
    student_log_probs: list[torch.Tensor],
    teacher_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> None:
    if not (len(student_log_probs) == len(teacher_log_probs) == len(loss_masks)):
        raise ValueError("OPD requires one student tensor, teacher tensor, and loss mask per sample.")
    for index, (student, teacher, mask) in enumerate(
        zip(student_log_probs, teacher_log_probs, loss_masks, strict=True)
    ):
        if student.shape != teacher.shape or student.shape != mask.shape:
            raise ValueError(
                f"OPD tensor shape mismatch for sample {index}: student={tuple(student.shape)}, "
                f"teacher={tuple(teacher.shape)}, mask={tuple(mask.shape)}."
            )


def compute_opd_selection(
    student_log_probs: list[torch.Tensor],
    teacher_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    *,
    token_filter: str,
    high_conf_threshold: float,
    bottom_fraction: float,
    geometry_alpha: float,
    top_fraction: float = 0.2,
    power_alpha: float = 0.5,
    sure_alpha: float = 1.0,
    bernoulli_clip: float = 0.0,
    hybrid_threshold: float = 0.5,
) -> OPDSelection:
    """Build a float selection mask and token-level diagnostics for one DP-local batch."""
    if token_filter not in _TOKEN_FILTERS:
        raise ValueError(f"Unsupported OPD token filter: {token_filter!r}.")
    _validate_inputs(student_log_probs, teacher_log_probs, loss_masks)

    if not student_log_probs:
        zero = torch.tensor(0.0)
        return OPDSelection([], {"opd/selected_fraction": zero, "opd/valid_tokens": zero})

    device = student_log_probs[0].device
    student_lp = torch.cat([x.detach().to(device=device, dtype=torch.float32) for x in student_log_probs])
    teacher_lp = torch.cat([x.detach().to(device=device, dtype=torch.float32) for x in teacher_log_probs])
    valid = torch.cat([x.to(device=device).bool() for x in loss_masks])
    selected = torch.zeros_like(student_lp, dtype=torch.float32)
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()

    if token_filter == "all":
        selected[valid_indices] = 1.0
    elif token_filter == "high_conf":
        keep = student_lp[valid_indices].exp() > high_conf_threshold
        selected[valid_indices[keep]] = 1.0
    elif valid_indices.numel() > 0:
        fraction = top_fraction if token_filter == "top_percent" else bottom_fraction
        selected_count = min(max(1, math.floor(fraction * valid_indices.numel())), valid_indices.numel())
        order = torch.argsort(student_lp[valid_indices], stable=True)
        chosen = order[-selected_count:] if token_filter == "top_percent" else order[:selected_count]
        selected[valid_indices[chosen]] = 1.0

    split_sizes = [x.numel() for x in student_log_probs]
    masks = [
        chunk.reshape_as(mask)
        for chunk, mask in zip(selected.split(split_sizes), loss_masks, strict=True)
    ]

    valid_float = valid.to(torch.float32)
    selected_bool = selected.bool()
    valid_count = valid_float.sum()
    selected_count = selected.sum()
    student_prob = student_lp.exp().clamp(0.0, 1.0)
    teacher_prob = teacher_lp.exp().clamp(0.0, 1.0)
    advantage = teacher_lp - student_lp
    prob_gap = teacher_prob - student_prob

    def valid_mean(value: torch.Tensor) -> torch.Tensor:
        return (value * valid_float).sum() / torch.clamp_min(valid_count, 1)

    def selected_mean(value: torch.Tensor) -> torch.Tensor:
        return (value * selected).sum() / torch.clamp_min(selected_count, 1)

    vanilla_grad = _sampled_logit_grad_abs(
        student_prob,
        teacher_prob,
        advantage,
        loss_type="vanilla",
        geometry_alpha=geometry_alpha,
    )
    corrected_grad = _sampled_logit_grad_abs(
        student_prob,
        teacher_prob,
        advantage,
        loss_type="geometry_corrected",
        geometry_alpha=geometry_alpha,
    )
    bernoulli_grad = _sampled_logit_grad_abs(
        student_prob,
        teacher_prob,
        advantage,
        loss_type="bernoulli",
        geometry_alpha=geometry_alpha,
        bernoulli_clip=bernoulli_clip,
    )
    hybrid_grad = _sampled_logit_grad_abs(
        student_prob,
        teacher_prob,
        advantage,
        loss_type="hybrid",
        geometry_alpha=geometry_alpha,
        hybrid_threshold=hybrid_threshold,
    )
    weighted_grad = _sampled_logit_grad_abs(
        student_prob,
        teacher_prob,
        advantage,
        loss_type="weighted",
        geometry_alpha=geometry_alpha,
    )
    power_log_grad = _sampled_logit_grad_abs(
        student_prob,
        teacher_prob,
        advantage,
        loss_type="power_log",
        geometry_alpha=geometry_alpha,
        power_alpha=power_alpha,
    )
    power_opd_grad = _sampled_logit_grad_abs(
        student_prob,
        teacher_prob,
        advantage,
        loss_type="power_opd",
        geometry_alpha=geometry_alpha,
        power_alpha=power_alpha,
    )
    sure_grad = _sampled_logit_grad_abs(
        student_prob,
        teacher_prob,
        advantage,
        loss_type="sure",
        geometry_alpha=geometry_alpha,
        sure_alpha=sure_alpha,
    )
    metrics = {
        "opd/selected_fraction": selected_count / torch.clamp_min(valid_count, 1),
        "opd/selected_tokens": selected_count,
        "opd/valid_tokens": valid_count,
        "opd/student_prob_mean": valid_mean(student_prob),
        "opd/teacher_prob_mean": valid_mean(teacher_prob),
        "opd/advantage_mean": valid_mean(advantage),
        "opd/advantage_abs_mean": valid_mean(advantage.abs()),
        "opd/advantage_positive_fraction": valid_mean((advantage > 0).to(torch.float32)),
        "opd/advantage_negative_fraction": valid_mean((advantage < 0).to(torch.float32)),
        "opd/prob_gap_mean": valid_mean(prob_gap),
        "opd/prob_gap_abs_mean": valid_mean(prob_gap.abs()),
        "opd/selected_student_prob_mean": selected_mean(student_prob),
        "opd/selected_teacher_prob_mean": selected_mean(teacher_prob),
        "opd/selected_advantage_mean": selected_mean(advantage),
        "opd/selected_advantage_abs_mean": selected_mean(advantage.abs()),
        "opd/selected_prob_gap_abs_mean": selected_mean(prob_gap.abs()),
        "opd/proxy_vanilla_abs_mean": valid_mean(vanilla_grad),
        "opd/proxy_corrected_abs_mean": valid_mean(corrected_grad),
        "opd/proxy_bernoulli_abs_mean": valid_mean(bernoulli_grad),
        "opd/proxy_hybrid_abs_mean": valid_mean(hybrid_grad),
        "opd/proxy_weighted_abs_mean": valid_mean(weighted_grad),
        "opd/proxy_power_log_abs_mean": valid_mean(power_log_grad),
        "opd/proxy_power_opd_abs_mean": valid_mean(power_opd_grad),
        "opd/proxy_sure_abs_mean": valid_mean(sure_grad),
        "opd/selected_proxy_vanilla_abs_mean": selected_mean(vanilla_grad),
        "opd/selected_proxy_corrected_abs_mean": selected_mean(corrected_grad),
        "opd/selected_proxy_bernoulli_abs_mean": selected_mean(bernoulli_grad),
        "opd/selected_proxy_hybrid_abs_mean": selected_mean(hybrid_grad),
        "opd/selected_proxy_weighted_abs_mean": selected_mean(weighted_grad),
        "opd/selected_proxy_power_log_abs_mean": selected_mean(power_log_grad),
        "opd/selected_proxy_power_opd_abs_mean": selected_mean(power_opd_grad),
        "opd/selected_proxy_sure_abs_mean": selected_mean(sure_grad),
    }
    for index in range(10):
        lower = index / 10
        upper = (index + 1) / 10
        in_bin = valid & (student_prob >= lower)
        in_bin &= student_prob <= upper if index == 9 else student_prob < upper
        bin_count = in_bin.to(torch.float32).sum()
        selected_in_bin = (in_bin & selected_bool).to(torch.float32).sum()
        metrics[f"opd/pS_bin_{index}_fraction"] = bin_count / torch.clamp_min(valid_count, 1)
        metrics[f"opd/pS_bin_{index}_selected_fraction"] = selected_in_bin / torch.clamp_min(bin_count, 1)
        bin_float = in_bin.to(torch.float32)
        bin_denom = torch.clamp_min(bin_count, 1)
        metrics[f"opd/pS_bin_{index}_teacher_prob_mean"] = (teacher_prob * bin_float).sum() / bin_denom
        metrics[f"opd/pS_bin_{index}_advantage_abs_mean"] = (advantage.abs() * bin_float).sum() / bin_denom
        metrics[f"opd/pS_bin_{index}_prob_gap_abs_mean"] = (prob_gap.abs() * bin_float).sum() / bin_denom
        metrics[f"opd/pS_bin_{index}_proxy_vanilla_abs_mean"] = (vanilla_grad * bin_float).sum() / bin_denom
        metrics[f"opd/pS_bin_{index}_proxy_corrected_abs_mean"] = (corrected_grad * bin_float).sum() / bin_denom
        metrics[f"opd/pS_bin_{index}_proxy_bernoulli_abs_mean"] = (bernoulli_grad * bin_float).sum() / bin_denom
        metrics[f"opd/pS_bin_{index}_proxy_hybrid_abs_mean"] = (hybrid_grad * bin_float).sum() / bin_denom
        metrics[f"opd/pS_bin_{index}_proxy_weighted_abs_mean"] = (weighted_grad * bin_float).sum() / bin_denom
        metrics[f"opd/pS_bin_{index}_proxy_power_log_abs_mean"] = (power_log_grad * bin_float).sum() / bin_denom
        metrics[f"opd/pS_bin_{index}_proxy_power_opd_abs_mean"] = (power_opd_grad * bin_float).sum() / bin_denom
        metrics[f"opd/pS_bin_{index}_proxy_sure_abs_mean"] = (sure_grad * bin_float).sum() / bin_denom

    return OPDSelection(loss_masks=masks, metrics={key: value.detach() for key, value in metrics.items()})


def compute_opd_update_diagnostics(
    pre_student_log_probs: list[torch.Tensor],
    post_student_log_probs: list[torch.Tensor],
    teacher_log_probs: list[torch.Tensor],
    valid_masks: list[torch.Tensor],
    selected_masks: list[torch.Tensor],
    *,
    loss_type: str,
    geometry_alpha: float,
    power_alpha: float = 0.5,
    sure_alpha: float = 1.0,
    bernoulli_clip: float = 0.0,
    hybrid_threshold: float = 0.5,
    balance_alpha: float = 0.5,
    balance_bins: int = 10,
    balance_max_weight: float = 5.0,
    eps: float = 1e-8,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Compare one batch before and after an optimizer update.

    Values are returned as ``(sum, count)`` pairs so callers can obtain exact
    token-weighted means across DP/CP ranks. The reported sampled-logit
    gradient is the analytic gradient of the unreduced token objective, not a
    decomposition of the full parameter-gradient norm.
    """
    if not (
        len(pre_student_log_probs)
        == len(post_student_log_probs)
        == len(teacher_log_probs)
        == len(valid_masks)
        == len(selected_masks)
    ):
        raise ValueError("OPD update diagnostics require one aligned tensor of each kind per sample.")
    if not pre_student_log_probs:
        return {}

    for index, tensors in enumerate(
        zip(
            pre_student_log_probs,
            post_student_log_probs,
            teacher_log_probs,
            valid_masks,
            selected_masks,
            strict=True,
        )
    ):
        shapes = {tuple(tensor.shape) for tensor in tensors}
        if len(shapes) != 1:
            raise ValueError(f"OPD update diagnostic shape mismatch for sample {index}: {sorted(shapes)}.")

    device = pre_student_log_probs[0].device
    pre_lp = torch.cat([value.detach().to(device=device, dtype=torch.float32) for value in pre_student_log_probs])
    post_lp = torch.cat([value.detach().to(device=device, dtype=torch.float32) for value in post_student_log_probs])
    teacher_lp = torch.cat([value.detach().to(device=device, dtype=torch.float32) for value in teacher_log_probs])
    valid = torch.cat([value.to(device=device).bool() for value in valid_masks])
    selected = torch.cat([value.to(device=device).bool() for value in selected_masks]) & valid

    pre_prob = pre_lp.exp().clamp(0.0, 1.0)
    post_prob = post_lp.exp().clamp(0.0, 1.0)
    teacher_prob = teacher_lp.exp().clamp(0.0, 1.0)
    advantage = teacher_lp - pre_lp
    post_advantage = teacher_lp - post_lp
    delta_logp = post_lp - pre_lp
    delta_prob = post_prob - pre_prob
    log_gap_pre = advantage.abs()
    log_gap_post = post_advantage.abs()
    prob_gap_pre = (teacher_prob - pre_prob).abs()
    prob_gap_post = (teacher_prob - post_prob).abs()
    counterfactual_grad = _sampled_logit_grad_abs(
        pre_prob,
        teacher_prob,
        advantage,
        loss_type=loss_type,
        geometry_alpha=geometry_alpha,
        power_alpha=power_alpha,
        sure_alpha=sure_alpha,
        bernoulli_clip=bernoulli_clip,
        hybrid_threshold=hybrid_threshold,
        balance_alpha=balance_alpha,
        balance_bins=balance_bins,
        balance_max_weight=balance_max_weight,
        balance_valid_mask=selected,
    )
    applied_grad = counterfactual_grad * selected.to(torch.float32)
    desired_progress = advantage.sign() * delta_logp
    evaluable_direction = (advantage.abs() > eps) & (delta_logp.abs() > eps)
    direction_agrees = (advantage * delta_logp) > 0
    toward_teacher = log_gap_post < (log_gap_pre - eps)
    away_from_teacher = log_gap_post > (log_gap_pre + eps)
    overshot_teacher = (advantage * post_advantage) < 0

    metrics: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def add_mean(name: str, value: torch.Tensor, mask: torch.Tensor) -> None:
        mask_float = mask.to(torch.float32)
        metrics[name] = ((value.to(torch.float32) * mask_float).sum(), mask_float.sum())

    def add_scope(prefix: str, mask: torch.Tensor) -> None:
        add_mean(f"{prefix}/student_prob_pre_mean", pre_prob, mask)
        add_mean(f"{prefix}/teacher_prob_mean", teacher_prob, mask)
        add_mean(f"{prefix}/advantage_mean", advantage, mask)
        add_mean(f"{prefix}/advantage_abs_mean", advantage.abs(), mask)
        add_mean(f"{prefix}/advantage_positive_fraction", (advantage > 0).to(torch.float32), mask)
        add_mean(f"{prefix}/advantage_negative_fraction", (advantage < 0).to(torch.float32), mask)
        add_mean(f"{prefix}/counterfactual_sampled_logit_grad_abs_mean", counterfactual_grad, mask)
        add_mean(f"{prefix}/applied_sampled_logit_grad_abs_mean", applied_grad, mask)
        add_mean(f"{prefix}/delta_logp_mean", delta_logp, mask)
        add_mean(f"{prefix}/delta_logp_abs_mean", delta_logp.abs(), mask)
        add_mean(f"{prefix}/delta_prob_mean", delta_prob, mask)
        add_mean(f"{prefix}/delta_prob_abs_mean", delta_prob.abs(), mask)
        add_mean(f"{prefix}/directional_progress_mean", desired_progress, mask)
        add_mean(f"{prefix}/teacher_log_gap_pre_abs_mean", log_gap_pre, mask)
        add_mean(f"{prefix}/teacher_log_gap_post_abs_mean", log_gap_post, mask)
        add_mean(f"{prefix}/teacher_log_gap_reduction_mean", log_gap_pre - log_gap_post, mask)
        add_mean(f"{prefix}/teacher_prob_gap_pre_abs_mean", prob_gap_pre, mask)
        add_mean(f"{prefix}/teacher_prob_gap_post_abs_mean", prob_gap_post, mask)
        add_mean(f"{prefix}/teacher_prob_gap_reduction_mean", prob_gap_pre - prob_gap_post, mask)
        add_mean(f"{prefix}/nonzero_update_fraction", (delta_logp.abs() > eps).to(torch.float32), mask)
        add_mean(f"{prefix}/toward_teacher_fraction", toward_teacher.to(torch.float32), mask)
        add_mean(f"{prefix}/away_from_teacher_fraction", away_from_teacher.to(torch.float32), mask)
        add_mean(f"{prefix}/overshoot_fraction", overshot_teacher.to(torch.float32), mask)
        direction_mask = mask & evaluable_direction
        add_mean(f"{prefix}/direction_evaluable_fraction", evaluable_direction.to(torch.float32), mask)
        add_mean(f"{prefix}/direction_agreement_fraction", direction_agrees.to(torch.float32), direction_mask)

    valid_count = valid.to(torch.float32).sum()
    selected_count = selected.to(torch.float32).sum()
    one = torch.ones((), device=device, dtype=torch.float32)
    metrics["valid_tokens_per_dp_cp_rank"] = (valid_count, one)
    metrics["selected_tokens_per_dp_cp_rank"] = (selected_count, one)
    metrics["selected_fraction"] = (selected_count, valid_count)
    add_scope("all", valid)
    add_scope("selected", selected)

    for index in range(10):
        lower = index / 10
        upper = (index + 1) / 10
        in_bin = valid & (pre_prob >= lower)
        in_bin &= pre_prob <= upper if index == 9 else pre_prob < upper
        bin_count = in_bin.to(torch.float32).sum()
        metrics[f"pS_bin_{index}/token_fraction"] = (bin_count, valid_count)
        metrics[f"pS_bin_{index}/selected_fraction"] = (
            (in_bin & selected).to(torch.float32).sum(),
            bin_count,
        )
        add_mean(f"pS_bin_{index}/advantage_abs_mean", advantage.abs(), in_bin)
        add_mean(f"pS_bin_{index}/counterfactual_sampled_logit_grad_abs_mean", counterfactual_grad, in_bin)
        add_mean(f"pS_bin_{index}/applied_sampled_logit_grad_abs_mean", applied_grad, in_bin)
        add_mean(f"pS_bin_{index}/delta_logp_mean", delta_logp, in_bin)
        add_mean(f"pS_bin_{index}/delta_logp_abs_mean", delta_logp.abs(), in_bin)
        add_mean(f"pS_bin_{index}/teacher_log_gap_reduction_mean", log_gap_pre - log_gap_post, in_bin)
        add_mean(f"pS_bin_{index}/toward_teacher_fraction", toward_teacher.to(torch.float32), in_bin)
        direction_bin = in_bin & evaluable_direction
        add_mean(
            f"pS_bin_{index}/direction_agreement_fraction",
            direction_agrees.to(torch.float32),
            direction_bin,
        )

    return {
        key: (sum_value.detach(), count.detach())
        for key, (sum_value, count) in metrics.items()
    }


def compute_opd_token_loss(
    current_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    *,
    loss_type: str,
    coef: float,
    geometry_alpha: float,
    power_alpha: float = 0.5,
    sure_alpha: float = 1.0,
    bernoulli_clip: float = 0.0,
    hybrid_threshold: float = 0.5,
    balance_alpha: float = 0.5,
    balance_bins: int = 10,
    balance_max_weight: float = 5.0,
    valid_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return an unreduced direct OPD loss over sampled response tokens."""
    if loss_type not in _LOSS_TYPES:
        raise ValueError(f"Unsupported OPD loss type: {loss_type!r}.")

    current_lp = current_log_probs.to(torch.float32)
    student_lp = student_log_probs.detach().to(device=current_lp.device, dtype=torch.float32)
    teacher_lp = teacher_log_probs.detach().to(device=current_lp.device, dtype=torch.float32)
    advantage = teacher_lp - student_lp

    if loss_type == "vanilla":
        token_loss = -advantage * current_lp
    elif loss_type == "hybrid":
        # Use a detached on-policy student probability to choose the objective.
        # Equality belongs to the Vanilla branch; only pS > threshold uses
        # Bernoulli, matching the strict high-confidence definition.
        student_prob = student_lp.exp().clamp(min=0.0, max=1.0)
        current_prob = current_lp.exp().clamp(min=eps, max=1.0 - eps)
        teacher_prob = teacher_lp.exp().clamp(min=eps, max=1.0 - eps)
        vanilla_loss = -advantage * current_lp
        bernoulli_loss = -teacher_prob * current_lp - (1.0 - teacher_prob) * torch.log1p(-current_prob)
        token_loss = torch.where(student_prob > hybrid_threshold, bernoulli_loss, vanilla_loss)
    elif loss_type == "geometry_corrected":
        current_prob = current_lp.exp().clamp(min=eps, max=1.0 - eps)
        current_log_odds = current_lp - torch.log1p(-current_prob)
        token_loss = -advantage * current_log_odds
    elif loss_type == "weighted":
        student_prob = student_lp.exp().clamp(min=0.0, max=1.0 - eps)
        weight = (1.0 - student_prob).clamp_min(eps).pow(-geometry_alpha)
        token_loss = -weight * advantage * current_lp
    elif loss_type == "power_log":
        # The endpoints deliberately reuse the established objectives exactly:
        # alpha=0 is GC-OPD and alpha=1 is Bernoulli OPD. Between and beyond
        # them, the detached power-log teacher signal is applied through the
        # sampled-token log-odds surrogate, whose sampled-logit derivative is 1.
        current_prob = current_lp.exp().clamp(min=eps, max=1.0 - eps)
        if power_alpha == 0:
            current_log_odds = current_lp - torch.log1p(-current_prob)
            token_loss = -advantage * current_log_odds
        elif power_alpha == 1:
            teacher_prob = teacher_lp.exp().clamp(min=eps, max=1.0 - eps)
            token_loss = -teacher_prob * current_lp - (1.0 - teacher_prob) * torch.log1p(-current_prob)
        else:
            # expm1 is more accurate than subtracting p**alpha near alpha=0.
            signal = (torch.expm1(power_alpha * teacher_lp) - torch.expm1(power_alpha * student_lp))
            signal = signal / power_alpha
            current_log_odds = current_lp - torch.log1p(-current_prob)
            token_loss = -signal * current_log_odds
    elif loss_type == "power_opd":
        # PowerOPD (arXiv:2606.17199) replaces the unbounded log-ratio
        # reward with the detached bounded power reward pT**alpha-pS**alpha,
        # while retaining the ordinary sampled log-probability surrogate.
        # The paper absorbs the optional 1/alpha factor into the learning
        # rate, so it is intentionally omitted here.  At alpha=1 the
        # sampled-token logit gradient is (pS-pT) * (1-pS), which is distinct
        # from Bernoulli OPD's pS-pT gradient.
        student_prob = student_lp.exp().clamp(min=0.0, max=1.0)
        teacher_prob = teacher_lp.exp().clamp(min=0.0, max=1.0)
        power_reward = teacher_prob.pow(power_alpha) - student_prob.pow(power_alpha)
        token_loss = -power_reward * current_lp
    elif loss_type == "sure":
        # Surprise-aware Reweighting (SuRe), Eq. (16)/(18) of
        # arXiv:2608.25643.  The student probability used by the bounded
        # weight is detached and the surrounding reducer retains the ordinary
        # unweighted token-count denominator.  Using the K2 loss directly also
        # makes alpha=0 the paper's vanilla objective; at the on-policy update
        # point its gradient is identical to the detached log-ratio surrogate.
        student_prob = student_lp.exp().clamp(min=0.0, max=1.0)
        sure_weight = 1.0 + sure_alpha * (1.0 - student_prob)
        current_gap = teacher_lp - current_lp
        token_loss = 0.5 * sure_weight * current_gap.square()
    elif loss_type in {"bernoulli", "balanced_bernoulli"}:
        current_prob = current_lp.exp().clamp(min=eps, max=1.0 - eps)
        teacher_prob = teacher_lp.exp().clamp(min=eps, max=1.0 - eps)
        token_loss = -teacher_prob * current_lp - (1.0 - teacher_prob) * torch.log1p(-current_prob)
        if loss_type == "balanced_bernoulli":
            student_prob = student_lp.exp().clamp(min=eps, max=1.0 - eps)
            balance_weight = _balanced_bernoulli_weights(
                student_prob,
                teacher_prob,
                valid_mask,
                alpha=balance_alpha,
                num_bins=balance_bins,
                max_weight=balance_max_weight,
                eps=eps,
            )
            token_loss = balance_weight * token_loss
        elif bernoulli_clip:
            gap_abs = (current_prob - teacher_prob).abs().detach()
            clip_scale = torch.where(
                gap_abs > bernoulli_clip,
                bernoulli_clip / gap_abs.clamp_min(eps),
                torch.ones_like(gap_abs),
            )
            token_loss = clip_scale * token_loss
    else:
        raise AssertionError(f"Unhandled OPD loss type: {loss_type!r}.")

    return coef * token_loss
