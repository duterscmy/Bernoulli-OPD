import math

import pytest
import torch

from slime.backends.megatron_utils.opd import (
    _balanced_bernoulli_weights,
    compute_opd_selection,
    compute_opd_token_loss,
    compute_opd_update_diagnostics,
)


def _sampled_logit_gradient(loss_type: str, *, alpha: float = 0.5, bernoulli_clip: float = 0.0):
    logits = torch.tensor([0.2, -0.4, 1.1], dtype=torch.float64, requires_grad=True)
    current_log_prob = torch.log_softmax(logits, dim=0)[1:2]
    student_log_prob = current_log_prob.detach().clone()
    teacher_log_prob = torch.tensor([math.log(0.35)], dtype=torch.float64)
    loss = compute_opd_token_loss(
        current_log_prob,
        student_log_prob,
        teacher_log_prob,
        loss_type=loss_type,
        coef=0.7,
        geometry_alpha=alpha,
        power_alpha=alpha,
        sure_alpha=alpha,
        bernoulli_clip=bernoulli_clip,
    ).sum()
    loss.backward()
    return logits.grad[1], student_log_prob.exp().item(), teacher_log_prob.exp().item(), (
        teacher_log_prob - student_log_prob
    ).item()


@pytest.mark.parametrize(
    ("loss_type", "alpha"),
    [
        ("vanilla", 0.5),
        ("geometry_corrected", 0.5),
        ("bernoulli", 0.5),
        ("hybrid", 0.5),
        ("balanced_bernoulli", 0.5),
        ("weighted", 0.4),
        ("power_log", 0.5),
        ("power_log", 2.0),
        ("power_opd", 0.5),
        ("power_opd", 1.0),
        ("sure", 1.0),
    ],
)
def test_opd_sampled_logit_gradients_match_analytic_forms(loss_type, alpha):
    gradient, student_prob, teacher_prob, advantage = _sampled_logit_gradient(loss_type, alpha=alpha)
    if loss_type == "vanilla":
        expected = -0.7 * advantage * (1.0 - student_prob)
    elif loss_type == "geometry_corrected":
        expected = -0.7 * advantage
    elif loss_type in {"bernoulli", "balanced_bernoulli"}:
        expected = 0.7 * (student_prob - teacher_prob)
    elif loss_type == "hybrid":
        expected = (
            0.7 * (student_prob - teacher_prob)
            if student_prob > 0.5
            else -0.7 * advantage * (1.0 - student_prob)
        )
    elif loss_type == "power_log":
        expected = -0.7 * (teacher_prob**alpha - student_prob**alpha) / alpha
    elif loss_type == "power_opd":
        expected = -0.7 * (teacher_prob**alpha - student_prob**alpha) * (1.0 - student_prob)
    elif loss_type == "sure":
        expected = -0.7 * advantage * (1.0 - student_prob) * (1.0 + alpha * (1.0 - student_prob))
    else:
        expected = -0.7 * advantage * (1.0 - student_prob) ** (1.0 - alpha)
    assert gradient.item() == pytest.approx(expected, rel=2e-5, abs=2e-6)


@pytest.mark.parametrize(
    ("student_prob", "teacher_prob", "expected_branch"),
    [
        (0.4, 0.7, "vanilla"),
        (0.5, 0.7, "vanilla"),
        (0.8, 0.6, "bernoulli"),
    ],
)
def test_hybrid_uses_strict_threshold_and_matches_branch_gradient(
    student_prob, teacher_prob, expected_branch
):
    logits = torch.tensor(
        [math.log(student_prob), math.log(1.0 - student_prob)],
        dtype=torch.float64,
        requires_grad=True,
    )
    current = torch.log_softmax(logits, dim=0)[0:1]
    student = current.detach().clone()
    teacher = torch.tensor([math.log(teacher_prob)], dtype=torch.float64)
    loss = compute_opd_token_loss(
        current,
        student,
        teacher,
        loss_type="hybrid",
        coef=0.7,
        geometry_alpha=0.5,
        hybrid_threshold=0.5,
    ).sum()
    loss.backward()

    advantage = math.log(teacher_prob) - math.log(student_prob)
    if expected_branch == "bernoulli":
        expected = 0.7 * (student_prob - teacher_prob)
    else:
        expected = -0.7 * advantage * (1.0 - student_prob)
    assert logits.grad[0].item() == pytest.approx(expected, rel=2e-5, abs=2e-6)


def test_clipped_bernoulli_caps_sampled_logit_gradient_without_changing_direction():
    gradient, student_prob, teacher_prob, _ = _sampled_logit_gradient(
        "bernoulli", bernoulli_clip=0.1
    )
    expected = 0.7 * max(-0.1, min(0.1, student_prob - teacher_prob))
    assert gradient.item() == pytest.approx(expected, rel=2e-5, abs=2e-6)


def test_balanced_bernoulli_partially_equalizes_bin_gradient_mass_and_preserves_scale():
    student_prob = torch.tensor([0.05] * 4 + [0.95] * 4)
    teacher_prob = torch.tensor([0.25] * 4 + [0.94] * 4)
    valid = torch.ones(8, dtype=torch.bool)
    weights = _balanced_bernoulli_weights(
        student_prob,
        teacher_prob,
        valid,
        alpha=0.5,
        num_bins=10,
        max_weight=5.0,
    )

    assert weights.mean().item() == pytest.approx(1.0, abs=2e-6)
    assert weights.min().item() >= 0.2 - 1e-6
    assert weights.max().item() <= 5.0 + 1e-6
    assert weights[0].item() < weights[-1].item()

    gap = (student_prob - teacher_prob).abs()
    raw_mass_ratio = gap[:4].sum() / gap[4:].sum()
    balanced_mass_ratio = (gap[:4] * weights[:4]).sum() / (gap[4:] * weights[4:]).sum()
    assert balanced_mass_ratio < raw_mass_ratio
    assert balanced_mass_ratio > 1.0


def test_balanced_bernoulli_alpha_zero_exactly_matches_bernoulli():
    current = torch.log(torch.tensor([0.2, 0.6, 0.9], requires_grad=True))
    student = current.detach().clone()
    teacher = torch.log(torch.tensor([0.4, 0.5, 0.7]))
    valid = torch.ones(3)
    balanced = compute_opd_token_loss(
        current,
        student,
        teacher,
        loss_type="balanced_bernoulli",
        coef=0.7,
        geometry_alpha=0.5,
        balance_alpha=0.0,
        valid_mask=valid,
    )
    bernoulli = compute_opd_token_loss(
        current,
        student,
        teacher,
        loss_type="bernoulli",
        coef=0.7,
        geometry_alpha=0.5,
    )
    torch.testing.assert_close(balanced, bernoulli)


def test_sure_alpha_zero_matches_vanilla_gradient_at_on_policy_update_point():
    vanilla_grad, *_ = _sampled_logit_gradient("vanilla")
    sure_grad, *_ = _sampled_logit_gradient("sure", alpha=0.0)
    assert sure_grad.item() == pytest.approx(vanilla_grad.item(), rel=2e-5, abs=2e-6)


def test_sure_uses_exact_detached_weighted_k2_loss_without_weight_normalization():
    current = torch.tensor([math.log(0.6)], dtype=torch.float64, requires_grad=True)
    student = current.detach().clone()
    teacher = torch.tensor([math.log(0.8)], dtype=torch.float64)
    alpha = 1.0
    actual = compute_opd_token_loss(
        current,
        student,
        teacher,
        loss_type="sure",
        coef=0.7,
        geometry_alpha=0.5,
        sure_alpha=alpha,
    )
    expected_weight = 1.0 + alpha * (1.0 - student.exp())
    expected = (0.7 * 0.5 * expected_weight * (teacher - current).square()).to(actual.dtype)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("power_alpha", "endpoint_loss_type"),
    [(0.0, "geometry_corrected"), (1.0, "bernoulli")],
)
def test_power_log_endpoints_exactly_reuse_existing_objectives(power_alpha, endpoint_loss_type):
    current = torch.tensor([math.log(0.65)], dtype=torch.float64, requires_grad=True)
    student = torch.tensor([math.log(0.6)], dtype=torch.float64)
    teacher = torch.tensor([math.log(0.8)], dtype=torch.float64)
    power_loss = compute_opd_token_loss(
        current,
        student,
        teacher,
        loss_type="power_log",
        coef=0.7,
        geometry_alpha=0.5,
        power_alpha=power_alpha,
    )
    endpoint_loss = compute_opd_token_loss(
        current,
        student,
        teacher,
        loss_type=endpoint_loss_type,
        coef=0.7,
        geometry_alpha=0.5,
    )
    torch.testing.assert_close(power_loss, endpoint_loss)


def test_power_opd_alpha_one_retains_score_function_geometry_and_differs_from_bernoulli():
    power_grad, student_prob, teacher_prob, _ = _sampled_logit_gradient("power_opd", alpha=1.0)
    bernoulli_grad, *_ = _sampled_logit_gradient("bernoulli", alpha=1.0)
    expected = 0.7 * (student_prob - teacher_prob) * (1.0 - student_prob)
    assert power_grad.item() == pytest.approx(expected, rel=2e-5, abs=2e-6)
    assert power_grad.abs().item() < bernoulli_grad.abs().item()


def test_power_opd_alpha_500_remains_finite_and_matches_analytic_gradient():
    student_prob = 0.99
    teacher_prob = 0.995
    logits = torch.tensor(
        [math.log(student_prob), math.log(1.0 - student_prob)],
        dtype=torch.float64,
        requires_grad=True,
    )
    current = torch.log_softmax(logits, dim=0)[0:1]
    student = current.detach().clone()
    teacher = torch.tensor([math.log(teacher_prob)], dtype=torch.float64)
    loss = compute_opd_token_loss(
        current,
        student,
        teacher,
        loss_type="power_opd",
        coef=0.7,
        geometry_alpha=0.5,
        power_alpha=500.0,
    ).sum()
    loss.backward()

    expected = -0.7 * (teacher_prob**500 - student_prob**500) * (1.0 - student_prob)
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0].item() == pytest.approx(expected, rel=2e-5, abs=2e-8)


def test_high_conf_filter_is_strict_and_mask_stays_float():
    student = [torch.log(torch.tensor([0.2, 0.5, 0.5001, 0.9]))]
    teacher = [torch.log(torch.tensor([0.3, 0.4, 0.6, 0.8]))]
    masks = [torch.ones(4, dtype=torch.int64)]
    out = compute_opd_selection(
        student,
        teacher,
        masks,
        token_filter="high_conf",
        high_conf_threshold=0.5,
        bottom_fraction=0.2,
        geometry_alpha=0.5,
    )
    assert out.loss_masks[0].is_floating_point()
    assert out.loss_masks[0].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert out.metrics["opd/selected_fraction"].item() == pytest.approx(0.5)


def test_bottom_percent_ranks_valid_tokens_across_samples():
    student = [torch.tensor([-0.1, -4.0, -0.2]), torch.tensor([-3.0, -2.0])]
    teacher = [torch.zeros(3), torch.zeros(2)]
    masks = [torch.tensor([1, 0, 1]), torch.ones(2, dtype=torch.int64)]
    out = compute_opd_selection(
        student,
        teacher,
        masks,
        token_filter="bottom_percent",
        high_conf_threshold=0.5,
        bottom_fraction=0.5,
        geometry_alpha=0.5,
    )
    # Four valid tokens -> floor(0.5 * 4) = 2; the masked -4 token is ignored.
    assert out.loss_masks[0].tolist() == [0.0, 0.0, 0.0]
    assert out.loss_masks[1].tolist() == [1.0, 1.0]


def test_top_percent_ranks_valid_tokens_across_samples():
    student = [torch.tensor([-0.1, -4.0, -0.2]), torch.tensor([-3.0, -2.0])]
    teacher = [torch.zeros(3), torch.zeros(2)]
    masks = [torch.tensor([1, 0, 1]), torch.ones(2, dtype=torch.int64)]
    out = compute_opd_selection(
        student,
        teacher,
        masks,
        token_filter="top_percent",
        high_conf_threshold=0.5,
        bottom_fraction=0.2,
        top_fraction=0.5,
        geometry_alpha=0.5,
    )
    # Four valid tokens -> floor(0.5 * 4) = 2; select the two largest logp values.
    assert out.loss_masks[0].tolist() == [1.0, 0.0, 1.0]
    assert out.loss_masks[1].tolist() == [0.0, 0.0]
    assert out.metrics["opd/selected_fraction"].item() == pytest.approx(0.5)


def test_objectives_remain_finite_near_probability_one():
    current = torch.tensor([-1e-8], requires_grad=True)
    student = current.detach().clone()
    teacher = torch.tensor([math.log(0.8)])
    losses = [
        compute_opd_token_loss(
            current,
            student,
            teacher,
            loss_type=loss_type,
            coef=1.0,
            geometry_alpha=0.5,
        )
        for loss_type in (
            "geometry_corrected",
            "bernoulli",
            "hybrid",
            "balanced_bernoulli",
            "weighted",
            "power_log",
            "power_opd",
            "sure",
        )
    ]
    assert all(torch.isfinite(loss).all() for loss in losses)


def _mean(diagnostics, key):
    total, count = diagnostics[key]
    return (total / count.clamp_min(1)).item()


def test_update_diagnostics_measure_teacher_alignment_and_selected_gradient():
    pre_prob = torch.tensor([0.8, 0.7, 0.4, 0.2])
    teacher_prob = torch.tensor([0.9, 0.5, 0.6, 0.1])
    post_prob = torch.tensor([0.85, 0.6, 0.45, 0.25])
    valid = torch.ones(4)
    selected = torch.tensor([1.0, 1.0, 0.0, 0.0])
    diagnostics = compute_opd_update_diagnostics(
        [pre_prob.log()],
        [post_prob.log()],
        [teacher_prob.log()],
        [valid],
        [selected],
        loss_type="vanilla",
        geometry_alpha=0.5,
    )

    assert _mean(diagnostics, "selected/direction_agreement_fraction") == pytest.approx(1.0)
    assert _mean(diagnostics, "selected/toward_teacher_fraction") == pytest.approx(1.0)
    assert _mean(diagnostics, "all/direction_agreement_fraction") == pytest.approx(0.75)
    assert _mean(diagnostics, "all/applied_sampled_logit_grad_abs_mean") == pytest.approx(
        (
            abs(math.log(0.9) - math.log(0.8)) * (1 - 0.8)
            + abs(math.log(0.5) - math.log(0.7)) * (1 - 0.7)
        )
        / 4,
        rel=2e-5,
    )
    assert _mean(diagnostics, "all/teacher_log_gap_reduction_mean") > 0


@pytest.mark.parametrize(
    "loss_type",
    [
        "vanilla",
        "geometry_corrected",
        "bernoulli",
        "hybrid",
        "balanced_bernoulli",
        "weighted",
        "power_log",
        "power_opd",
        "sure",
    ],
)
def test_update_diagnostic_gradient_matches_objective_formula(loss_type):
    student_prob = torch.tensor([0.75])
    teacher_prob = torch.tensor([0.5])
    diagnostics = compute_opd_update_diagnostics(
        [student_prob.log()],
        [torch.tensor([0.7]).log()],
        [teacher_prob.log()],
        [torch.ones(1)],
        [torch.ones(1)],
        loss_type=loss_type,
        geometry_alpha=0.5,
    )
    measured = _mean(diagnostics, "selected/counterfactual_sampled_logit_grad_abs_mean")
    advantage = abs(math.log(0.5) - math.log(0.75))
    if loss_type == "vanilla":
        expected = advantage * (1 - 0.75)
    elif loss_type == "geometry_corrected":
        expected = advantage
    elif loss_type in {"bernoulli", "balanced_bernoulli", "hybrid"}:
        expected = 0.25
    elif loss_type == "power_log":
        expected = 2 * (math.sqrt(0.75) - math.sqrt(0.5))
    elif loss_type == "power_opd":
        expected = (math.sqrt(0.75) - math.sqrt(0.5)) * (1 - 0.75)
    elif loss_type == "sure":
        expected = advantage * (1 - 0.75) * (1 + 1.0 * (1 - 0.75))
    else:
        expected = advantage * math.sqrt(1 - 0.75)
    assert measured == pytest.approx(expected, rel=2e-5)
