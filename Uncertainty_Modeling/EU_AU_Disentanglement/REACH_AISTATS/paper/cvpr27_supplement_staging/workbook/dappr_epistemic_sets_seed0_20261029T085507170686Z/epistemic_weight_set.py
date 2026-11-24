"""Probly representations of MI ranges over a declared set of dropout weights.

For probabilities p_t and rho in [0, 1], W_rho = (1-rho)/T + rho*simplex(T).
The image of W_rho under weighted mutual information is a closed interval.
Its minimum is attained at a vertex; its maximum is concave optimization.
We return a feasible maximum and a Frank-Wolfe tangent upper bound separately.
These bounds condition on the sampled predictions, not all possible models.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from probly.representation import Representation
from probly.representer import Representer
from probly.representation.distribution.torch_categorical import (
    TorchProbabilityCategoricalDistribution,
    TorchCategoricalDistributionSample,
)
from probly.quantification import decompose, entropy


def categorical_entropy(p):
    """Use Probly's entropy primitive with natural logarithms."""
    return entropy(TorchProbabilityCategoricalDistribution(p))


def weighted_eu(probabilities, weights):
    """Probly entropy-based weighted MI; (..., T, K), (..., T) -> (...,).

    Explicit weighting is intentional: the installed Probly 0.9.0 categorical
    sample MI implementation averages uniformly even when sample weights exist.
    """
    mean = (weights.unsqueeze(-1)*probabilities).sum(-2)
    return categorical_entropy(mean) - (weights*categorical_entropy(probabilities)).sum(-1)


@dataclass(frozen=True)
class EpistemicInterval(Representation):
    """Per-input interval plus witnesses and numerical endpoint uncertainty."""
    probabilities: torch.Tensor  # (N, T, K), retained for sensitivity analyses
    rho: float
    lower: torch.Tensor
    upper: torch.Tensor  # numerical tangent outer bound, not just feasible max
    upper_attained: torch.Tensor
    lower_weights: torch.Tensor
    upper_weights: torch.Tensor
    nominal_eu: torch.Tensor
    nominal_tu: torch.Tensor
    nominal_au: torch.Tensor
    iterations: int

    @property
    def midpoint(self):
        return (self.lower + self.upper)/2

    @property
    def width(self):
        return self.upper-self.lower

    @property
    def optimization_gap(self):
        return (self.upper-self.upper_attained).clamp_min(0)

    @property
    def mean_probabilities(self):
        return self.probabilities.mean(1)


def bound_epistemic(probabilities, rho=.5, tolerance=1e-6, max_iter=200):
    """Bound weighted MI for every row using double-precision arithmetic.

    Returns an exact vertex minimum (up to floating point), a feasible maximum,
    and an outer upper endpoint. The latter remains valid by concavity even if
    max_iter is reached, apart from floating-point error. Inspect the gap before
    interpreting differences between endpoints or scalar scores.
    """
    p = torch.as_tensor(probabilities).detach().to(dtype=torch.float64)
    if p.ndim != 3 or min(p.shape) < 1 or p.shape[-1] < 2:
        raise ValueError('Expected nonempty (N, T, K) probabilities with K >= 2')
    if not torch.isfinite(p).all() or (p < 0).any():
        raise ValueError('Probabilities must be finite and nonnegative')
    if not torch.allclose(p.sum(-1), torch.ones_like(p[..., 0]), atol=1e-6, rtol=1e-6):
        raise ValueError('Every categorical probability vector must sum to one')
    if not 0 <= rho <= 1 or tolerance <= 0 or max_iter < 1:
        raise ValueError('Need 0 <= rho <= 1, tolerance > 0 and max_iter >= 1')
    # Normalize rounding error; exact zeros are retained.
    p = p/p.sum(-1, keepdim=True)
    n, t, k = p.shape
    rows = torch.arange(n, device=p.device)
    uniform = p.new_full((n, t), 1/t)
    h = categorical_entropy(p)
    mean_p, mean_h = p.mean(1), h.mean(1)
    sample = TorchCategoricalDistributionSample(
        tensor=TorchProbabilityCategoricalDistribution(p), sample_dim=1)
    decomposition = decompose(sample)
    nominal_eu = decomposition.epistemic
    nominal_tu, nominal_au = decomposition.total, decomposition.aleatoric
    assert torch.allclose(nominal_eu, weighted_eu(p, uniform), atol=1e-10, rtol=1e-8)

    # All T extreme weight vectors, evaluated for each input at once.
    vertex_means = (1-rho)*mean_p[:, None, :] + rho*p
    vertex_eu = categorical_entropy(vertex_means) - ((1-rho)*mean_h[:, None] + rho*h)
    lower, minimizing_vertex = vertex_eu.min(1)
    lower = lower.clamp_min(0)
    lower_weights = (1-rho)*uniform
    lower_weights[rows, minimizing_vertex] += rho

    q = uniform.clone()
    best_value = nominal_eu.clone()
    best_weights = uniform.clone()
    upper = p.new_full((n,), math.log(k))
    if rho == 0:
        return EpistemicInterval(p, rho, nominal_eu, nominal_eu, nominal_eu,
                                 uniform, uniform, nominal_eu, nominal_tu, nominal_au, 0)

    for iteration in range(1, max_iter+1):
        weights = (1-rho)*uniform + rho*q
        mixture_q = (q.unsqueeze(-1)*p).sum(1)
        mixture = (1-rho)*mean_p + rho*mixture_q
        value = categorical_entropy(mixture) - (weights*h).sum(1)
        improved = value > best_value
        best_value = torch.maximum(best_value, value)
        best_weights[improved] = weights[improved]
        # dEU/dq_t = rho*(KL(p_t || mixture)-1). Constant -1 cancels.
        kl = -(p*mixture.clamp_min(torch.finfo(p.dtype).tiny).log()[:, None, :]).sum(-1)-h
        chosen = kl.argmax(1)
        gap = (rho*(kl[rows, chosen] - (q*kl).sum(1))).clamp_min(0)
        upper = torch.minimum(upper, value+gap)
        # Small roundoff can make an otherwise valid upper bound < best witness.
        upper = torch.maximum(upper, best_value)
        active = upper-best_value > tolerance
        if not active.any() or iteration == max_iter:
            break
        # Away steps remove weak active vertices and avoid slow boundary convergence.
        support_kl = torch.where(q > 1e-14, kl, torch.full_like(kl, torch.inf))
        away_vertex = support_kl.argmin(1)
        away_gain = rho*((q*kl).sum(1)-kl[rows, away_vertex])
        away_mass = q[rows, away_vertex]
        use_away = (away_gain > gap) & (away_mass < 1-1e-14)
        direction_p = torch.where(use_away[:, None], mixture_q-p[rows, away_vertex], p[rows, chosen]-mixture_q)
        q_h = (q*h).sum(1)
        direction_h = torch.where(use_away, q_h-h[rows, away_vertex], h[rows, chosen]-q_h)
        gamma_max = torch.where(use_away, away_mass/(1-away_mass).clamp_min(1e-14), torch.ones_like(away_mass))
        # Exact 1-D concave line search by bisection, including boundary optima.
        lo, hi = p.new_zeros(n), gamma_max.clone()
        for _ in range(32):
            gamma = (lo+hi)/2
            moved = mixture + rho*gamma[:, None]*direction_p
            derivative = rho*(-(direction_p*moved.clamp_min(torch.finfo(p.dtype).tiny).log()).sum(1)-direction_h)
            lo = torch.where(derivative > 0, gamma, lo)
            hi = torch.where(derivative > 0, hi, gamma)
        endpoint = mixture+rho*gamma_max[:, None]*direction_p
        endpoint_derivative = rho*(-(direction_p*endpoint.clamp_min(torch.finfo(p.dtype).tiny).log()).sum(1)-direction_h)
        gamma = torch.where(endpoint_derivative >= 0, gamma_max, (lo+hi)/2)
        gamma = torch.where(active, gamma, torch.zeros_like(gamma))
        sign = torch.where(use_away, -torch.ones_like(gamma), torch.ones_like(gamma))
        q *= 1-(sign*gamma)[:, None]
        q[rows, torch.where(use_away, away_vertex, chosen)] += sign*gamma
        q.clamp_min_(0)
        q /= q.sum(1, keepdim=True)

    assert torch.all(lower <= nominal_eu+1e-9)
    assert torch.all(nominal_eu <= upper+1e-9)
    assert torch.allclose(weighted_eu(p, lower_weights), lower, atol=1e-9)
    assert torch.allclose(weighted_eu(p, best_weights), best_value, atol=1e-9)
    return EpistemicInterval(p, rho, lower, upper, best_value, lower_weights,
                             best_weights, nominal_eu, nominal_tu, nominal_au, iteration)


class DropoutEpistemicRepresenter(Representer):
    """Wrap an nn.Module returning logits; __call__/predict return an interval.

    Dropout modules alone are activated, including Dropout2d/3d. Other module
    states and torch RNG state are restored. Bounds depend on rho and the finite
    dropout sample. No ID/OOD labels are supplied to this wrapper.
    """
    def __init__(self, predictor, passes=30, rho=.5, seed=0, tolerance=1e-6, max_iter=200):
        super().__init__(predictor)
        if passes < 2:
            raise ValueError('Use at least two MC passes')
        self.passes, self.rho, self.seed = passes, rho, seed
        self.tolerance, self.max_iter = tolerance, max_iter

    def represent(self, x, *, seed=None):
        model = self.predictor
        modes = {m: m.training for m in model.modules()}
        devices = [x.device.index if x.device.index is not None else torch.cuda.current_device()] if x.is_cuda else []
        try:
            model.eval()
            for module in model.modules():
                if isinstance(module, nn.modules.dropout._DropoutNd):
                    module.train()
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                local_seed = self.seed if seed is None else seed
                torch.random.default_generator.manual_seed(local_seed)
                if x.is_cuda:
                    with torch.cuda.device(x.device):
                        torch.cuda.manual_seed(local_seed)
                p = torch.stack([model(x).softmax(-1) for _ in range(self.passes)], dim=1)
        finally:
            for module, mode in modes.items():
                module.training = mode
        return bound_epistemic(p, rho=self.rho, tolerance=self.tolerance, max_iter=self.max_iter)
