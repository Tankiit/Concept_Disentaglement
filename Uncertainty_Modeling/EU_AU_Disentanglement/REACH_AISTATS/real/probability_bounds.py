"""Probly probability-bound representations with explicit admissible-set semantics.

Envelope: singleton-event extrema over the convex hull of finite MC predictions.
Source mixture: sharp event ratios over an observation-compatible joint-mass set.
Neither is an interval of mutual information. Quantification uses max class width.
"""
from dataclasses import dataclass
import numpy as np
import torch
from probly.representation import Representation
from probly.representer import Representer
from probly.quantification import quantify
from .source_acquisition import polytope, ratio_interval


def evaluate_probability_ood(id_scores,ood_scores):
    """Report tie-correct OOD AUROC plus Probly's unmodified result for audit.

    Installed Probly 0.9.0 does not group equal scores in its NumPy ROC path.
    We do not modify that installation or silently present its output as corrected.
    """
    from probly.evaluation.ood import evaluate_ood
    from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve
    a,b=np.asarray(id_scores),np.asarray(ood_scores)
    if a.ndim!=1 or b.ndim!=1 or not len(a) or not len(b) or not np.isfinite(np.r_[a,b]).all():
        raise ValueError('OOD evaluation needs nonempty finite 1-D scores; report rejected rows separately')
    y=np.r_[np.zeros(len(a)),np.ones(len(b))]; score=np.r_[a,b]
    raw=evaluate_ood(a,b)['auroc']; auc=roc_auc_score(y,score)
    fpr,tpr,_=roc_curve(y,score)
    return dict(auroc=float(auc),probly_raw_auroc=float(raw),probly_auroc_difference=float(raw-auc),
        aupr_ood=float(average_precision_score(y,score)),
        dappr_aupr_id=float(average_precision_score(1-y,-score)),
        fpr95=float(fpr[np.flatnonzero(tpr>=.95)[0]]))


@dataclass(frozen=True)
class ProbabilityBounds(Representation):
    lower: torch.Tensor  # N,K; separate event extrema, not a distribution
    upper: torch.Tensor
    nominal: torch.Tensor  # reference prediction, not necessarily in source set
    semantics: str
    samples: torch.Tensor | None = None
    cells: torch.Tensor | None = None
    candidate_lower: torch.Tensor | None = None
    candidate_upper: torch.Tensor | None = None
    status: tuple | None = None

    def __post_init__(self):
        if self.lower.ndim != 2 or self.lower.shape != self.upper.shape:
            raise ValueError('Expected matching N,K lower/upper probability arrays')
        if self.nominal.shape != self.lower.shape:
            raise ValueError('Nominal predictions must have shape N,K')
        good = torch.isfinite(self.lower) & torch.isfinite(self.upper)
        if torch.any(self.lower[good] < -1e-7) or torch.any(self.upper[good] > 1+1e-7):
            raise ValueError('Probability endpoints must be in [0,1]')
        if torch.any(self.lower[good] > self.upper[good]+1e-7):
            raise ValueError('Lower endpoint exceeds upper endpoint')

    @property
    def width(self):
        return self.upper-self.lower

    @property
    def valid(self):
        return torch.isfinite(self.lower).all(-1) & torch.isfinite(self.upper).all(-1)


@dataclass(frozen=True)
class ProbabilityImprecision:
    """Custom Probly quantification. No aleatoric/total decomposition is asserted."""
    epistemic: torch.Tensor  # max coordinate width; chosen scalar summary
    class_widths: torch.Tensor
    measure: str = 'maximum class-probability width (not mutual information)'


@quantify.register(ProbabilityBounds)
def quantify_probability_bounds(rep, *args, **kwargs):
    if args or kwargs:
        raise TypeError('This custom quantifier accepts no extra options')
    return ProbabilityImprecision(rep.width.max(-1).values,rep.width)


def envelope_from_samples(probabilities):
    p=torch.as_tensor(probabilities).detach()
    if p.ndim!=3 or p.shape[1]<1 or p.shape[-1]<2:
        raise ValueError('Expected N,T,K sampled probabilities')
    if not torch.isfinite(p).all() or torch.any(p<0) or not torch.allclose(p.sum(-1),torch.ones_like(p[...,0]),atol=1e-6):
        raise ValueError('Samples must be finite normalized categorical probabilities')
    return ProbabilityBounds(p.min(1).values,p.max(1).values,p.mean(1),
        'Convex hull of finite MC predictions; no population coverage guarantee',samples=p)


class DropoutEnvelopeRepresenter(Representer):
    """Wrap an actual Probly sampler's represent() result, without changing Probly."""
    def __init__(self, base_representer):
        super().__init__(base_representer)

    def represent(self,*args,**kwargs):
        with torch.no_grad():
            sample=self.predictor.represent(*args,**kwargs)
        # Probly 0.9.0 stores distributions inside a TorchSample. Read the
        # probability property explicitly; applying softmax again would be wrong.
        p=sample.tensor.probabilities
        p=p.movedim(sample.sample_dim,1)
        return envelope_from_samples(p)

    @staticmethod
    def event_bounds(rep,classes):
        """Exact hull bounds for a union of classes, not sums of coordinate extrema."""
        ids=list(classes)
        if len(ids)!=len(set(ids)) or any(k<0 or k>=rep.lower.shape[1] for k in ids):
            raise ValueError('Event classes must be distinct valid class indices')
        values=rep.samples[:,:,ids].sum(-1)
        return values.min(1).values,values.max(1).values


class SourceMixtureBoundsRepresenter(Representer):
    """Use a fixed image->logit predictor and a shared deployment compatibility set.

    Last source is the candidate. lower/upper are K known-class membership
    probabilities conditional on known membership. Candidate probability is
    returned separately. Source-cell laws must use this same deterministic map.
    """
    def __init__(self,predictor,row_lower,row_upper,marginal,weights=None,cell_fn=None):
        super().__init__(predictor)
        lo,hi=np.asarray(row_lower),np.asarray(row_upper)
        if lo.ndim!=2 or lo.shape!=hi.shape or lo.shape[0]<3:
            raise ValueError('Expected K known sources plus one candidate, by D cells')
        if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo<0) or np.any(hi>1) or np.any(lo>hi):
            raise ValueError('Invalid source-law boxes')
        self.K,self.D=lo.shape[0]-1,lo.shape[1]
        self.cell_fn=cell_fn
        self.poly=polytope(lo,hi,marginal,weights)
        self.cache={}

    def _event_at_cell(self,cell,classes,conditional_known):
        shape=(self.K+1,self.D)
        numerator=np.zeros(shape); numerator[list(classes),cell]=1
        denominator=np.zeros(shape)
        denominator[:self.K if conditional_known else self.K+1,cell]=1
        return ratio_interval(self.poly,numerator,denominator)

    def _at_cell(self,cell):
        if cell not in self.cache:
            known=[self._event_at_cell(cell,[k],True) for k in range(self.K)]
            candidate=self._event_at_cell(cell,[self.K],False)
            self.cache[cell]=(known,candidate)
        return self.cache[cell]

    def represent(self,x):
        with torch.no_grad():
            logits=self.predictor(x)
            nominal=logits.softmax(-1).detach()
        if nominal.ndim!=2 or nominal.shape[1]!=self.K:
            raise ValueError('Expected K known-class logits')
        if self.cell_fn is None and self.D!=self.K:
            raise ValueError('Provide cell_fn when D differs from K')
        cells=nominal.argmax(-1) if self.cell_fn is None else self.cell_fn(logits)
        return self.represent_cells(cells,nominal)

    def represent_cells(self,cells,nominal=None):
        cells=torch.as_tensor(cells,dtype=torch.long).reshape(-1)
        if torch.any(cells<0) or torch.any(cells>=self.D):
            raise ValueError('Invalid cell index')
        if nominal is None:
            nominal=torch.full((len(cells),self.K),1/self.K,dtype=torch.float64)
        lower=[];upper=[];cl=[];cu=[];status=[]
        for cell in cells.cpu().tolist():
            known,candidate=self._at_cell(cell)
            lower.append([a['lower'] for a in known]);upper.append([a['upper'] for a in known])
            cl.append(candidate['lower']);cu.append(candidate['upper'])
            status.append(tuple(a['status'] for a in known)+(candidate['status'],))
        tensor=lambda a:torch.as_tensor(a,dtype=torch.float64,device=nominal.device)
        return ProbabilityBounds(tensor(lower).reshape(-1,self.K),tensor(upper).reshape(-1,self.K),nominal,
            'Observation-compatible source mixture; known-class probabilities conditional on known membership',
            cells=cells,candidate_lower=tensor(cl),candidate_upper=tensor(cu),status=tuple(status))

    def event_bounds(self,cells,classes,conditional_known=True):
        ids=list(classes)
        size=self.K if conditional_known else self.K+1
        if len(ids)!=len(set(ids)) or any(k<0 or k>=size for k in ids):
            raise ValueError('Invalid event class indices')
        out=[self._event_at_cell(int(c),ids,conditional_known) for c in cells]
        return np.array([a['lower'] for a in out]),np.array([a['upper'] for a in out])
