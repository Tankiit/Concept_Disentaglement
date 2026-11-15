"""Build the narrated notebook. Run with Python containing nbformat."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
n=nbf.v4.new_notebook(); cells=[]
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))
md(r'''# Which information reduces epistemic uncertainty?
## A reproducible four-dataset finite-cell source-mixture example

**MNIST · EMNIST Letters · Fashion-MNIST · BloodMNIST**

**Question.** For a specified probability target, what ambiguity remains after observing inputs, and which additional observations resolve it?

**Reading order.** We first state the assumptions and sources; validate the inference on an exactly solvable example; construct image-derived source laws; compute their infinite-input identification limits; then acquire four kinds of information separately. Finally we inspect probability endpoints, repeated-run inclusion and limitations.

**What is being run.** This is a controlled, image-derived simulation. We reuse the cached predictions and class assignments of the existing four-dataset pilot as a frozen finite cell map. We do not retrain a network. Images supply the initial cell histograms; subsequent acquisitions are independent simulated categorical draws from those declared histograms. A million simulated draws are **not** a million newly collected images. The estimands are probabilities in this declared finite model, not guaranteed probabilities in the natural image population.

**Connection to DAPPr [D].** We adopt the notebook-to-workbook workflow and the principle of retaining alternatives compatible with evidence. We do not use DAPPr's loss, Dirichlet possibility functions, or scalar epistemic score. This experiment implements the source-mixture scope discussed with the user. It is also distinct from the earlier weighted-dropout MI interval.

Every code cell has an explanatory preceding cell and an opening goal comment. Run all cells in order. Each run creates a fresh `workbook/source_acquisition_<UTC timestamp>/` folder. No network access or model download is needed.''')
md('''### Sources and their exact roles

These are all external papers/theories used in this notebook. The mathematical derivations below are written out; a citation does not substitute for an assumption.

| Key | Source | What we use; what we do not claim |
|---|---|---|
| D | Ni et al., [Possibilistic Predictive Uncertainty for Deep Learning](https://arxiv.org/abs/2605.00600) | Workflow motivation only; no reproduction or comparison of its reported scores. |
| E | Hüllermeier & Waegeman (2021), [Aleatoric and epistemic uncertainty](https://doi.org/10.1007/s10994-021-05946-3) | Sets of possible distributions express incomplete knowledge. No entropy decomposition is estimated here. |
| L | Lipton et al. (2018), [Detecting and Correcting for Label Shift with Black Box Predictors](https://proceedings.mlr.press/v80/lipton18a.html) | Fixed source-conditionals and the mixture moment equation. Our set inference allows non-identification. |
| H | Hoeffding (1963), [Probability Inequalities for Sums of Bounded Random Variables](https://doi.org/10.1080/01621459.1963.10500830) ([accessible paper](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf)) | Fixed-sample coordinate concentration plus an explicitly counted union bound. Not an anytime confidence sequence. |
| C | Charnes & Cooper (1962), [Programming with Linear Fractional Functionals](https://doi.org/10.1002/nav.3800090303) | Transform each probability-ratio extremization into a linear program; the transformation is derived below. |
| M | LeCun et al., [MNIST dataset](https://yann.lecun.com/exdb/mnist/) | Cached digit images. Official site returned a gateway error during source checking; no new download or unseen content is relied on. |
| EL | Cohen et al. (2017), [EMNIST](https://arxiv.org/abs/1702.05373), [NIST dataset page](https://www.nist.gov/itl/products-and-services/emnist-dataset) | Cached EMNIST Letters images, transposed as in the prior pilot. |
| F | Xiao et al. (2017), [Fashion-MNIST](https://arxiv.org/abs/1708.07747) | Cached clothing images; the pilot repartitioned the official test pool. Not an official benchmark protocol. |
| B | Yang et al. (2023), [MedMNIST v2](https://doi.org/10.1038/s41597-022-01721-8) | Cached BloodMNIST images, converted to grayscale in the prior pilot. No clinical interpretation. |

Local foundations: `paper/sections/setup.tex` and `paper/sections/theory.tex` supply the source-mixture question and identification arguments. This notebook uses **coordinate boxes and joint-mass LPs**, not that draft's Euclidean MMD ball. It preserves reference uncertainty jointly and does not import an unproved contraction rate from the draft.

The paper fetches were read-only. The run exports this source list, the notebook, helper source, input hashes, and configuration for auditing.''')
code('''# Goal: locate the existing finite-image pilot and create an isolated artifact directory.
from pathlib import Path
import sys, json, hashlib, shutil, platform, importlib.metadata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
ROOT=Path.cwd()
if not (ROOT/'real/source_acquisition.py').exists():
    ROOT=ROOT/'Uncertainty_Modeling/EU_AU_Disentanglement/REACH_AISTATS'
assert (ROOT/'real/source_acquisition.py').exists()
sys.path.insert(0,str(ROOT))
from real.source_acquisition import (frequency_box, polytope, ratio_interval,
    membership_bounds, anchored_label_bounds, categorical_stream)
SOURCE=ROOT/'workbook/dappr_epistemic_sets_seed0_20261029T085507170686Z'
assert SOURCE.exists(), f'Required existing pilot missing: {SOURCE}'
DATASETS=['mnist','emnist','fashion','blood']
SEED=20261029
REPEATS=6
BUDGETS=[0,100,1000,10000,100000]
BASE=dict(inputs=500,references=200,target_labels=100,source_ids=0)
ALPHA=.05
ETA=.10  # specified symmetric target-label flip probability; never estimated
SMOOTHING=.02  # explicit uniform-cell random replacement probability
W_TRUE=np.array([.45,.25,.30])
THETA=np.array([.2,.8])  # P(clean binary target=1 | known source), specified
ANCHOR=np.array([.5,.5])  # fixed reference source prevalences, not deployment weights
RUN_DIR=ROOT/'workbook'/('source_acquisition_'+pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ'))
RUN_DIR.mkdir(parents=True,exist_ok=False)
plt.rcdefaults()
print('Artifacts:',RUN_DIR)
print('Source pilot:',SOURCE)
''')
md(r'''### 1. Specify the targets before calculating uncertainty

There are **three sources**: two groups of known image classes, (S=0,1), and a specified candidate-novel source (S=2). These source groups are separate from classifier-output cells (X=c). Let (A_{sc}=P(X=c\mid S=s)), (w_s=P(S=s)), and (p_c=\sum_s w_s A_{sc}).

We study, at one fixed probe cell per dataset:

1. **Known-source membership:** (T_K(c)=w_0A_{0c}/(w_0A_{0c}+w_1A_{1c})). This is membership in known group 0 **conditional on being from a known group**. It is not the probability that an input is ID.
2. **Candidate-novel membership:** (T_N(c)=w_2A_{2c}/p_c).
3. **Reference-anchored binary target:** (T_Y(c)=\sum_{s<2}a_s A_{sc}\theta_s/\sum_{s<2}a_s A_{sc}), with fixed anchor (a=(1/2,1/2)). This is not the deployment label probability, and novel-source labels are outside this target.

For each target, its lower/upper endpoints range over all compatible models. Their difference is epistemic imprecision within this model [E]. These three widths are **not an additive decomposition**. A point interval can describe a probability strictly between zero and one: knowing a probability does not determine every individual outcome.

**Acquisition assumptions:** sources have frozen cell laws [L]; draws are i.i.d.; reference sampling is stratified by reliable source identity; clean target (Y) is independent of the cell given source; observed target labels flip independently with known rate (eta<1/2); deployment source identities are sampled representatively and without error. The source-ID acquisition uses only identity counts, not paired cells. Thus it can learn weights but does not additionally improve reference rows. No aleatoric uncertainty is estimated.

**Novelty assumption:** candidate (q=A_2) is specified, not learned from the unlabeled deployment sample. Held-out class data are used by the experiment designer to define it. This is candidate-source inference, not arbitrary unknown-class detection.''')
md(r'''### 2. How finite information defines the compatible set

For (D) categorical frequencies estimated from (n) i.i.d. observations, use radius
[
r(n,D,\delta)=\sqrt{\log(2D/\delta)/(2n)}.
]
Hoeffding's bound for indicators and a union bound imply simultaneous coordinate inclusion with probability at least (1-\delta) [H]. At zero observations the box is ([0,1]^D).

For one run and budget, split (alpha=.05) into four acquisition channels. Split reference and label shares again between the two sources. This gives **at least 95% simultaneous inclusion of all three targets at that fixed setting**, conditional on the declared model and exact solver arithmetic. It does not give 95% simultaneous inclusion across all plotted budgets, datasets, or repeats. The six repeats provide a descriptive inclusion check, not a precise coverage estimate.

Let (z_{sc}=P(S=s,X=c)). Require nonnegativity, (\sum z=1), and
[
\underline A_{sc}\sum_j z_{sj}\le z_{sc}\le
\overline A_{sc}\sum_j z_{sj}.
]
Also constrain column sums by the deployment-input box and row sums by the source-identity box. The candidate row has exact lower=upper (q). This is a joint model set: we do not plug estimated rows into membership ratios and pretend they are exact.

Each membership target is (c^Tz/d^Tz). With (t=1/(d^Tz)), (y=tz), inequalities (Gz\le h) become (Gy\le ht), normalization becomes (\sum y=t), and (d^Ty=1). Minimizing/maximizing (c^Ty) is a linear program [C]. A zero-denominator compatible model triggers an explicit ([0,1]) policy; an empty set is logged as model rejection, not as zero uncertainty.

For noisy labels, (P(\widetilde Y=1\mid S=s)=\eta+(1-2\eta)\theta_s). Invert the observed-label interval and intersect with ([0,1]). The known anchor fixes source weights in a separate two-source LP; positive coefficients make the label endpoints occur at the corresponding (	heta) bounds.

**Transition:** before applying this machinery to images, check an example whose true endpoints are known exactly.''')
code('''# Goal: validate the solver against a complete analytic non-identification example.
A_toy=np.array([[1.,0.],[0.,1.],[.5,.5]])
p_toy=np.array([.5,.5])
toy=membership_bounds(A_toy,A_toy,(p_toy,p_toy),None,0)
assert np.allclose([toy['candidate_membership']['lower'],toy['candidate_membership']['upper']],[0,1])
known_w=np.array([.35,.35,.30])
identified=membership_bounds(A_toy,A_toy,(p_toy,p_toy),(known_w,known_w),0)
assert np.allclose([identified['candidate_membership']['lower'],identified['candidate_membership']['upper']],[.3,.3])
display(pd.DataFrame([dict(information='Exact input law',**toy['candidate_membership']),
                      dict(information='Exact inputs + source proportions',**identified['candidate_membership'])]))
print('For every t: w=((1-t)/2,(1-t)/2,t) gives p=(1/2,1/2), but novel membership=t.')
''')
md('''### 3. Construct the four image-derived finite worlds

The input pilot assigned ID classes before model fitting and trained separate small dropout CNNs on ID images only. It saved 20-pass prediction probabilities, original labels, source-image indices, and class roles. We freeze each saved mean-prediction argmax as the cell for that finite image. No dropout is resampled here. These cells are a map on the cached finite population; no out-of-sample mapping claim is needed.

Split the ordered ID classes into two approximately equal groups; keep every OOD class in the candidate group. Use equal weight per original class inside each group, so unequal class counts do not silently change the source definition. All assignments are exported. The remaining cells are the classifier's original ID outputs, not the three source labels.

The empirical source histogram is modified by a declared 2% uniform-cell replacement mechanism: (A=(1-0.02)A_{emp}+0.02/d). This makes probe probabilities positive and is part of the simulation model, not an estimated fact about the datasets.

Two regimes share the same known rows:

- **Image-derived candidate:** (q) comes from cached held-out class images. Separation is measured, not presumed.
- **Constructed confounding:** replace (q) by ((A_0+A_1)/2). This source is a controlled cell-level construction, not the actual held-out image law. Its purpose is to create exact observational equivalence.

Choose the probe with largest known-reference cell mass, before observing any simulated acquisition data. The same probe is used in both regimes. We condition on this fixed empirical world; uncertainty from estimating a natural-image population is not included.

**Transition:** the exported laws let us calculate what unlimited inputs can identify before we inspect finite samples.''')
code('''# Goal: derive and save exact simulation laws and trace every cached image to its role.
scores=pd.read_csv(SOURCE/'scores.csv')
roles=pd.read_csv(SOURCE/'class_roles.csv')
assignments=pd.read_csv(SOURCE/'assignments.csv')
worlds={}; source_rows=[]; law_rows=[]; image_rows=[]
for name in DATASETS:
    f=scores[scores.dataset==name].sort_values('test_position').copy()
    probabilities=np.load(SOURCE/f'{name}_dropout_predictions.npz')['probabilities']
    d=probabilities.shape[-1]
    assert len(f)==len(probabilities)
    assert np.array_equal(f.prediction.to_numpy(),probabilities.mean(1).argmax(1))
    ordered=roles[(roles.dataset==name)&(roles.role=='ID')].label.to_numpy()
    groups=[ordered[:len(ordered)//2],ordered[len(ordered)//2:],
            roles[(roles.dataset==name)&(roles.role=='OOD')].label.to_numpy()]
    A=[]
    for s,classes in enumerate(groups):
        hist=[]
        for label in classes:
            sub=f[f.original_label==label]
            assert len(sub)>0
            hist.append(np.bincount(sub.prediction,minlength=d)/len(sub))
            source_rows.append(dict(dataset=name,source=s,original_class=int(label),
                classifier_role='ID' if s<2 else 'OOD',within_source_class_weight=1/len(classes)))
            for _,r in sub.iterrows():
                image_rows.append(dict(dataset=name,source=s,original_class=int(label),
                    source_index=int(r.source_index),test_position=int(r.test_position),cell=int(r.prediction)))
        A.append((1-SMOOTHING)*np.mean(hist,axis=0)+SMOOTHING/d)
    A=np.array(A)
    probe=int((ANCHOR@A[:2]).argmax())
    for regime in ['image_derived','confounded']:
        rows=A.copy()
        if regime=='confounded': rows[2]=rows[:2].mean(0)
        p=W_TRUE@rows
        sigma=float(np.linalg.svd((rows[:2]-rows[2]).T,compute_uv=False)[-1])
        worlds[name,regime]=dict(A=rows,p=p,probe=probe,sigma=sigma)
        for s in range(3):
            for c in range(d):
                law_rows.append(dict(dataset=name,regime=regime,source=s,cell=c,probability=rows[s,c],probe=probe))
source_roles=pd.DataFrame(source_rows)
image_assignments=pd.DataFrame(image_rows)
laws=pd.DataFrame(law_rows)
source_roles.to_csv(RUN_DIR/'source_roles.csv',index=False)
image_assignments.to_csv(RUN_DIR/'image_assignments.csv',index=False)
laws.to_csv(RUN_DIR/'source_laws.csv',index=False)
display(source_roles.groupby(['dataset','source']).original_class.apply(list).to_frame())
geometry=pd.DataFrame([dict(dataset=n,regime=r,probe=w['probe'],cells=w['A'].shape[1],
    affine_difference_smallest_singular_value=w['sigma']) for (n,r),w in worlds.items()])
display(geometry)
geometry.to_csv(RUN_DIR/'geometry.csv',index=False)
''')
md('''### 4. Inspect the source laws before interpreting uncertainty

Each row below is a source's probability distribution over frozen classifier cells. Similar rows are hard to distinguish from inputs. Exact confounding is stronger than visual similarity: a candidate row that is a convex combination of known rows can trade off with their mixture weights without changing the observable input law.

The singular value reported above is for the two difference columns ((A_0-q,A_1-q)); it diagnoses affine independence but is **not** the draft's tangent-space-normalized singular value. We use it descriptively, not as a theorem constant.''')
code('''# Goal: visualize the observable source laws for every dataset and both regimes.
fig,axes=plt.subplots(2,4,figsize=(17,6),squeeze=False)
for col,name in enumerate(DATASETS):
    for row,regime in enumerate(['image_derived','confounded']):
        ax=axes[row,col]; w=worlds[name,regime]
        im=ax.imshow(w['A'],vmin=0,vmax=1,aspect='auto',cmap='viridis')
        ax.set(title=f'{name}: {regime}',xlabel='Frozen prediction cell',yticks=[0,1,2],yticklabels=['Known 0','Known 1','Candidate'])
fig.tight_layout()
fig.savefig(RUN_DIR/'source_laws.png',dpi=160,bbox_inches='tight')
fig.savefig(RUN_DIR/'source_laws.pdf',bbox_inches='tight')
plt.show()
''')
md(r'''### 5. Experiment A — the population identification floor

We now give inference the **exact** input laws (A,q,p), but not the true deployment weights. The compatible set is (\{w:w^TA=p,w\in\Delta\}). Min/max membership over this set is the input-only identification interval.

**Fundamental:** if two weights have the same marginal but different membership probabilities, no number of unlabeled inputs distinguishes them. This is a direct algebraic observational-equivalence argument, not an empirical claim about dropout.

Then reveal exact deployment weights as an oracle comparison. With exact rows and positive probe mass, membership probabilities become identified. The reference target remains an independent question: inputs do not depend on (	heta) in this model. With no target-label information, (	heta\in[0,1]^2) gives a label interval ([0,1]); exact (	heta) collapses it.

These oracle calculations are benchmarks, not acquisitions given to finite-sample inference. In particular, the finite reference estimates below are genuinely uncertain.''')
code('''# Goal: compute sharp oracle input-only membership floors and the identity oracle.
oracle_rows=[]
for (name,regime),w in worlds.items():
    A,p,c=w['A'],w['p'],w['probe']
    w['truth']={'known_membership':W_TRUE[0]*A[0,c]/(W_TRUE[:2]@A[:2,c]),
                'candidate_membership':W_TRUE[2]*A[2,c]/p[c],
                'anchored_target':float((ANCHOR*A[:2,c])@THETA/(ANCHOR@A[:2,c]))}
    for information,weight_box in [('exact_inputs',None),('exact_inputs_and_weights',(W_TRUE,W_TRUE))]:
        bounds=membership_bounds(A,A,(p,p),weight_box,c)
        for target,out in bounds.items():
            oracle_rows.append(dict(dataset=name,regime=regime,target=target,information=information,
                truth=w['truth'][target],**out))
    for information,tlo,thi in [('exact_inputs_no_target_labels',np.zeros(2),np.ones(2)),
                                ('exact_inputs_and_target_rates',THETA,THETA)]:
        out=anchored_label_bounds(A,A,tlo,thi,c)
        oracle_rows.append(dict(dataset=name,regime=regime,target='anchored_target',information=information,
            truth=w['truth']['anchored_target'],**out))
oracle=pd.DataFrame(oracle_rows)
assert (oracle[oracle.information=='exact_inputs_and_weights'].width<1e-7).all()
assert (oracle[(oracle.regime=='confounded')&(oracle.target=='candidate_membership')&
               (oracle.information=='exact_inputs')].width>0.1).all()
oracle.to_csv(RUN_DIR/'oracle_intervals.csv',index=False)
display(oracle.round(5))
''')
md(r'''### 6. Experiment B — acquire one information channel at a time

The baseline is 500 deployment inputs, 200 reference cells **per known source**, 100 noisy binary target labels **per known source**, and zero deployment source identities. At each budget add information to exactly one channel; all other observations remain the same within a repeat.

- `inputs`: add (b) unlabeled deployment cells.
- `references`: add (b) source-stratified reference cells **per known source** (total cost (2b) observations).
- `target_labels`: add (b) source-stratified noisy target labels **per known source** (total (2b)). These labels have no paired cells in this experiment.
- `source_ids`: add (b) reliable representative deployment identities (total (b)); paired cell information is deliberately unused.

These are information-response curves, **not a cost-normalized acquisition ranking**. We save both per-source sample sizes and total acquired observations. Counts are nested within a channel; the confidence boxes are recomputed and need not be nested because their centers move. Consequently individual realized widths need not decrease at every step.

For each setting we save lower/upper target endpoints, true target, inclusion, width, solver residual, and status. We also save all categorical count streams: they are sufficient statistics for these calculations. The candidate row stays specified; known rows are estimated with confidence boxes. No target truth is used in interval computation.

**Predictions to test, not force:** input acquisition should approach an identification floor; source identities can cross a positive input-only floor, although finite reference uncertainty may still limit them. Target labels should narrow the anchored-target interval, while reference inputs can affect all three targets. Six repeats illustrate behavior; they do not establish precise frequentist coverage.''')
code('''# Goal: run paired acquisitions with uncertain known rows and save every sufficient statistic.
all_rows=[]; count_rows=[]
ARMS=['inputs','references','target_labels','source_ids']
for world_index,((name,regime),w) in enumerate(worlds.items()):
    A,p,c=w['A'],w['p'],w['probe']; d=A.shape[1]
    for repeat in range(REPEATS):
        rng=np.random.default_rng(np.random.SeedSequence([SEED,world_index,repeat]))
        streams={}
        streams['inputs']=categorical_stream(p,[BASE['inputs']+b for b in BUDGETS],rng)
        streams['source_ids']=categorical_stream(W_TRUE,BUDGETS,rng)
        for s in range(2):
            streams[f'references_{s}']=categorical_stream(A[s],[BASE['references']+b for b in BUDGETS],rng)
            rate=ETA+(1-2*ETA)*THETA[s]
            streams[f'target_labels_{s}']=categorical_stream([1-rate,rate],[BASE['target_labels']+b for b in BUDGETS],rng)
        for channel,stream in streams.items():
            for size,counts in stream.items():
                count_rows.append(dict(dataset=name,regime=regime,repeat=repeat,channel=channel,n=size,counts=json.dumps(counts.tolist())))
        for arm in ARMS:
            for budget in BUDGETS:
                sizes={k:v+(budget if k==arm else 0) for k,v in BASE.items()}
                lo,hi=A.copy(),A.copy()
                tlo,thi=[],[]
                for s in range(2):
                    lo[s],hi[s]=frequency_box(streams[f'references_{s}'][sizes['references']],ALPHA/8)
                    yl,yu=frequency_box(streams[f'target_labels_{s}'][sizes['target_labels']],ALPHA/8)
                    # A noise-model-incompatible interval is an empty model set, not clipped certainty.
                    tlo.append(max(0.,(yl[1]-ETA)/(1-2*ETA)))
                    thi.append(min(1.,(yu[1]-ETA)/(1-2*ETA)))
                marginal=frequency_box(streams['inputs'][sizes['inputs']],ALPHA/4)
                weights=frequency_box(streams['source_ids'][sizes['source_ids']],ALPHA/4)
                bounds=membership_bounds(lo,hi,marginal,weights,c)
                if np.any(np.array(tlo)>np.array(thi)):
                    bounds['anchored_target']=dict(lower=np.nan,upper=np.nan,width=np.nan,status='empty',residual=np.nan)
                else:
                    bounds['anchored_target']=anchored_label_bounds(lo,hi,tlo,thi,c)
                for target,out in bounds.items():
                    truth=w['truth'][target]
                    all_rows.append(dict(dataset=name,regime=regime,repeat=repeat,arm=arm,budget=budget,
                        total_added_observations=budget*(2 if arm in ['references','target_labels'] else 1),
                        **{f'n_{k}':v for k,v in sizes.items()},target=target,truth=truth,
                        included=bool(out['lower']-1e-7<=truth<=out['upper']+1e-7),**out))
    print('Completed',name,regime,flush=True)
    pd.DataFrame(all_rows).to_csv(RUN_DIR/'acquisition_intervals.csv',index=False)
    pd.DataFrame(count_rows).to_csv(RUN_DIR/'acquisition_counts.csv',index=False)
results=pd.DataFrame(all_rows)
assert np.nanmax(results.residual)<1e-6
summary=results.groupby(['dataset','regime','arm','budget','target'],as_index=False).agg(
    mean_width=('width','mean'),min_width=('width','min'),max_width=('width','max'),
    mean_lower=('lower','mean'),mean_upper=('upper','mean'),inclusion_rate=('included','mean'),
    rejected=('status',lambda x:int((x=='empty').sum())),max_solver_residual=('residual','max'))
summary.to_csv(RUN_DIR/'acquisition_summary.csv',index=False)
print('Intervals:',len(results),'| empty sets:',int((results.status=='empty').sum()),
      '| undefined denominator policies:',int((results.status=='undefined_denominator_policy').sum()))
display(summary[summary.budget==max(BUDGETS)].round(4))
''')
md('''### 7. Read the acquisition curves against the correct benchmark

Each dataset gets six panels: three targets by two regimes. Lines are mean widths over six repeats; shading is the **minimum-to-maximum observed width**, not a confidence band. A width of zero means the target probability is determined inside the set, not that the target event is deterministic.

For the two membership targets, the dashed line is the **exact-reference, exact-input, input-only** identification floor. It is a population comparison, not a universal lower bound on noisy realized confidence intervals. Identity information can legitimately cross it; noisy finite-sample sets can also miss truth on their failure event. For the anchored target, the dashed line is the no-target-label oracle width of one. Label data were already present at baseline, so being below it is expected.

The horizontal axis is added observations per channel/source as defined above. All panels use probability-width units [0,1].''')
code('''# Goal: show which acquisitions change which target widths, without implying additivity.
COLORS=dict(inputs='#2171b5',references='#238b45',target_labels='#88419d',source_ids='#d94801')
TARGETS=['known_membership','anchored_target','candidate_membership']
for name in DATASETS:
    fig,axes=plt.subplots(2,3,figsize=(15,8),squeeze=False)
    for row,regime in enumerate(['image_derived','confounded']):
        for col,target in enumerate(TARGETS):
            ax=axes[row,col]
            for arm in ARMS:
                f=summary[(summary.dataset==name)&(summary.regime==regime)&(summary.target==target)&(summary.arm==arm)]
                ax.plot(f.budget,f.mean_width,'o-',color=COLORS[arm],label=arm)
                ax.fill_between(f.budget,f.min_width,f.max_width,color=COLORS[arm],alpha=.12)
            info='exact_inputs_no_target_labels' if target=='anchored_target' else 'exact_inputs'
            floor=float(oracle[(oracle.dataset==name)&(oracle.regime==regime)&(oracle.target==target)&(oracle.information==info)].width.iloc[0])
            ax.axhline(floor,color='black',ls='--',lw=1,label='Input-only oracle width')
            ax.set(xscale='symlog',xlim=(0,max(BUDGETS)),ylim=(-.02,1.03),
                   title=f'{regime}: {target}',xlabel='Added samples (per source for references/labels)',ylabel='Probability interval width')
    axes[0,0].legend(fontsize=8)
    fig.suptitle(f'{name}: acquisition-dependent imprecision; bands are repeat ranges')
    fig.tight_layout()
    for ext in ['png','pdf']: fig.savefig(RUN_DIR/f'{name}_acquisition_curves.{ext}',dpi=160,bbox_inches='tight')
    plt.show()
''')
md('''### 8. Inspect actual endpoints, not only widths

We select repeat 0 in the constructed-confounding regime. For candidate membership, display the baseline and each maximum-budget acquisition alongside the true probability. This is a predetermined illustrative repeat, not a selected success case. Different interval centers can yield the same width. Numerical lower/upper target bounds must not be confused with lower/upper values of an entropy-based epistemic score.

**Transition:** endpoint plots show the uncertainty object; the next cell checks inclusion, empty sets, and which acquisition effects were actually observed.''')
code('''# Goal: display interpretable membership intervals and retain the plotted numerical table.
endpoint_rows=[]
fig,axes=plt.subplots(1,4,figsize=(17,4),squeeze=False)
for ax,name in zip(axes[0],DATASETS):
    f=results[(results.dataset==name)&(results.regime=='confounded')&(results.repeat==0)&(results.target=='candidate_membership')]
    selected=pd.concat([f[(f.arm=='inputs')&(f.budget==0)],f[f.budget==max(BUDGETS)]]).copy()
    selected['display_label']=['baseline']+selected.arm.iloc[1:].tolist()
    endpoint_rows.append(selected)
    for j,(_,r) in enumerate(selected.iterrows()):
        ax.plot([r.lower,r.upper],[j,j],lw=3,color=COLORS[r.arm])
        ax.plot([r.lower,r.upper],[j,j],'|',color='black')
    ax.axvline(selected.truth.iloc[0],color='black',ls=':',label='True membership')
    ax.set(title=name,yticks=range(len(selected)),yticklabels=selected.display_label,xlim=(0,1),xlabel='Candidate membership probability')
fig.tight_layout()
for ext in ['png','pdf']: fig.savefig(RUN_DIR/f'candidate_endpoints.{ext}',dpi=160,bbox_inches='tight')
plt.show()
pd.concat(endpoint_rows).to_csv(RUN_DIR/'plotted_endpoints.csv',index=False)
''')
md('''### 9. What did this run establish, and what remains conditional?

We report width changes, target inclusion, empty-set counts, and numerical residuals. An interval that is narrow but excludes the true target is not counted as a scientific success. An empirical inclusion fraction from six repeats cannot validate a nominal 95% guarantee; the guarantee rests on [H], the union allocation, the model assumptions, and correct optimization.

We explicitly check two experimental invariances: varying target-label counts leaves membership intervals unchanged; varying deployment inputs/identities leaves the reference-anchored target interval unchanged. These follow from the declared observation channels and fixed anchor, not from a universal separation of epistemic uncertainty types. Reference acquisitions can affect all targets.

No automatic monotonicity assertion is made for changing finite-sample boxes. Exact input-only identification floors are model properties; finite widths also include estimation uncertainty and conservatism. The chosen cell representation can itself conceal distinctions present in raw images. The confounded candidate is synthetic, the candidate law and label-noise rate are specified, and natural population generalization is untested.''')
code('''# Goal: audit invariances and generate a result-dependent report rather than a canned conclusion.
for keys,f in results.groupby(['dataset','regime','repeat','target']):
    target=keys[-1]
    invariant_arms=['target_labels'] if target!='anchored_target' else ['inputs','source_ids']
    for arm in invariant_arms:
        g=f[f.arm==arm]
        assert np.allclose(g.lower,g.lower.iloc[0],equal_nan=True)
        assert np.allclose(g.upper,g.upper.iloc[0],equal_nan=True)
checks=results.groupby(['dataset','regime','target'],as_index=False).agg(
    empirical_inclusion=('included','mean'),empty_sets=('status',lambda s:int((s=='empty').sum())),
    undefined_denominators=('status',lambda s:int((s=='undefined_denominator_policy').sum())),
    max_solver_residual=('residual','max'))
checks.to_csv(RUN_DIR/'inclusion_and_solver_checks.csv',index=False)
# Pooled inclusion below is descriptive: arms share observations and are not independent trials.
display(checks)
last=summary[summary.budget==max(BUDGETS)]
report=['# Acquisition experiment results','',
    'This is an image-derived finite-cell simulation, not a natural-image benchmark.',
    f'{REPEATS} paired repeats; {len(results)} target intervals; budget maximum {max(BUDGETS)}.',
    'Reference and target-label budgets are per known source; their total cost is twice the x-axis value.','',
    '## Candidate membership under constructed confounding','']
for name in DATASETS:
    floor=oracle[(oracle.dataset==name)&(oracle.regime=='confounded')&(oracle.target=='candidate_membership')&(oracle.information=='exact_inputs')].width.iloc[0]
    f=last[(last.dataset==name)&(last.regime=='confounded')&(last.target=='candidate_membership')].set_index('arm')
    report.append(f"- {name}: exact input-only width {floor:.4f}; finite mean width after inputs {f.loc['inputs','mean_width']:.4f}; after source identities {f.loc['source_ids','mean_width']:.4f}.")
report += ['', '## Interpretation limits','',
    'Positive exact floors demonstrate input-only non-identification in the declared model. Whether finite identity acquisition crosses them is read from the numbers above; finite reference uncertainty remains.',
    'All target endpoints, inclusion indicators, solver residuals, class roles, source laws and acquisition counts are exported.',
    'Widths concern distinct probabilities and are not an additive decomposition. Noise is specified; aleatoric uncertainty is not estimated.',
    'Inclusion summaries pool dependent settings and are descriptive. Six repeats do not establish precise calibration.',
    'See SOURCES.md for every external theoretical and dataset source, and config.json for assumptions.']
report_text='\\n'.join(report)
(RUN_DIR/'RESULTS.md').write_text(report_text)
display(Markdown(report_text))
''')
md('''### 10. Save the complete audit trail

The folder contains all numerical tables, figures (PNG/PDF), exact empirical laws, finite acquisition counts, selected source roles, upstream file hashes, dependency versions, the solver source, this notebook, and the external source list. The executed notebook is saved by the execution runner as `executed.ipynb`; the editable notebook stays next to the earlier DAPPr-style notebook.

Reproducibility has two levels: the input pilot artifacts reproduce the frozen finite population; saved count tables reproduce each inference calculation without drawing again. Regenerating with the same configuration and dependency versions also reproduces the simulation streams. No artifact in an earlier run is overwritten.

A next study could increase repeats, rotate class partitions, alter representation cells, vary the candidate mismatch, or normalize acquisition costs. Those are separate experiments; this notebook first establishes a transparent example of the existing scope.''')
code('''# Goal: export provenance and a machine-readable manifest for this exact run.
config=dict(seed=SEED,repeats=REPEATS,budgets=BUDGETS,baseline=BASE,alpha_per_setting=ALPHA,
    label_noise_rate=ETA,smoothing=SMOOTHING,true_deployment_weights=W_TRUE.tolist(),
    target_rates=THETA.tolist(),anchor=ANCHOR.tolist(),datasets=DATASETS,
    source_pilot=str(SOURCE.relative_to(ROOT)),
    population='uniform-per-class empirical cell laws with 2% uniform replacement',
    candidate='specified empirical OOD group or exact known-row average',
    inference='joint-mass coordinate-box polytope; Charnes-Cooper LP',
    source_identity_channel='representative source counts only; paired cells unused',
    python=sys.version,versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','matplotlib']})
(RUN_DIR/'config.json').write_text(json.dumps(config,indent=2))
inputs=[SOURCE/'scores.csv',SOURCE/'class_roles.csv',SOURCE/'assignments.csv',SOURCE/'config.json']
inputs += [SOURCE/f'{name}_dropout_predictions.npz' for name in DATASETS]
inputs += [ROOT/'real/source_acquisition.py',ROOT/'paper/sections/setup.tex',ROOT/'paper/sections/theory.tex']
manifest=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in inputs]
(RUN_DIR/'input_manifest.json').write_text(json.dumps(manifest,indent=2))
shutil.copy2(ROOT/'real/source_acquisition.py',RUN_DIR/'source_acquisition.py')
# Source snapshot is cleaned of prior outputs below.
nb=json.loads((ROOT/'source_acquisition_four_datasets.ipynb').read_text())
for cell in nb['cells']:
    if cell['cell_type']=='code':
        cell['outputs']=[]
        cell['execution_count']=None
(RUN_DIR/'source_notebook.ipynb').write_text(json.dumps(nb,indent=1))
source_text=''.join(nb['cells'][1]['source'])
(RUN_DIR/'SOURCES.md').write_text(source_text)
(RUN_DIR/'README.md').write_text('# Four-dataset acquisition example\\n\\nStart with executed.ipynb (narrative + outputs), RESULTS.md and SOURCES.md.\\n\\n'
    'Tables: source_laws, source_roles, image_assignments, geometry, oracle_intervals, acquisition_counts, '
    'acquisition_intervals, acquisition_summary, plotted_endpoints, inclusion_and_solver_checks.\\n\\n'
    'Figures: source_laws, four acquisition_curves, candidate_endpoints (PNG/PDF). '
    'config.json and input_manifest.json record assumptions and provenance. '
    'The editable source notebook is two directories above this run folder.\\n')
output_manifest=[dict(file=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                 for p in sorted(RUN_DIR.iterdir()) if p.is_file()]
(RUN_DIR/'artifact_manifest.json').write_text(json.dumps(output_manifest,indent=2))
print('Complete:',RUN_DIR)
''')

# Dollar delimiters are portable in Jupyter; construct LaTeX backslashes explicitly.
cells[3].source = r'''### 1. Specify the targets before calculating uncertainty

There are three sources: two known class groups $S=0,1$ and one specified candidate-novel source $S=2$. Sources differ from frozen classifier-output cells $X=c$. Define $A_{sc}=P(X=c@mid S=s)$, $w_s=P(S=s)$, and $p_c=@sum_s w_s A_{sc}$.

At a fixed probe cell, the three targets are:

1. **Known-source membership:** $T_K(c)=w_0A_{0c}/(w_0A_{0c}+w_1A_{1c})$. This is membership in known group 0 conditional on belonging to a known group. It is not the probability of being ID.
2. **Candidate-novel membership:** $T_N(c)=w_2A_{2c}/p_c$.
3. **Reference-anchored binary target:** $T_Y(c)=@sum_{s<2}a_s A_{sc}@theta_s/@sum_{s<2}a_s A_{sc}$, with fixed anchor $a=(1/2,1/2)$. This is not the deployment label probability. Novel-source labels are outside this target.

For each target, endpoints range over compatible models. Their difference is epistemic imprecision within this model [E]. These widths are **not an additive decomposition**. A point interval can describe a probability between zero and one: knowing a probability does not determine individual outcomes.

**Assumptions:** source cell laws remain fixed [L]; draws are i.i.d.; references are stratified by reliable source identity; clean binary target $Y$ is independent of cell given source; observed target labels flip independently at a specified rate $@eta<1/2$; deployment source identities are representative and correct. The identity channel uses identity counts only, not paired cells. It learns weights without also improving reference rows. Aleatoric uncertainty is not estimated.

**Novelty assumption:** candidate $q=A_2$ is specified, not inferred from the unlabeled sample. Held-out class data define it for the experiment designer. This is candidate-source inference, not arbitrary unknown-class detection.
'''.replace('@', chr(92))
cells[4].source = r'''### 2. How finite information defines the compatible set

For $D$ categorical frequencies from $n$ i.i.d. observations, use

$$r(n,D,@delta)=@sqrt{@log(2D/@delta)/(2n)}.$$

Hoeffding's bound for indicator variables and a union bound give simultaneous coordinate inclusion with probability at least $1-@delta$ [H]. With no observations the box is $[0,1]^D$.

At one fixed run/budget, split $@alpha=.05$ into four channel shares. Split the reference and target-label shares again between the two sources. This gives **at least 95% simultaneous inclusion of all three targets at that fixed setting**, conditional on the model and exact optimization. It is not a simultaneous guarantee over all datasets, repeats or budgets; it is not an anytime confidence sequence.

Let $z_{sc}=P(S=s,X=c)$. Require nonnegativity, $@sum_{sc}z_{sc}=1$, and

$$@underline A_{sc}@sum_j z_{sj}@le z_{sc}@le@overline A_{sc}@sum_j z_{sj}.$$

Constrain column sums using the deployment-input box and row sums using the identity box. The candidate row has lower=upper $q$. This retains row uncertainty jointly rather than substituting estimates into the posterior.

Each membership target is $c^Tz/d^Tz$. Set $t=1/(d^Tz)$ and $y=tz$. Then $Gz@le h$ becomes $Gy@le ht$, normalization becomes $@sum y=t$, and $d^Ty=1$. Min/max of $c^Ty$ is a linear program [C]. We derive this transformation here; we do not assume the citation establishes validity for an undefined denominator. A compatible denominator at or below numerical tolerance $10^{-9}$ triggers an explicit $[0,1]$ policy; an empty set is recorded as rejection, not certainty.

For target-label noise, $P(@widetilde Y=1@mid S=s)=@eta+(1-2@eta)@theta_s$. Invert the observed-label interval and intersect with $[0,1]$. The fixed anchor defines a separate two-source LP; positivity makes target-label endpoints occur at the matching bounds on $@theta$.

**Transition:** check an analytically solvable example before using image-derived laws.
'''.replace('@', chr(92))
cells[10].source = r'''### 5. Experiment A — the population identification floor

Give inference the exact input laws $A,q,p$, but not deployment weights. The compatible set is $@{w:w^TA=p,w@in@Delta@}$. Min/max membership over this set is the input-only identification interval.

**Fundamental:** equal observable input laws cannot distinguish different membership probabilities. This is algebraic observational equivalence, not a dropout claim.

Next reveal exact deployment weights as an oracle comparison. With exact rows and positive probe mass, membership becomes identified. Inputs do not depend on $@theta$ under the declared model. With no target labels, $@theta@in[0,1]^2$ gives anchored-target interval $[0,1]$; exact target rates collapse it.

These oracles are benchmarks, not extra observations supplied to finite-sample inference. Known reference rows below remain uncertain. The equality $q=(A_0+A_1)/2$ defines the confounded data-generating world; finite inference knows the specified $q$, but is not additionally told this relation constrains its uncertain reference rows.
'''.replace('@', chr(92))

for cell in cells:
    if cell.cell_type == 'markdown':
        cell.source = cell.source.replace(chr(92)*2, chr(92))
n['cells']=cells
n.metadata={'kernelspec':{'display_name':'Python (reach-aistats)','language':'python','name':'reach-aistats'},'language_info':{'name':'python','version':'3.13'}}
nbf.write(n,ROOT/'source_acquisition_four_datasets.ipynb')
print(ROOT/'source_acquisition_four_datasets.ipynb')
