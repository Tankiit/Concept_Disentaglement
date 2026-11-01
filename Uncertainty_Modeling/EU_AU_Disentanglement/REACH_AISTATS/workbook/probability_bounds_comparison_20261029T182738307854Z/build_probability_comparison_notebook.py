"""Build the actual Probly probability-representation comparison notebook."""
from pathlib import Path
import nbformat as nb
ROOT=Path(__file__).resolve().parents[1]
old=nb.read(ROOT/'dappr_epistemic_sets.ipynb',as_version=4)
cells=[]
def md(t): cells.append(nb.v4.new_markdown_cell(t))
def code(t): cells.append(nb.v4.new_code_cell(t))
md('''# Replacing possibilistic outputs with lower and upper probabilities
## DAPPr-loss baseline and two Probly representations on four datasets

**MNIST, EMNIST Letters, Fashion-MNIST, BloodMNIST.**

We now implement the requested prediction pipeline, not just the earlier acquisition simulation. Train an ID classifier, call `represent(...).lower/upper` on real ID/OOD images, quantify interval imprecision, and evaluate separation. Compare two explicitly different sets: the convex hull of MC-dropout predictions and the source-mixture set compatible with reference/deployment observations. A separate model trained with the **DAPPr objective** provides the original scalar-score baseline.

This is a matched small-CNN adaptation of DAPPr's code, not a reproduction of its ResNet50 benchmarks. The CE branch supplies both new interval methods. Replacing the representation does not by itself derive a new learning objective: this pilot uses cross-entropy for that branch and post-training set inference. We do not train a network to approximate source-mixture endpoints.

Every experiment's goal is stated before its code cell. Run in `reach-aistats`. All outputs go to a fresh `workbook/probability_bounds_comparison_<UTC timestamp>/`. No previous experiment is overwritten.''')
md('''### Explicit sources and adaptation map

| Source | Used here |
|---|---|
| [DAPPr repository](https://github.com/MaxwellYaoNi/DAPPr), [train.py](https://github.com/MaxwellYaoNi/DAPPr/blob/main/train.py) | AdamW, warmup/cosine scheduling, ID-validation checkpoint selection, ID-positive confidence/OOD AUPR convention. Architecture, datasets and schedule length are explicitly adapted. |
| [DAPPr.py](https://github.com/MaxwellYaoNi/DAPPr/blob/main/DAPPr.py), [paper](https://arxiv.org/abs/2605.00600) | Actual DAPPr loss formula and original score K/sum(alpha), implemented below. No claim that our interval score is this scalar. |
| [Probly README](https://github.com/pwhofman/probly), [representer interface](https://github.com/pwhofman/probly/blob/main/src/probly/representer/_representer.py) | Actual `dropout`, `representer`, `Representation`, custom `quantify` dispatch and OOD evaluation API. Installed version is recorded; source snapshots are saved. |
| Gal & Ghahramani (2016), [Dropout as a Bayesian Approximation](https://proceedings.mlr.press/v48/gal16.html) | MC-dropout sampled predictors. This does not supply coverage guarantees for sample min/max. |
| Hüllermeier & Waegeman (2021), [uncertainty concepts](https://doi.org/10.1007/s10994-021-05946-3) | Sets of candidate probability distributions describe incomplete knowledge. |
| Lipton et al. (2018), [label-shift estimation](https://proceedings.mlr.press/v80/lipton18a.html) | Frozen source-conditionals and mixture moment equation; identification depends on source distinguishability. |
| Hoeffding (1963), [bounded-variable concentration](https://doi.org/10.1080/01621459.1963.10500830) | Coordinate confidence boxes for known-source reference rows and deployment histogram, conditional on sampling/model assumptions. |
| Charnes & Cooper (1962), [linear-fractional programming](https://doi.org/10.1002/nav.3800090303) | Exact ratio extrema over the declared joint-mass polytope, subject to numerical tolerances. |
| [MNIST](https://yann.lecun.com/exdb/mnist/), [EMNIST](https://arxiv.org/abs/1702.05373), [Fashion-MNIST](https://arxiv.org/abs/1708.07747), [MedMNIST v2](https://doi.org/10.1038/s41597-022-01721-8) | Dataset provenance. Fashion uses the previously cached official-test repartition and is not an official benchmark result. |

Local mathematical foundation: `real/source_acquisition.py` and the previous `source_acquisition_four_datasets.ipynb`. Here known sources are individual ID classes, rather than the previous two groups. The candidate comprises all within-dataset OOD classes, mixed with equal class weight. This method has candidate-reference access; dropout and DAPPr training do not. Therefore the comparison is not an equal-information superiority test.''')
code('''# Goal: declare all choices before seeing results and create a fresh artifact directory.
from pathlib import Path
import sys,json,copy,math,hashlib,inspect,importlib.metadata,shutil
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import display,Markdown
from probly.method import dropout
from probly.representer import representer
from probly.quantification import quantify
from probly.evaluation.ood import evaluate_ood
ROOT=Path.cwd()
if not (ROOT/'real/probability_bounds.py').exists():
    ROOT=ROOT/'Uncertainty_Modeling/EU_AU_Disentanglement/REACH_AISTATS'
sys.path.insert(0,str(ROOT))
from real.probability_bounds import (DropoutEnvelopeRepresenter,SourceMixtureBoundsRepresenter,
    envelope_from_samples,evaluate_probability_ood)
from real.source_acquisition import frequency_box
DATASETS=['mnist','emnist','fashion','blood']
ID_CLASS_COUNT=dict(mnist=4,emnist=13,fashion=4,blood=3)
SEED=0; TRAIN_PER_CLASS=400; VAL_PER_CLASS=80; TEST_PER_CLASS=100
REFERENCE_PER_CLASS=200
EPOCHS=15; BATCH_SIZE=128; LR=2e-3; WEIGHT_DECAY=1e-4
WARMUP_EPOCHS=2; DROPOUT=.25; MC_PASSES=20
ALPHA=.05; CELL_CONFIDENCE_THRESHOLD=.7
CACHE=ROOT/'cache/mnist_reach'
DEVICE=torch.device('cpu'); torch.set_num_threads(4)
RUN_DIR=ROOT/'workbook'/('probability_bounds_comparison_'+pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ'))
RUN_DIR.mkdir(parents=True,exist_ok=False)
plt.rcdefaults()
print('Probly',importlib.metadata.version('probly'),'|',DEVICE,'| outputs:',RUN_DIR)
''')
md(r'''### 1. What each representation means

**Dropout envelope.** A real Probly sampler returns probability vectors $p_t(x)$, $t=1,\ldots,T$. Define their convex hull $C_{MC}(x)$. Because each class probability is a linear functional, its exact hull endpoints are

$$L_k(x)=\min_t p_{tk}(x),\qquad U_k(x)=\max_t p_{tk}(x).$$

This represents disagreement in a finite sampled family. More MC draws can widen the envelope by discovering additional extremes; this is not evidence that additional training data increase epistemic uncertainty. Missing hypotheses cannot be recovered by taking a min/max. There is no claimed frequentist coverage.

**Source-mixture set.** Sources are K known classes plus a candidate. A frozen deterministic cell map assigns each image to one of 2K cells: predicted class and whether max-softmax is below the preset 0.7 threshold. Keeping only K argmax cells would make K+1 source rows necessarily affinely dependent, since the K-cell simplex has dimension K-1. The extra confidence bin permits, but does not guarantee, identification.

Estimate each known-source cell law from a separate reference pool. Declare the candidate law q from candidate-reference images; this is a specified reference-anchored candidate, not a guaranteed deployment population law. Infer a compatible set from these row boxes and the unlabeled test histogram. Joint masses z[s,c] satisfy row-conditional boxes, marginal boxes, nonnegativity and total mass one. Each known class endpoint is an optimized ratio z[k,c]/sum_known(z[:,c]); candidate membership is z[candidate,c]/sum_all(z[:,c]). The helper solves these linear-fractional programs via Charnes–Cooper.

**Comparison scope.** Both main outputs are N×K lower/upper known-class probabilities. For source inference these are explicitly conditional on known membership. On an actual OOD example that conditional target is a model diagnostic, not its true image class. Only source inference has a separate candidate-membership interval. These are different uncertainty sets and should not be expected to agree.

**Quantification.** We register `quantify(ProbabilityBounds).epistemic = max_k(U_k-L_k)`. This is our declared imprecision summary in probability units, not mutual information or an additive decomposition. Lower/upper vectors need not sum to one, and their midpoint is not automatically a coherent predictive distribution. General event bounds must be optimized over the underlying set, not formed by adding singleton extrema.''')
code('''# Goal: exercise the requested Probly interface and expose the tie-handling audit before evaluation.
check_net=dropout(nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,3)),p=.25,predictor_type='logit_classifier')
check_net.eval()
base_rep=representer(check_net,num_samples=MC_PASSES)
rep=DropoutEnvelopeRepresenter(base_rep)
out=rep.represent(torch.ones(2,4))
print('lower:',out.lower,'upper:',out.upper)
print('epistemic width:',quantify(out).epistemic)
audit=evaluate_probability_ood(np.zeros(4),np.zeros(4))
print('Identical ID/OOD constant scores:',audit)
assert audit['auroc']==.5
# Installed Probly 0.9.0 NumPy ROC handling does not group ties. Never silently use it for tied cells.
''')
md('''### 2. Load real images and fix within-dataset roles

MNIST and EMNIST use official train/test caches with a stratified validation split of official training data. EMNIST is transposed to match the prior pilot. BloodMNIST uses official splits and grayscale conversion. Fashion's cached official-test pool is repartitioned 60/20/20 as in the prior pilot; all source indices are preserved. Images are divided by 255.

A seeded class permutation chooses ID classes before training. All remaining classes are OOD. Only ID images train the classifiers or select checkpoints. Additional reference rows are sampled from validation-pool images **excluding the selected checkpoint-validation rows**. OOD reference images never enter network training. Test images remain disjoint from all training, validation and reference images.

The unlabeled test images do supply the deployment histogram for source inference: this is explicitly a transductive comparison. OOD test labels are used only to evaluate scores, never to construct the compatibility set. Candidate references are a separate, disclosed information advantage.''')
code('# Goal: load cached images with the documented orientation and pool boundaries.\n'+old.cells[7].source.replace('plt.show()',"plt.gcf().savefig(RUN_DIR/'dataset_examples.png',dpi=150,bbox_inches='tight')\nplt.show()"))
code('# Goal: freeze ID/OOD class roles and disjoint image assignments before training.\n'+old.cells[9].source+'''
reference_rows=[]
for name,w in worlds.items():
    pool=raw[name]['val']; rng=np.random.default_rng(w['seed']+3000)
    used=set(w['val']['source_index'].tolist()); chosen=[]
    for c in np.r_[w['id_classes'],w['ood_classes']]:
        ids=[i for i in np.flatnonzero(pool['labels']==c) if int(pool['source_index'][i]) not in used]
        chosen.extend(rng.permutation(ids)[:REFERENCE_PER_CLASS])
    ix=np.asarray(chosen,dtype=int)
    w['reference']=dict(images=pool['images'][ix],label=pool['labels'][ix],source_index=pool['source_index'][ix])
    for i in ix:
        reference_rows.append(dict(dataset=name,split='reference',source=pool['source'],index=int(pool['source_index'][i]),label=int(pool['labels'][i])))
all_assignments=pd.concat([assignments,pd.DataFrame(reference_rows)],ignore_index=True)
assert not all_assignments.duplicated(['dataset','source','index']).any()
all_assignments.to_csv(RUN_DIR/'assignments.csv',index=False)
class_roles.to_csv(RUN_DIR/'class_roles.csv',index=False)
display(all_assignments.groupby(['dataset','split']).size().rename('count').to_frame())
''')
md('''### 3. Train matched CE and DAPPr-loss models

For each dataset, create one small CNN and pass it through actual `probly.method.dropout(..., predictor_type="logit_classifier")`. Clone its initialized parameters for two independently trained branches. One uses cross-entropy; the other uses DAPPr's loss formula from `DAPPr.py`. They share images, minibatch-order seeds, AdamW, learning-rate schedule, dropout architecture and ID-validation checkpoint selection. The original repository used ResNet50 and a longer schedule; those are not reproduced here.

DAPPr sets alpha=softplus(logits)+1; its surrogate loss and incorrect-class evidence regularizer are implemented directly below. Its baseline EU remains K/sum(alpha). Cross-entropy is the learning objective for both probability-bound representations, which differ at the representation/inference stage. This explicitly implements two alternatives rather than claiming the source-mixture bounds imply a new neural loss.

Reference images and test labels do not select checkpoints. Every selected checkpoint and full training history is saved. The loss regularization uses the original default coefficient 2e-4, warmed up over ten epochs; the short-run learning-rate warmup is two epochs for both branches.''')
code('''# Goal: use the actual DAPPr objective as a baseline and a matched CE model for both bound methods.
class SmallCNN(nn.Module):
    def __init__(self,k):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten(),
            nn.Linear(32*7*7,64),nn.ReLU(),nn.Linear(64,k))
    def forward(self,x): return self.net(x)

def dappr_loss(logits,labels,lamb):
    alpha=nn.functional.softplus(logits)+1
    y=nn.functional.one_hot(labels,logits.shape[-1]).to(alpha.dtype)
    a_star=alpha-y+1e-8
    p_star=(a_star/a_star.sum(-1,keepdim=True)).detach()
    a0=alpha.sum(-1)
    fit=a0*a0.log()+(alpha*(p_star/alpha).log()).sum(-1)
    penalty=(alpha*(1-y)).square().sum(-1)
    return fit.mean()+lamb*penalty.mean()

def image_tensor(images): return torch.from_numpy(np.ascontiguousarray(images)).float()[:,None]/255.

def logits_for(model,images):
    model.eval(); out=[]
    with torch.no_grad():
        for start in range(0,len(images),BATCH_SIZE):
            out.append(model(image_tensor(images[start:start+BATCH_SIZE]).to(DEVICE)).cpu())
    return torch.cat(out)

history_rows=[]; accuracy_rows=[]
for name,w in worlds.items():
    torch.manual_seed(w['seed'])
    initial=dropout(SmallCNN(w['K']),p=DROPOUT,predictor_type='logit_classifier')
    w['models']={}
    for method in ['ce','dappr']:
        torch.manual_seed(w['seed'])
        model=copy.deepcopy(initial).to(DEVICE)
        opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
        loader=DataLoader(TensorDataset(image_tensor(w['train']['images']),torch.tensor(w['train']['target'])),
            batch_size=BATCH_SIZE,shuffle=True,generator=torch.Generator().manual_seed(w['seed']))
        best=-1.; state=None
        for epoch in range(EPOCHS):
            lr=LR*(epoch+1)/WARMUP_EPOCHS if epoch<WARMUP_EPOCHS else 1e-5+(LR-1e-5)*.5*(1+math.cos(math.pi*(epoch-WARMUP_EPOCHS)/max(1,EPOCHS-WARMUP_EPOCHS-1)))
            for group in opt.param_groups:group['lr']=lr
            model.train();total=0.
            for xb,yb in loader:
                opt.zero_grad();logits=model(xb.to(DEVICE))
                loss=nn.functional.cross_entropy(logits,yb.to(DEVICE)) if method=='ce' else dappr_loss(logits,yb.to(DEVICE),2e-4*min(1.,epoch/10.))
                loss.backward();opt.step();total+=float(loss.detach())*len(yb)
            val=logits_for(model,w['val']['images'])
            acc=float((val.argmax(-1)==torch.tensor(w['val']['target'])).float().mean())
            history_rows.append(dict(dataset=name,method=method,epoch=epoch+1,loss=total/len(w['train']['target']),val_accuracy=acc,lr=lr))
            if acc>=best:best=acc;best_epoch=epoch+1;state=copy.deepcopy(model.state_dict())
        model.load_state_dict(state);model.eval();w['models'][method]=model
        torch.save(state,RUN_DIR/f'{name}_{method}.pt')
        test_logits=logits_for(model,w['test']['images']);idmask=~w['test']['is_ood']
        test_acc=float((test_logits.argmax(-1).numpy()[idmask]==w['test']['target'][idmask]).mean())
        w[f'{method}_logits']=test_logits
        accuracy_rows.append(dict(dataset=name,method=method,selected_epoch=best_epoch,val_accuracy=best,test_id_accuracy=test_acc))
        print(name,method,'selected epoch',best_epoch,'ID accuracy',round(test_acc,4),flush=True)
        pd.DataFrame(history_rows).to_csv(RUN_DIR/'training_history.csv',index=False)
history=pd.DataFrame(history_rows);accuracy=pd.DataFrame(accuracy_rows)
accuracy.to_csv(RUN_DIR/'accuracy.csv',index=False)
display(accuracy)
''')
md('''### 4. Represent every ID/OOD image with both kinds of probability bounds

The dropout wrapper calls the real Probly sampler and reads its categorical probability property; it never takes softmax twice. It returns the coordinate hull endpoints plus all sampled vectors. Source inference uses a deterministic eval-mode CE classifier for both reference and test cells; resampling stochastic cells would change the observation law and invalidate the comparison.

For source inference, divide failure budget alpha between the K reference-row boxes and one marginal box, using Hoeffding with a union over each row's cells. Candidate q is the equal-class average of OOD-reference histograms and is treated as specified. Thus formal inclusion would require that declared q and the frozen known conditionals actually hold in deployment. We do not know true per-image class probabilities in real datasets and therefore do not report a measured probability-coverage guarantee here. The previous controlled acquisition notebook remains the identification/inclusion experiment.

Source bounds are shared by inputs in the same cell. They are sharp within the declared coordinate-box set, apart from solver tolerance; that set can be conservative. Zero-denominator-compatible cases return [0,1] with a status; empty sets return NaN and are reported separately. Candidate membership is a separate event, not an uncertainty score: we evaluate its lower/upper endpoints separately rather than calling a high membership probability high epistemic uncertainty.''')
code('''# Goal: perform the requested represent/quantify pipeline and retain the full lower/upper arrays.
def cell_fn(logits):
    p=logits.softmax(-1)
    return p.argmax(-1)+p.shape[1]*(p.max(-1).values<CELL_CONFIDENCE_THRESHOLD).long()

metric_rows=[]; score_frames=[]; bound_frames=[]; source_law_rows=[]
for name,w in worlds.items():
    model=w['models']['ce']; model.eval()
    # deepcopy does not consistently preserve instance registry markers in Probly 0.9.0;
    # the explicit sampler class uses the same sampling engine and a categorical factory if needed.
    from probly.representer.sampler import Sampler
    from probly.representation.distribution.torch_categorical import TorchLogitCategoricalDistribution
    from probly.representation.sample import create_sample
    def categorical_sample(logits,sample_axis=-1):
        return create_sample([TorchLogitCategoricalDistribution(tensor=z) for z in logits],sample_axis=sample_axis)
    base_rep=Sampler(model,num_samples=MC_PASSES,sample_factory=categorical_sample)
    rep=DropoutEnvelopeRepresenter(base_rep)
    reps=[]
    torch.manual_seed(w['seed']+20000)
    for start in range(0,len(w['test']['images']),BATCH_SIZE):
        reps.append(rep.represent(image_tensor(w['test']['images'][start:start+BATCH_SIZE]).to(DEVICE)))
    draws=torch.cat([r.samples.cpu() for r in reps])
    mc=envelope_from_samples(draws)
    d=2*w['K'];ref_logits=logits_for(model,w['reference']['images'])
    ref_cells=cell_fn(ref_logits).numpy();test_cells=cell_fn(w['ce_logits']).numpy()
    rowlo=[];rowhi=[];nominal_rows=[]
    for k,label in enumerate(w['id_classes']):
        counts=np.bincount(ref_cells[w['reference']['label']==label],minlength=d)
        assert counts.sum()>0
        lo,hi=frequency_box(counts,ALPHA/(w['K']+1))
        rowlo.append(lo);rowhi.append(hi);nominal_rows.append(counts/counts.sum())
        for cell in range(d): source_law_rows.append(dict(dataset=name,source=int(label),candidate=False,cell=cell,count=int(counts[cell]),n=int(counts.sum()),lower=lo[cell],upper=hi[cell]))
    q=np.mean([np.bincount(ref_cells[w['reference']['label']==label],minlength=d)/sum(w['reference']['label']==label) for label in w['ood_classes']],axis=0)
    rowlo.append(q);rowhi.append(q);nominal_rows.append(q)
    for cell in range(d): source_law_rows.append(dict(dataset=name,source=-1,candidate=True,cell=cell,count=np.nan,n=np.nan,lower=q[cell],upper=q[cell]))
    counts=np.bincount(test_cells,minlength=d)
    marginal=frequency_box(counts,ALPHA/(w['K']+1))
    source_rep=SourceMixtureBoundsRepresenter(model,np.array(rowlo),np.array(rowhi),marginal,cell_fn=cell_fn)
    # Same public call as the MC wrapper, on real images. LP work is cached per cell.
    source=source_rep.represent(image_tensor(w['test']['images']).to(DEVICE))
    dappr_alpha=nn.functional.softplus(w['dappr_logits'])+1
    dappr_eu=(w['K']/dappr_alpha.sum(-1)).numpy()
    mean_p=draws.mean(1)
    entropy=lambda p:-(p*p.clamp_min(1e-30).log()).sum(-1)
    mi=(entropy(mean_p)-entropy(draws).mean(1)).numpy()
    scores_by_method={'dropout_probability_width':quantify(mc).epistemic.numpy(),
        'source_probability_width':quantify(source).epistemic.cpu().numpy(),
        'dropout_mutual_information':mi,'dappr_original_eu':dappr_eu,
        'negative_max_softmax':-w['ce_logits'].softmax(-1).max(-1).values.numpy(),
        'candidate_membership_lower':source.candidate_lower.cpu().numpy(),
        'candidate_membership_upper':source.candidate_upper.cpu().numpy()}
    isood=w['test']['is_ood']
    frame=pd.DataFrame(dict(dataset=name,test_position=np.arange(len(isood)),original_label=w['test']['label'],is_ood=isood,cell=test_cells,**scores_by_method))
    score_frames.append(frame)
    for method,values in scores_by_method.items():
        valid=np.isfinite(values)
        entry=dict(dataset=name,method=method,n_id=int((valid&~isood).sum()),n_ood=int((valid&isood).sum()),rejected=int((~valid).sum()))
        if (valid&~isood).any() and (valid&isood).any():entry.update(evaluate_probability_ood(values[valid&~isood],values[valid&isood]))
        metric_rows.append(entry)
    for method,out in [('dropout_envelope',mc),('source_mixture',source)]:
        lo=out.lower.cpu().numpy();hi=out.upper.cpu().numpy()
        for i in range(len(isood)):
            for k,label in enumerate(w['id_classes']):
                bound_frames.append(dict(dataset=name,method=method,test_position=i,is_ood=bool(isood[i]),original_label=int(w['test']['label'][i]),event_class=int(label),lower=lo[i,k],upper=hi[i,k],width=hi[i,k]-lo[i,k],status='finite_sample_envelope' if method=='dropout_envelope' else source.status[i][k]))
    w.update(mc=mc,source=source,source_rep=source_rep,scores=frame,draws=draws)
    np.savez_compressed(RUN_DIR/f'{name}_representations.npz',dropout_samples=draws.numpy(),
        mc_lower=mc.lower.numpy(),mc_upper=mc.upper.numpy(),source_lower=source.lower.cpu().numpy(),source_upper=source.upper.cpu().numpy(),
        candidate_lower=source.candidate_lower.cpu().numpy(),candidate_upper=source.candidate_upper.cpu().numpy(),
        row_lower=np.array(rowlo),row_upper=np.array(rowhi),marginal_lower=marginal[0],marginal_upper=marginal[1],
        reference_cells=ref_cells,test_cells=test_cells,test_counts=counts,ce_logits=w['ce_logits'].numpy(),dappr_logits=w['dappr_logits'].numpy())
    print(name,'source invalid rows',int((~source.valid).sum()),flush=True)
metrics=pd.DataFrame(metric_rows);scores=pd.concat(score_frames,ignore_index=True);bounds=pd.DataFrame(bound_frames)
metrics.to_csv(RUN_DIR/'ood_metrics.csv',index=False);scores.to_csv(RUN_DIR/'scores.csv',index=False)
bounds.to_csv(RUN_DIR/'probability_bounds.csv',index=False)
pd.DataFrame(source_law_rows).to_csv(RUN_DIR/'reference_laws.csv',index=False)
display(metrics.round(4))
''')
md('''### 5. Compare ID/OOD separation without conflating scores

Higher scores are treated as more OOD. The principal comparison is maximum class-probability width for both interval representations. Additional rows report ordinary dropout mutual information, DAPPr's original scalar EU, and negative max-softmax. Candidate-membership lower/upper values are separate detection statistics, not epistemic magnitudes.

Report tie-correct AUROC, OOD-positive AUPR, DAPPr's ID-positive AUPR convention, and FPR at 95% OOD recall. AUPR depends on the sampled prevalence, which is saved in counts. Probly's raw AUROC is retained alongside the corrected value because the installed implementation mishandles ties. No package files are modified.

ID accuracy uses deterministic class predictions and is reported separately. The width does not define a class prediction. These are one-seed pilot results with an explicitly stronger information channel for the source method; neither architecture matching nor common test images removes that distinction.''')
code('''# Goal: visualize the representation comparison and report the distinct information assumptions.
main_methods=['dropout_probability_width','source_probability_width','dropout_mutual_information','dappr_original_eu']
fig,axes=plt.subplots(1,4,figsize=(17,4),squeeze=False)
for ax,name in zip(axes[0],DATASETS):
    f=metrics[(metrics.dataset==name)&metrics.method.isin(main_methods)].set_index('method').reindex(main_methods)
    ax.bar(range(4),f.auroc,color=['#2171b5','#238b45','#888888','#c65d21'])
    ax.axhline(.5,color='black',ls=':');ax.set(ylim=(0,1),title=name,ylabel='OOD AUROC (ties handled)',xticks=range(4),xticklabels=['MC width','Source width','MC MI','DAPPr EU'])
    ax.tick_params(axis='x',rotation=25)
fig.suptitle('Source method has candidate-reference and unlabeled deployment access')
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(RUN_DIR/f'ood_comparison.{ext}',dpi=160,bbox_inches='tight')
plt.show()
summary=bounds.groupby(['dataset','method','is_ood'],as_index=False).agg(mean_width=('width','mean'),mean_lower=('lower','mean'),mean_upper=('upper','mean'),
    undefined_policy_rows=('status',lambda x:int((x=='undefined_denominator_policy').sum())))
summary.to_csv(RUN_DIR/'bound_summary.csv',index=False)
display(summary)
''')
md('''### 6. Inspect image-level class-probability intervals

For each dataset choose the first two ID and first two OOD test rows; selection does not depend on uncertainty. For each image show the interval for the CE classifier's predicted known class under both representations. The image's true class is displayed for interpretation, but was not supplied to either representer.

A narrow dropout interval on OOD can occur when sampled predictors agree on an inappropriate known class. A wide source interval can reflect non-identification, sparse reference information, or the undefined-denominator policy. The plotted probability endpoints are not upper/lower entropy values. Bounds for a single class do not describe the entire joint probability set.''')
code('''# Goal: show the actual images and the lower/upper probabilities returned by represent().
fig,axes=plt.subplots(4,4,figsize=(13,12),squeeze=False)
example_rows=[]
for row,name in enumerate(DATASETS):
    w=worlds[name];ood=w['test']['is_ood'];chosen=np.r_[np.flatnonzero(~ood)[:2],np.flatnonzero(ood)[:2]]
    for ax,i in zip(axes[row],chosen):
        k=int(w['ce_logits'][i].argmax());label=int(w['id_classes'][k])
        ml,mu=float(w['mc'].lower[i,k]),float(w['mc'].upper[i,k])
        sl,su=float(w['source'].lower[i,k]),float(w['source'].upper[i,k])
        ax.imshow(w['test']['images'][i],cmap='gray',vmin=0,vmax=255);ax.axis('off')
        ax.set_title(f"{name} {'OOD' if ood[i] else 'ID'} | true={w['test']['label'][i]}\\nP(class {label}): MC [{ml:.2f},{mu:.2f}]\\nSource [{sl:.2f},{su:.2f}]",fontsize=9)
        example_rows.append(dict(dataset=name,test_position=int(i),event_class=label,mc_lower=ml,mc_upper=mu,source_lower=sl,source_upper=su))
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(RUN_DIR/f'image_probability_intervals.{ext}',dpi=160,bbox_inches='tight')
plt.show()
pd.DataFrame(example_rows).to_csv(RUN_DIR/'image_examples.csv',index=False)
''')
md('''### 7. Distinguish sampling effort from information acquisition

On the same saved dropout draws, compare prefixes of 2, 5, 10 and 20 members. Adding members enlarges their convex hull, so exact singleton lower endpoints cannot increase and upper endpoints cannot decrease. This sweep changes the represented hypothesis family; it does not acquire labels or retrain the network.

For source-mixture bounds, the previous acquisition notebook changes observations while retaining the target definition. Those are different operations. The current wrapper exposes the source-compatible set through Probly, but it does not make a finite dropout hull satisfy identification or acquisition guarantees.''')
code('''# Goal: verify and display finite-hull sensitivity without interpreting it as learning.
sensitivity=[]
for name,w in worlds.items():
    previous=None
    for t in [2,5,10,MC_PASSES]:
        out=envelope_from_samples(w['draws'][:,:t])
        if previous is not None:
            assert torch.all(out.lower<=previous.lower+1e-7)
            assert torch.all(out.upper>=previous.upper-1e-7)
        for ood in [False,True]:
            mask=w['test']['is_ood']==ood
            sensitivity.append(dict(dataset=name,is_ood=ood,draws=t,mean_max_width=float(quantify(out).epistemic[mask].mean())))
        previous=out
sensitivity=pd.DataFrame(sensitivity);sensitivity.to_csv(RUN_DIR/'mc_draw_sensitivity.csv',index=False)
fig,axes=plt.subplots(1,4,figsize=(16,3.5),squeeze=False)
for ax,name in zip(axes[0],DATASETS):
    for ood in [False,True]:
        f=sensitivity[(sensitivity.dataset==name)&(sensitivity.is_ood==ood)]
        ax.plot(f.draws,f.mean_max_width,'o-',label='OOD' if ood else 'ID')
    ax.set(title=name,xlabel='Number of saved MC draws',ylabel='Mean maximum probability width');ax.legend()
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(RUN_DIR/f'mc_draw_sensitivity.{ext}',dpi=160,bbox_inches='tight')
plt.show()
''')
md('''### 8. Save the complete comparison and state what was implemented

The deliverable now contains actual Probly representations with `.lower`, `.upper`, per-class `.width`, and a registered `quantify(...).epistemic` summary. The dropout version wraps the sampler's `.represent`; the source version wraps a frozen model and observation-compatible set. A source representation additionally exposes candidate-membership endpoints and solver statuses.

The DAPPr baseline was trained using its objective, rather than treating DAPPr only as an organizational reference. This remains a small-CNN adaptation. No claim of improvement, formal coverage on real images, or equal-information comparison is made without support from the results. The candidate law is reference-defined; source-conditionals may differ across real train/test pools. Undefined-denominator policies and empty sets are reported explicitly.

The workbook saves checkpoints, source identities, all class endpoints, MC predictions, reference/marginal bounds, baseline logits, metrics, plots, source references, implementation snapshots and dependency versions. The executed notebook is saved by the runner. The prior acquisition notebook remains unchanged.''')
code('''# Goal: save result-dependent conclusions, source references, and a reproducible implementation snapshot.
report=['# Probly probability-bound comparison','',
    'Two actual probability-bound representations, plus a DAPPr-loss baseline; one-seed small-CNN pilot.',
    'Source inference has candidate-reference access and uses the unlabeled test histogram. This is not an equal-information comparison.','',
    '| Dataset | MC width AUROC | Source width AUROC | DAPPr EU AUROC |',
    '|---|---:|---:|---:|']
for name in DATASETS:
    f=metrics[metrics.dataset==name].set_index('method')
    report.append(f"| {name} | {f.loc['dropout_probability_width','auroc']:.4f} | {f.loc['source_probability_width','auroc']:.4f} | {f.loc['dappr_original_eu','auroc']:.4f} |")
report += ['',f"Maximum Probly raw vs tie-correct AUROC discrepancy: {metrics.probly_auroc_difference.abs().max():.6f}.",
    f"Source class-event rows using undefined-denominator [0,1] policy: {int((bounds.status=='undefined_denominator_policy').sum())}.",
    'Probability endpoint vectors need not sum to one. Their width is imprecision, not MI. Candidate membership is a separate event.',
    'No true conditional probabilities are observed for real images; empirical interval coverage is not claimed.',
    'Read SOURCES.md and the executed notebook for the full assumptions and derivations.']
report_text='\\n'.join(report);(RUN_DIR/'RESULTS.md').write_text(report_text);display(Markdown(report_text))
config=dict(seed=SEED,datasets=DATASETS,epochs=EPOCHS,train_per_class=TRAIN_PER_CLASS,val_per_class=VAL_PER_CLASS,test_per_class=TEST_PER_CLASS,
    reference_per_class_cap=REFERENCE_PER_CLASS,dropout=DROPOUT,mc_passes=MC_PASSES,alpha=ALPHA,cell_confidence_threshold=CELL_CONFIDENCE_THRESHOLD,
    candidate='equal OOD-class-weight reference histogram, treated as specified',
    source_target='known-class membership conditional on known membership',quantification='maximum coordinate probability width',
    architecture=str(worlds['mnist']['models']['ce']),device=str(DEVICE),
    versions={p:importlib.metadata.version(p) for p in ['torch','probly','numpy','scipy','pandas','scikit-learn']})
(RUN_DIR/'config.json').write_text(json.dumps(config,indent=2))
for file in ['probability_bounds.py','source_acquisition.py']:
    shutil.copy2(ROOT/'real'/file,RUN_DIR/file)
notebook=json.loads((ROOT/'probly_probability_bounds_four_datasets.ipynb').read_text())
for cell in notebook['cells']:
    if cell['cell_type']=='code':cell['outputs']=[];cell['execution_count']=None
(RUN_DIR/'source_notebook.ipynb').write_text(json.dumps(notebook,indent=1))
(RUN_DIR/'SOURCES.md').write_text(''.join(notebook['cells'][1]['source']))
# Preserve the exact installed API implementations audited here, without modifying the package.
import probly.representer._representer as rep_module
import probly.evaluation.ood as ood_module
import probly.metrics.array as metric_module
for label,module in [('probly_representer',rep_module),('probly_ood',ood_module),('probly_numpy_metrics',metric_module)]:
    (RUN_DIR/f'{label}_snapshot.py').write_text(inspect.getsource(module))
inputs=[ROOT/'real/probability_bounds.py',ROOT/'real/source_acquisition.py']+sorted(CACHE.glob('*.npz'))
manifest=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs]
(RUN_DIR/'input_manifest.json').write_text(json.dumps(manifest,indent=2))
(RUN_DIR/'README.md').write_text('# Probability bounds comparison\\n\\nStart with executed.ipynb, RESULTS.md and SOURCES.md.\\n\\n'
    'ood_metrics.csv contains tie-correct metrics and unmodified Probly AUROC. probability_bounds.csv retains per-image class endpoints. '
    'Each dataset NPZ stores the sampled probabilities, source constraints and baseline logits; PT files are selected checkpoints. '
    'assignments.csv proves pool separation. source_notebook.ipynb has clean source cells.\\n')
print('Saved:',RUN_DIR)
''')
for c in cells:
    if c.cell_type=='markdown':c.source=c.source.replace(chr(92)*2,chr(92))
n=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'reach-aistats','display_name':'Python (reach-aistats)','language':'python'},'language_info':{'name':'python','version':'3.13'}})
nb.write(n,ROOT/'probly_probability_bounds_four_datasets.ipynb')
print(ROOT/'probly_probability_bounds_four_datasets.ipynb')
