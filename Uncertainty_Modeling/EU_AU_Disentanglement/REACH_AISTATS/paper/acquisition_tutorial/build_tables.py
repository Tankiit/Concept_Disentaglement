#!/usr/bin/env python3
"""Extract tutorial tables and audit their saved-run inputs (standard library only)."""
from pathlib import Path
from collections import defaultdict, Counter
import csv, hashlib, json, math
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RUNS = [
 'source_acquisition_20261029T151111775438Z',
 'acquisition_counterexample_20261029T192454214626Z',
 'camelyon17_acquisition_20261030T092023425816Z',
 'camelyon17_counterexample_20261030T092024988890Z',
 'fmow_acquisition_20261030T094932286096Z',
 'fmow_counterexample_20261030T094929883846Z']
NOTEBOOKS = ['source_acquisition_four_datasets','acquisition_counterexample','camelyon17_acquisition','camelyon17_counterexample','fmow_acquisition','fmow_counterexample']
used = {ROOT/'dappr_epistemic_sets.ipynb', ROOT/'workbook/dappr_epistemic_sets_seed0_20261029T085507170686Z/config.json'}
def read(i, name):
 p = ROOT/'workbook'/RUNS[i-1]/name; used.add(p)
 with p.open() as f: return list(csv.DictReader(f))
def esc(s):
 return str(s).replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('%',r'\%').replace('&',r'\&').replace('#',r'\#')
def num(s, digits=6):
 x=float(s)
 if not math.isfinite(x): return '--'
 if x and abs(x)<1e-5:
  a,b=f'{x:.5e}'.split('e');return f'${a}\\times10^{{{int(b)}}}$'
 return f'{x:.{digits}f}'
out=[]
def text(s):
 if s.startswith((r'\section{',r'\subsection{')):out.append(r'\reservetable')
 out.append(s+'\n')
def table(title, headers, rows, widths=None):
 text('\\subsection{'+esc(title)+'}')
 cols=''.join(f'p{{{w}\\textwidth}}' for w in widths) if widths else 'l'+'r'*(len(headers)-1)
 text('\\begingroup\\small\n\\setlength{\\tabcolsep}{4pt}\n\\begin{longtable}{'+cols+'}\n\\toprule')
 text(' & '.join(headers)+r'\\\midrule\endhead')
 for row in rows:text(' & '.join(str(x) for x in row)+r'\\')
 text('\\bottomrule\n\\end{longtable}\n\\endgroup')
text(r'\section{Four-dataset sources and geometry}')
roles=defaultdict(list)
for r in read(1,'source_roles.csv'):roles[r['dataset'],r['source']].append(r['original_class'])
table('Original class groups (recorded numeric labels)', ['Dataset','Source','Original labels'], [(esc(d),s,', '.join(ls)) for (d,s),ls in roles.items()],[.14,.09,.64])
table('Probe cells and affine geometry',['Dataset','Regime','Probe','Cells','Smallest singular value'],[(r['dataset'],esc(r['regime']),r['probe'],r['cells'],num(r['affine_difference_smallest_singular_value'],9)) for r in read(1,'geometry.csv')])
text(r'Each law below is listed in ascending cell order. Semicolons separate probabilities. Every row is normalized; displayed values are rounded. The candidate row in the confounded regime is the average of the two known rows.')
laws=defaultdict(list)
for r in read(1,'source_laws.csv'):laws[r['dataset'],r['regime'],r['source']].append((int(r['cell']),float(r['probability'])))
for d in ['mnist','emnist','fashion','blood']:
 rows=[]
 for (ds,reg,s),vals in laws.items():
  if ds==d:
   assert abs(sum(v for _,v in vals)-1)<1e-10
   rows.append((esc(reg),s,'; '.join(f'{v:.5f}' for _,v in sorted(vals))))
 table(d+' source-cell laws',['Regime','Source','Probabilities'],rows,[.20,.08,.60])
text(r'\section{Exact input-only membership intervals}')
table('Exact rows and exact deployment cell probabilities',['Dataset','Regime','Target','Truth','Lower','Upper','Width'],[(r['dataset'],esc(r['regime']),{'known_membership':'K','candidate_membership':'N'}[r['target']],*[num(r[k]) for k in ['truth','lower','upper','width']]) for r in read(1,'oracle_intervals.csv') if r['information']=='exact_inputs'])
text(r'\section{All saved-budget mean widths for the four-dataset experiment}')
text(r'K denotes known membership, Y the anchored label probability, and N candidate membership. All values average six repeats. Reference and target-label budgets are per source. The CSV also retains repeat ranges, mean endpoints, inclusion, and residuals.')
summ=read(1,'acquisition_summary.csv')
for d in ['mnist','emnist','fashion','blood']:
 for reg in ['image_derived','confounded']:
  pivot=defaultdict(dict)
  for r in summ:
   if (r['dataset'],r['regime'])==(d,reg):pivot[r['arm'],int(r['budget'])][r['target']]=r['mean_width']
  table(d+': '+reg,['Arm','Budget','K width','Y width','N width'],[(esc(a),b,*[num(v[t]) for t in ['known_membership','anchored_target','candidate_membership']]) for (a,b),v in sorted(pivot.items())])
table('Figure 6: repeat-zero candidate endpoints',['Dataset','Arm','Truth','Lower','Upper','Width'],[(r['dataset'],esc(r['display_label']),*[num(r[k]) for k in ['truth','lower','upper','width']]) for r in read(1,'plotted_endpoints.csv')])
text(r'\section{Exact counterexample: every saved budget}')
text(r'The informative response is identical for World M identities and World L source-0 labels because the saved informative draws are paired. The table shows World M identities. All uninformative arms preserve variance 0.09, MI 0.192744757, MI endpoints 0.147044215 and 0.192744757, confidence span 0.6, rejection zero, and inclusion one.')
toy=[r for r in read(2,'acquisition_summary.csv') if r['world']=='mixture_unknown' and r['action']=='source_id']
table('Exact and simulated informative-channel variance',['Budget','Exact variance','Simulated variance','Monte Carlo SE'],[(r['n'],*[num(r[k],9) for k in ['expected_variance','mean_variance','mc_standard_error']]) for r in toy])
table('Exact expected MI and endpoint values (nats)',['Budget','MI','Lower MI','Upper MI'],[(r['n'],*[num(r[k],9) for k in ['expected_eu','expected_eu_lower','expected_eu_upper']]) for r in toy])
table('Confidence-set illustration',['Budget','Conditional mean width','Rejection fraction','Inclusion fraction'],[(r['n'],*[num(r[k]) for k in ['conditional_mean_confidence_width','rejection_rate','inclusion']]) for r in toy])
for i,name in [(3,'CAMELYON17'),(5,'FMoW')]:
 text('\\section{'+name+' image pilot}')
 split=read(i,'split_counts.csv')
 table('Full audited partition (FMoW includes later years)',['Source name','Role','Records','Groups'],[(esc(r['source_name']),r['role'],r['records'],r['groups']) for r in split])
 table('Classification accuracy',['Method','Internal source','Images','Accuracy'],[(r['method'],r['source'],r['n'],num(r['accuracy'])) for r in read(i,'accuracy.csv')])
 table('Detection: OOD-positive metrics',['Score','AUROC','OOD AP','FPR95'],[(esc(r['method']),*[num(r[k]) for k in ['auroc','aupr_ood','fpr95']]) for r in read(i,'ood_scores.csv')])
 table('Additional metric audit',['Score','Raw Probly AUROC','ID-oriented AP'],[(esc(r['method']),num(r['probly_raw_auroc']),num(r['dappr_aupr_id'])) for r in read(i,'ood_scores.csv')])
 pivot=defaultdict(dict)
 for r in read(i,'acquisition_summary.csv'):pivot[r['action'],int(r['budget'])][r['target']]=r
 table('All saved-budget mean widths',['Action','Budget','K width','Y width','N width'],[(esc(a),b,*[num(v[t]['mean_width']) for t in ['known_membership','reference_label','candidate_membership']]) for (a,b),v in sorted(pivot.items())])
 table('Maximum-budget denominator-policy fractions',['Action','Known','Label','Candidate'],[(esc(a),*[num(v[t]['undefined_fraction']) for t in ['known_membership','reference_label','candidate_membership']]) for (a,b),v in sorted(pivot.items()) if b==2048])
 law=defaultdict(dict)
 for r in read(i,'source_cell_law_audit.csv'):law[r['source'],r['role']][int(r['cell'])]=float(r['probability'])
 table('Cell-law transfer audit',['Source','Reference law','Evaluation law','TV'],[(s,', '.join(f'{law[s,"reference"][c]:.4f}' for c in range(4)),', '.join(f'{law[s,"evaluation"][c]:.4f}' for c in range(4)),num(.5*sum(abs(law[s,'reference'][c]-law[s,'evaluation'][c]) for c in range(4)))) for s in ['0','1','2']],[.08,.34,.34,.09])
for i,name in [(4,'CAMELYON17'),(6,'FMoW')]:
 text('\\section{'+name+' record counterexample}')
 table('Eligible fit and held-out populations',['Population','Internal source','Records','Positive rate'],[(r['split'],r['source'],r['n'],num(r['target_rate'],9)) for r in read(i,'population_rates.csv')])
 states=read(i,'selected_states.csv'); assert len(states)==2
 table('Selected state summaries',['State','Identity count','Labels/source','Mean','Variance','MI'],[(r['state_id'],r['n_identity'],r['n_labels_per_source'],*[num(r[k],9) for k in ['mean','variance','mi']]) for r in states])
 table('Posterior shapes ordered as W, Theta0, Theta1',['State','Shapes'],[(r['state_id'],esc(r['shapes'])) for r in states],[.1,.76])
 table('Model action values and quadrature check',['State','Identity value','Label value','Absolute MI change'],[(r['state_id'],num(r['value_identity'],10),num(r['value_label'],10),num(abs(float(r['mi'])-float(r['mi_order48'])),12)) for r in states])
 table('Exact held-out one-step errors',['State','Action','Initial squared error','Expected squared error'],[(r['state_id'],esc(r['action']),num(r['initial_squared_error'],12),num(r['expected_squared_error'],12)) for r in read(i,'heldout_exact_one_step.csv')])
 table('Initial credible enclosures',['State','Lower','Upper','Width'],[(r['state_id'],*[num(r[k],9) for k in ['lower','upper','width']]) for r in read(i,'target_credible_enclosures.csv')])
 table('MI cross-check (nats)',['State','Quadrature','Probly samples'],[(r['state_id'],num(r['quadrature'],12),num(r['probly'],12)) for r in read(i,'probly_check.csv')])
 for role in ['identity_preferred','label_preferred']:
  table(role+': all trajectory budgets',['Action','Budget','Mean variance','MSE','MSE Monte Carlo SE'],[(esc(r['action']),r['n'],*[num(r[k],9) for k in ['variance','mse','mc_se']]) for r in read(i,'acquisition_summary.csv') if r['role']==role])
(HERE/'numerical_tables.tex').write_text('\n'.join(out))
# Assertions about saved artifacts: audit extraction without rerunning experiments.
audit={}
for i,total,undefined in [(1,2880,348),(3,1200,366),(5,1200,140)]:
 rows=read(i,'acquisition_intervals.csv'); counts=Counter(r['status'] for r in rows)
 assert len(rows)==total and counts['undefined_denominator_policy']==undefined and counts['empty']==0
 if i==1: assert all(r['included']=='True' for r in rows)
 audit[f'run{i}']=dict(intervals=len(rows),statuses=dict(counts),max_residual=max(float(r['residual']) for r in rows))
for i,confirmed in [(4,True),(6,False)]:
 p=ROOT/'workbook'/RUNS[i-1]/'result.json';used.add(p);result=json.loads(p.read_text())
 assert result['confirmed'] is confirmed
 exact=read(i,'heldout_exact_one_step.csv');by=defaultdict(dict)
 for r in exact:by[r['role']][r['action']]=float(r['expected_squared_error'])
 advantages=[by['identity_preferred']['reference_label']-by['identity_preferred']['source_id'],by['label_preferred']['source_id']-by['label_preferred']['reference_label']]
 assert all(x>0 for x in advantages)==confirmed
 assert math.isclose(advantages[0],result['identity_advantage'],abs_tol=1e-14)
 assert math.isclose(advantages[1],result['label_advantage'],abs_tol=1e-14)
 audit[f'run{i}']=result
# Independently recompute all searched-state closed-form moments/action values.
for i in (4,6):
 states=read(i,'candidate_states.csv')
 assert len(states)==600
 for r in states:
  shapes=json.loads(r['shapes']); totals=[a+b for a,b in shapes]
  m=[a/(a+b) for a,b in shapes];v=[x*(1-x)/(n+1) for x,n in zip(m,totals)]
  w,t0,t1=m;vw,v0,v1=v
  expected={'mean':(1-w)*t0+w*t1,
   'variance':((1-w)**2+vw)*v0+(w*w+vw)*v1+vw*(t1-t0)**2,
   'value_identity':(t1-t0)**2*vw/(totals[0]+1),
   'value_label':.5*((1-w)**2*v0/(totals[1]+1)+w*w*v1/(totals[2]+1))}
  for key,value in expected.items():assert math.isclose(float(r[key]),value,rel_tol=1e-10,abs_tol=1e-15),(i,r['state_id'],key)
audit['beta_states_recomputed']=1200
figs=json.loads((HERE/'figure_manifest.json').read_text());assert len(figs)==14
for f in figs:
 original=ROOT/f['source'];used.add(original)
 assert hashlib.sha256(original.read_bytes()).hexdigest()==f['sha256']
 assert hashlib.sha256((HERE/'figures'/f['figure']).read_bytes()).hexdigest()==f['sha256']
# Trace both present notebooks and archived implementation; no import of moving helpers.
prov=[]
for i,(run,nb) in enumerate(zip(RUNS,NOTEBOOKS),1):
 p=ROOT/'workbook'/run
 prov.extend(['\\section{Run '+str(i)+'}',r'Notebook: \path{'+nb+'.ipynb}.',r'Archived run: \path{workbook/'+run+'}.',r'Online notebook: \url{https://github.com/Tankiit/Concept_Disentaglement/blob/reach-mateus-workbook/Uncertainty_Modeling/EU_AU_Disentanglement/REACH_AISTATS/'+nb+'.ipynb}.'])
 used.add(ROOT/(nb+'.ipynb'))
 for f in p.iterdir():
  if f.suffix in ['.py','.json','.md'] or f.name in ['executed.ipynb','source_notebook.ipynb']:used.add(f)
 version=p/('config.json' if i in [1,2] else 'versions.json')
 data=json.loads(version.read_text());versions=data.get('versions',data)
 if i in [3,4,5,6] or 'numpy' in versions:
  prov.append('Recorded package versions: '+', '.join(esc(k)+' '+esc(v) for k,v in versions.items())+'.')
 prov.append('Primary numerical and code artifacts used here:')
 prov.append(r'\begin{itemize}')
 for f in sorted(used):
  if f.parent==p and f.suffix in ['.csv','.py']:prov.append(r'\item \path{'+f.name+'}')
 prov.append(r'\end{itemize}')
(HERE/'provenance.tex').write_text('\n\n'.join(prov))
manifest=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(used)]
(HERE/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
audit['figures_verified']=len(figs);audit['source_files_hashed']=len(manifest)
(HERE/'artifact_checks.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2))
