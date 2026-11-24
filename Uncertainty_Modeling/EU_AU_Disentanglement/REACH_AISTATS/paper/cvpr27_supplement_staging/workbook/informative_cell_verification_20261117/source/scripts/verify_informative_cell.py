"""Run from REACH_AISTATS: python scripts/verify_informative_cell.py.

Runs independent tests, then creates the supplemental table, diagram, and
provenance manifest. Standard library + matplotlib; no data download/training.
"""
from pathlib import Path
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/reach-matplotlib')
from real.informative_cell_counterexample import (
    P, WORLDS, ACTIONS, BUDGETS, predictive_joint, rates, risk, value,
)


def main():
    # Locally generated exact n=200 fractions exceed Python's default decimal
    # serialization limit. No externally supplied integers are parsed here.
    sys.set_int_max_str_digits(100000)
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'),
                                                pattern='test_informative_cell_counterexample.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = ROOT/'workbook/informative_cell_verification_20261117'
    out.mkdir(exist_ok=True)
    def table(name, fields, rows):
        with (out/name).open('w', newline='') as f:
            w = csv.writer(f); w.writerow(fields); w.writerows(rows)
    table('one_step.csv', ['world','action','rate_low','rate_high','remaining_variance','value','value_exact'],
          [(w,a,*map(str,rates(w,a)),float(risk(rates(w,a),1)),float(value(w,a)),str(value(w,a)))
           for w in WORLDS for a in ACTIONS])
    table('joint_laws.csv', ['world','p','c','y','probability_exact'],
          [(w,str(p),c,y,str(v)) for w in WORLDS for p in P
           for (c,y),v in predictive_joint(w,p).items()])
    rows = [(n,risk(rates('M','identity'),n),risk(rates('M','cell'),n)) for n in BUDGETS]
    table('multi_budget.csv', ['n','direct_risk','cell_risk','direct_exact','cell_exact'],
          [(n,float(d),float(c),str(d),str(c)) for n,d,c in rows])
    (out/'table3.tex').write_text(
        '\\begin{tabular}{rrr}\n\\hline\n$n$ & Direct field & Visual cell \\\\\n\\hline\n' +
        ''.join(f'{n} & {float(d):.9g} & {float(c):.9g} \\\\\n' for n,d,c in rows) +
        '\\hline\n\\end{tabular}\n')
    fig, ax = plt.subplots(figsize=(7,2.6)); ax.axis('off')
    ax.text(.5,.98,'Identical predictive beliefs over (cell C, label Y)',ha='center',va='top',weight='bold')
    ax.text(.5,.85,'p = 0.2 or 0.8, equally likely; P(C = Y | Y) = 3/4',ha='center',fontsize=10)
    for x,w,title,law in [(.25,'M','World M: unknown source mixture','S | p ~ Bern(p); Y = S'),
                          (.75,'L','World L: unknown event rates','S ~ Bern(1/2); Y | S,p ~ Bern(p)')]:
        ax.text(x,.69,title,ha='center',weight='bold',fontsize=10)
        ax.text(x,.58,law,ha='center',fontsize=10)
        for y,a,label in zip((.43,.32,.21),ACTIONS,('Source identity','Source-0 label','Visual cell')):
            ax.text(x,y,f'{label}: {float(value(w,a)):.4f}',ha='center',fontsize=10)
    ax.text(.5,.04,'Exact target-variance reductions; equal-cost fields; minimax regret = 0.0162',
            ha='center',fontsize=9)
    fig.tight_layout()
    for ext in ('pdf','png'):
        fig.savefig(out/f'supp_exact_worlds.{ext}',dpi=200,
                    metadata={'Creator':'verify_informative_cell.py'})
    plt.close(fig)
    sources = ['real/informative_cell_counterexample.py',
               'tests/test_informative_cell_counterexample.py','scripts/verify_informative_cell.py']
    for source in sources:
        dst = out/'source'/source; dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/source,dst)
    (out/'README.md').write_text(
        '# Informative-cell verification\n\n'
        'From REACH_AISTATS run `python scripts/verify_informative_cell.py`.\n\n'
        'This separately archived finite-model calculation verifies PDF Eqs. (3)–(7), '
        'Appendices 8.2–8.5 and Table 3. It does not replace or modify the earlier '
        'constant-input experiment. All probabilities and risks use exact rational '
        'arithmetic; displayed decimals are rounded. Ordered-sequence enumeration '
        'independently checks count enumeration for budgets 0–6 at three priors. '
        'The affine regret identity is checked on all policy-simplex vertices, '
        'with the attaining equal-identity/label policy checked separately.\n\n'
        'Figure provenance: supp_exact_worlds.pdf/png is generated directly from '
        'the verified one-step values by scripts/verify_informative_cell.py. '
        'table3.tex is generated from the rational multi-budget risks. '
        'PDF bytes can differ between runs due to creation metadata.\n')
    manifest = {'python':platform.python_version(),'matplotlib':matplotlib.__version__,
                'command':'python scripts/verify_informative_cell.py','tests_passed':result.testsRun,
                'model':'informative finite cell; not an image-training experiment',
                'artifacts':{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(out)


if __name__ == '__main__':
    main()
