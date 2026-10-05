"""Run the CPU-bound real audit on Modal, next to the feature volume (no feature download).

    modal run real/modal_audit.py --arms corruption --grouping super4
    modal volume get reach-aistats results/audit_super4_corruption results/

Same code as `python main.py audit`; this file only ships the source tree and points
cache/features at the volume.
"""
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App('reach-aistats-audit')
volume = modal.Volume.from_name('reach-aistats')
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('numpy<2.3', 'scipy', 'pandas', 'matplotlib', 'cvxpy', 'clarabel')
         .add_local_dir(ROOT, '/root/reach', ignore=['cache', 'results', 'paper', 'lean', '**/__pycache__']))


@app.function(image=image, volumes={'/vol': volume}, cpu=32, memory=32768, timeout=4*3600)
def audit(args: list[str]):
    import os, subprocess
    os.makedirs('/root/reach/cache', exist_ok=True)
    if not os.path.exists('/root/reach/cache/features'):
        os.symlink('/vol/features', '/root/reach/cache/features')
    subprocess.run(['python', 'main.py', 'audit', *args], cwd='/root/reach', check=True)
    volume.commit()


@app.function(image=image, volumes={'/vol': volume}, timeout=1800)
def resummarise(out: str):
    """Recompute every coverage table from saved rows after a summary-code fix."""
    import json, sys
    sys.path.insert(0, '/root/reach')
    import pandas as pd
    from real.audit import summarise_coverage
    path = f'{out}/summary.json'
    summary = summarise_coverage(pd.read_csv(f'{out}/rows.csv.gz'), json.load(open(path)))
    open(path, 'w').write(json.dumps(summary, indent=2, default=float))
    volume.commit()


@app.function(image=image, volumes={'/vol': volume}, cpu=32, memory=32768, timeout=4*3600)
def blindspot_remote(out: str, module: str = 'real.blindspot'):
    import os, subprocess
    os.makedirs('/root/reach/cache', exist_ok=True)
    if not os.path.exists('/root/reach/cache/features'):
        os.symlink('/vol/features', '/root/reach/cache/features')
    subprocess.run(['python', '-m', module, '--out', out], cwd='/root/reach', check=True)
    volume.commit()


@app.local_entrypoint()
def blindspot():
    blindspot_remote.remote('/vol/results/blindspot')


@app.function(image=image, volumes={'/vol': volume}, cpu=32, memory=32768, timeout=4*3600)
def exchange_remote(backbone: str, out: str):
    import os, subprocess
    os.makedirs('/root/reach/cache', exist_ok=True)
    if not os.path.exists('/root/reach/cache/features'):
        os.symlink('/vol/features', '/root/reach/cache/features')
    subprocess.run(['python', '-m', 'real.exchange', '--setting', 'real', '--backbone', backbone, '--out', out],
                   cwd='/root/reach', check=True)
    volume.commit()


@app.local_entrypoint()
def exchange(backbone: str = 'resnet50'):
    exchange_remote.remote(backbone, f'/vol/results/exchange_{backbone}')


@app.local_entrypoint()
def blindspot_v2():
    blindspot_remote.remote('/vol/results/blindspot_v2', 'real.blindspot_v2')


@app.local_entrypoint()
def main(arms: str = 'corruption', grouping: str = 'super4', seeds: int = 50,
         corruption_seeds: int = 20, resummarise_only: bool = False, extra: str = '', name: str = ''):
    """extra: further `main.py audit` flags, space separated; name: results folder override."""
    out = f'/vol/results/{name}' if name else f'/vol/results/audit_{grouping}_{arms.replace(" ", "_")}'
    if resummarise_only:
        resummarise.remote(out)
        return
    audit.remote(['--arms', *arms.split(), '--grouping', grouping, '--seeds', str(seeds),
                  '--corruption-seeds', str(corruption_seeds), '--out', out, *extra.split()])


@app.function(image=image, volumes={'/vol': volume}, cpu=32, memory=65536, timeout=6*3600)
def bridge_remote(part: str):
    import os, subprocess
    os.makedirs('/root/reach/cache', exist_ok=True)
    if not os.path.exists('/root/reach/cache/features'):
        os.symlink('/vol/features', '/root/reach/cache/features')
    subprocess.run(['python', '-m', 'real.bridge', '--part', part, '--out', f'/vol/results/bridge/{part}'],
                   cwd='/root/reach', check=True)
    volume.commit()
    return part


@app.local_entrypoint()
def bridge(parts: str = 'H_c10_clip H_c10_dinov2 H_c10_resnet50 L_c10_clip L_c10_dinov2 L_c10_resnet50 '
                        'H_c100_clip H_c100_dinov2 L_c100_clip L_c100_dinov2 S_c10_clip'):
    for done in bridge_remote.map(parts.split(), return_exceptions=True):
        print('finished', done, flush=True)


@app.function(image=image, volumes={'/vol': volume}, cpu=32, memory=65536, timeout=8*3600)
def bridge2_remote(part: str):
    import os, subprocess
    os.makedirs('/root/reach/cache', exist_ok=True)
    if not os.path.exists('/root/reach/cache/features'):
        os.symlink('/vol/features', '/root/reach/cache/features')
    subprocess.run(['python', '-m', 'real.bridge2', '--part', part, '--out', f'/vol/results/bridge2/{part}'],
                   cwd='/root/reach', check=True)
    volume.commit()
    return part


@app.local_entrypoint()
def bridge2(parts: str = 'syn_K3_p1 syn_K3_p3 syn_K6_p1 syn_K6_p2 syn_K10_p1 H_c10_clip H_c10_dinov2 '
                         'H_c10_resnet50 L_c10_clip L_c10_dinov2 L_c10_resnet50 H_c100_clip H_c100_dinov2 '
                         'L_c100_clip L_c100_dinov2'):
    for done in bridge2_remote.map(parts.split(), return_exceptions=True, wrap_returned_exceptions=False):
        print('finished', done, flush=True)


@app.function(image=image, volumes={'/vol': volume}, cpu=32, memory=65536, timeout=8*3600)
def gauss_remote(arm: str, extra: str = '', name: str = ''):
    import os, subprocess
    os.makedirs('/root/reach/cache', exist_ok=True)
    if not os.path.exists('/root/reach/cache/features'):
        os.symlink('/vol/features', '/root/reach/cache/features')
    subprocess.run(['python', '-m', 'real.gauss', '--arm', arm, '--out', f'/vol/results/gauss/{name or arm}', *extra.split()],
                   cwd='/root/reach', check=True, env=dict(os.environ, REACH_GAUSS_DIR='/tmp/reach_gauss'))
    volume.commit()
    return arm


@app.local_entrypoint()
def gauss(arms: str = 'held_out shift'):
    for done in gauss_remote.map(arms.split(), return_exceptions=True, wrap_returned_exceptions=False):
        print('finished', done, flush=True)


@app.local_entrypoint()
def gauss_explore():
    """EXPLORATORY: shift inside the head's subspace, larger tau."""
    print(gauss_remote.remote('shift', '--shift-dir mean --taus 0 0.5 1 2 4 8', 'shift_explore_mean'))
