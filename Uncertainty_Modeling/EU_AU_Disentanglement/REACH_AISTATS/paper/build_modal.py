"""Compile the paper on Modal (no local TeX): modal run paper/build_modal.py"""
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
app = modal.App('reach-aistats-paper')
image = (modal.Image.debian_slim()
         .apt_install('texlive-latex-recommended', 'texlive-latex-extra', 'texlive-fonts-recommended',
                      'texlive-bibtex-extra', 'latexmk')
         .add_local_dir(HERE, '/paper', ignore=['main.pdf', 'build_modal.py', '*.bak']))


@app.function(image=image, timeout=600)
def build():
    import subprocess
    r = subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', 'main.tex'], cwd='/paper',
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    log = open('/paper/main.log', encoding='utf-8', errors='replace').read()
    lines = log.splitlines()
    errors = [' | '.join(lines[i:i+3]) for i, l in enumerate(lines) if l.startswith('!')]
    warnings = errors + [l for l in lines if ('Warning' in l and 'undefined' not in l) or 'Overfull' in l]
    import os
    if os.path.exists('/paper/main.blg'):
        warnings += ['BLG: '+l for l in open('/paper/main.blg', encoding='utf-8', errors='replace').read().splitlines()
                     if 'error' in l.lower() or 'warning' in l.lower() or "I couldn't" in l]
    pdf = open('/paper/main.pdf', 'rb').read() if os.path.exists('/paper/main.pdf') else b''
    return r.returncode, warnings, pdf


@app.local_entrypoint()
def main():
    code, warnings, pdf = build.remote()
    print('exit', code); print('\n'.join(warnings[:40]))
    if pdf:
        (HERE/'main.pdf').write_bytes(pdf); print('wrote', HERE/'main.pdf')
