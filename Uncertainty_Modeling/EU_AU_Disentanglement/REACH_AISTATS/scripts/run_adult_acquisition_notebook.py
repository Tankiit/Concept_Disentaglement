"""Execute the notebook with the existing reach-aistats interpreter; save outputs.

Run using a Python with nbformat/nbclient (the system Python here). The kernel
uses reach-aistats; a temporary kernelspec avoids altering the user's settings.
"""
from pathlib import Path
import json, tempfile, hashlib
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'adult_acquisition_counterexample.ipynb'
notebook=nbformat.read(path,as_version=4)
prior=set((ROOT/'workbook').glob('adult_acquisition_*'))
with tempfile.TemporaryDirectory(prefix='source-acquisition-kernel-') as temp:
    folder=Path(temp)/'reach-aistats'; folder.mkdir()
    (folder/'kernel.json').write_text(json.dumps(dict(
        argv=['/home/mtleao/.conda/envs/reach-aistats/bin/python','-m','ipykernel_launcher','-f','{connection_file}'],
        display_name='Python (reach-aistats)',language='python',
        env={'MPLCONFIGDIR':'/tmp/reach-mpl','XDG_CACHE_HOME':'/tmp/reach-cache','MPLBACKEND':'module://matplotlib_inline.backend_inline','OMP_NUM_THREADS':'2'})))
    manager=KernelManager(kernel_name='reach-aistats',
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[temp]))
    def progress(cell,cell_index,**kwargs):
        print(f'Executing cell {cell_index+1}/{len(notebook.cells)}',flush=True)
    client=NotebookClient(notebook,km=manager,timeout=600,resources={'metadata':{'path':str(ROOT)}},
                          on_cell_start=progress)
    try:
        client.execute(cwd=str(ROOT))
    finally:
        created=set((ROOT/'workbook').glob('adult_acquisition_*'))-prior
        if len(created)==1:
            out=created.pop()
            nbformat.write(notebook,out/'executed.ipynb')
            # Keep outputs in the main notebook too, so opening it shows the completed example.
            nbformat.write(notebook,path)
            manifest=[dict(file=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                      for p in sorted(out.iterdir()) if p.is_file() and p.name!='artifact_manifest.json']
            (out/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2))
            print('Saved executed notebook:',out/'executed.ipynb',flush=True)
