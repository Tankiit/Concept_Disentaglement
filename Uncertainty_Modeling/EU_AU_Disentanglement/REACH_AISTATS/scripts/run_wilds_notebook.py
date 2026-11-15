"""Execute one dataset notebook. System Python supplies nbclient; kernel is scientific env."""
from pathlib import Path
import argparse
import os
import json
import tempfile
import hashlib
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

def main():
    p=argparse.ArgumentParser()
    p.add_argument('dataset',choices=['camelyon17','fmow'])
    p.add_argument('kind',choices=['acquisition','counterexample'])
    p.add_argument('--data-root',type=Path)
    p.add_argument('--weights',type=Path)
    p.add_argument('--python',default='/home/mtleao/.conda/envs/reach-aistats/bin/python')
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    name=f'{a.dataset}_{a.kind}';path=root/f'{name}.ipynb'
    notebook=nbformat.read(path,as_version=4)
    prior=set((root/'workbook').glob(name+'_*'))
    env=dict(MPLCONFIGDIR='/tmp/wilds-mpl',MPLBACKEND='module://matplotlib_inline.backend_inline',OMP_NUM_THREADS='4')
    if a.data_root:env['WILDS_DATA_ROOT']=str(a.data_root.expanduser().resolve())
    if a.weights:env['WILDS_ENCODER_WEIGHTS']=str(a.weights.expanduser().resolve())
    status='failed'
    with tempfile.TemporaryDirectory(prefix='wilds-kernel-') as temp:
        folder=Path(temp)/'reach-aistats';folder.mkdir()
        (folder/'kernel.json').write_text(json.dumps(dict(argv=[a.python,'-m','ipykernel_launcher','-f','{connection_file}'],display_name='reach-aistats',language='python',env=env)))
        manager=KernelManager(kernel_name='reach-aistats',kernel_spec_manager=KernelSpecManager(kernel_dirs=[temp]))
        def progress(cell,cell_index,**kwargs):print(f'{name}: cell {cell_index+1}/{len(notebook.cells)}',flush=True)
        client=NotebookClient(notebook,km=manager,timeout=3600,on_cell_start=progress)
        try:
            client.execute(cwd=str(root));status='completed'
        finally:
            created=set((root/'workbook').glob(name+'_*'))-prior
            for out in created:
                nbformat.write(notebook,out/'executed.ipynb');nbformat.write(notebook,path)
                (out/'execution_status.json').write_text(json.dumps(dict(status=status),indent=2))
                manifest=[dict(file=str(f.relative_to(out)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest())
                          for f in sorted(out.rglob('*')) if f.is_file() and f.name!='artifact_manifest.json']
                (out/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2))
                print('Saved:',out,flush=True)

if __name__=='__main__':main()
