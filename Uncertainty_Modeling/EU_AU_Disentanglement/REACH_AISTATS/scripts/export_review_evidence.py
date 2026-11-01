"""Export existing notebook outputs for the retrospective. Never execute cells."""
from pathlib import Path
import base64
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'workbook/experiment_review_assets'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index=[]
    for name in ['eu_blindspot_mnist_emnist_fashion_blood','eu_blindspot_within_datasets']:
        source=ROOT/(name+'.ipynb')
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        notebook=json.loads(source.read_text())
        folder=OUT/name;folder.mkdir(exist_ok=True)
        for i,cell in enumerate(notebook['cells']):
            for j,output in enumerate(cell.get('outputs',[])):
                data=output.get('data',{})
                for mime,ext in [('image/png','png'),('text/html','html'),('text/plain','txt')]:
                    if mime not in data:continue
                    value=data[mime]
                    if isinstance(value,list):value=''.join(value)
                    target=folder/f'cell_{i:02d}_output_{j:02d}.{ext}'
                    if ext=='png':target.write_bytes(base64.b64decode(value))
                    else:target.write_text(value)
                    index.append(dict(notebook=source.name,notebook_sha256=digest,cell_index=i,
                        output_index=j,mime=mime,artifact=str(target.relative_to(OUT)),
                        provenance='exported existing saved notebook output; no execution or recomputation'))
                if 'text' in output:
                    target=folder/f'cell_{i:02d}_output_{j:02d}_stream.txt'
                    target.write_text(''.join(output['text']))
                    index.append(dict(notebook=source.name,notebook_sha256=digest,cell_index=i,
                        output_index=j,mime='text/plain (stream)',artifact=str(target.relative_to(OUT)),
                        provenance='exported existing saved notebook stream; no execution or recomputation'))
    with (OUT/'extracted_outputs_index.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(index[0]));writer.writeheader();writer.writerows(index)
    (OUT/'README.md').write_text('# Saved-output extracts\n\nThese artifacts were extracted from the two existing blind-spot notebooks, without executing any cell. '
        'Their export cells have SAVE_RESULTS=False and the named workbook directories are absent. '
        'Tables preserve rendered notebook precision and may be truncated; they are not reconstructed raw per-image data. '
        'Cell/output indices are zero-based. Source notebook hashes and exact output locations are in extracted_outputs_index.csv.\n')
    print(f'Exported {len(index)} existing outputs to {OUT}')

if __name__=='__main__':main()
