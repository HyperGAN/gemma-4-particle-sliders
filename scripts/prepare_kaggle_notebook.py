"""Embed a small, already-validated agent bundle in a CPU-only Kaggle notebook.

This prepares local files only. Kaggle publication/execution and competition
submission are separate explicit CLI/API operations with retained identifiers.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact',type=Path,required=True)
    p.add_argument('--owner',required=True)
    p.add_argument('--slug',required=True)
    p.add_argument('--title',required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('notebook preparation is immutable')
    archive=args.artifact/'submission.zip';payload=archive.read_bytes()
    if len(payload)>512*1024:raise ValueError('inline packaging is limited to small prompt-only baselines; use a dataset for adapters')
    receipt=json.loads((args.artifact/'build.json').read_text())
    sha=hashlib.sha256(payload).hexdigest()
    if receipt['sha256']!=sha or receipt['arm']!='off':raise ValueError('exact OFF baseline build required')
    with ZipFile(archive) as z:
        if 'agent.yaml' not in z.namelist() or any(i.is_dir() for i in z.infolist()):
            raise ValueError('agent.yaml at archive root and file entries only required')
    encoded=base64.b64encode(payload).decode()
    source=("from pathlib import Path\nimport base64\nimport hashlib\nimport os\n"
        "working = Path(os.environ.get('KAGGLE_WORKING_DIR', '/kaggle/working'))\n"
        "working.mkdir(parents=True, exist_ok=True)\n"
        f"payload = base64.b64decode({encoded!r}, validate=True)\n"
        f"assert hashlib.sha256(payload).hexdigest() == {sha!r}\n"
        "destination = working / 'submission.zip'\n"
        "destination.write_bytes(payload)\n"
        "print({'artifact': destination.name, 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()})\n")
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
        'cells':[{'cell_type':'markdown','id':'purpose','metadata':{},'source':
            '# Gemma 4 Particle Sliders: OFF control\n\n'
            'This notebook emits the exact prompt-only control bundle. It performs no model inference or training. '
            'The competition evaluates the uploaded agent separately. No learned Particle Sliders result is claimed.\n\n'
            'Source and experiment records: https://github.com/HyperGAN/gemma-4-particle-sliders\n'},
            {'cell_type':'code','id':'emit-bundle','metadata':{},'source':source,'execution_count':None,'outputs':[]}]}
    metadata={'id':args.owner+'/'+args.slug,'title':args.title,'code_file':'baseline.ipynb',
        'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':False,'enable_tpu':False,
        'enable_internet':False,'competition_sources':['gemma-4-developer-agent'],
        'dataset_sources':[],'kernel_sources':[],'model_sources':[]}
    args.out.mkdir(parents=True)
    (args.out/'baseline.ipynb').write_text(json.dumps(notebook,indent=2)+'\n')
    (args.out/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (args.out/'preparation.json').write_text(json.dumps({'schema':'gemma4-kaggle-baseline-notebook-v1',
        'artifact_sha256':sha,'artifact_build':receipt,'gpu_enabled':False,'model_inference':False,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'notebook_sha256':hashlib.sha256((args.out/'baseline.ipynb').read_bytes()).hexdigest()},indent=2)+'\n')
    print(json.dumps({'prepared':str(args.out),'artifact_sha256':sha,'gpu_enabled':False}))


if __name__=='__main__':main()
