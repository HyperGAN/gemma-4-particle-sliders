"""Stage a measured learned agent dataset and an offline CPU emitter notebook.

No upload, notebook execution or competition submission occurs here. The
dataset slug contains the exact ZIP hash. A binary envelope preserves that ZIP
without relying on how an upload service handles archive file extensions.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
from zipfile import ZipFile

from gemma_particle.harness_identity import sha256_file
from gemma_particle.lora_serving import prepare_lora

MAGIC = b'GEMMA4_PARTICLE_AGENT_BUNDLE_V1\n'


def notebook_source(owner, slug, wrapper_sha, archive_sha):
    return ("from pathlib import Path\nimport hashlib\nimport os\nimport shutil\nfrom zipfile import ZipFile\n"
        "def sha(path):\n"
        "    with path.open('rb') as stream:\n"
        "        return hashlib.file_digest(stream, 'sha256').hexdigest()\n"
        "inputs = Path(os.environ.get('KAGGLE_INPUT_DIR', '/kaggle/input'))\n"
        f"candidates = [inputs / 'datasets' / {owner!r} / {slug!r} / 'agent_bundle.bin', inputs / {slug!r} / 'agent_bundle.bin']\n"
        "sources = {p.resolve() for p in candidates if p.is_file()}\n"
        "assert sources, 'Declared dataset bundle missing'\n"
        f"assert all(sha(p) == {wrapper_sha!r} for p in sources), 'Dataset bytes differ'\n"
        "source = sorted(sources)[0]\n"
        "working = Path(os.environ.get('KAGGLE_WORKING_DIR', '/kaggle/working'))\n"
        "working.mkdir(parents=True, exist_ok=True)\n"
        "temporary = working / '.submission.zip.tmp'\n"
        "with source.open('rb') as stream, temporary.open('xb') as destination:\n"
        f"    assert stream.read({len(MAGIC)}) == {MAGIC!r}, 'Bundle envelope differs'\n"
        "    shutil.copyfileobj(stream, destination)\n"
        f"assert sha(temporary) == {archive_sha!r}, 'Agent archive bytes differ'\n"
        "with ZipFile(temporary) as archive:\n"
        "    assert 'agent.yaml' in archive.namelist()\n"
        "    assert not any(info.is_dir() for info in archive.infolist())\n"
        "    assert archive.testzip() is None\n"
        "destination = working / 'submission.zip'\n"
        f"assert not destination.exists() or sha(destination) == {archive_sha!r}, 'Existing output differs'\n"
        "temporary.replace(destination)\n"
        "print({'artifact': destination.name, 'bytes': destination.stat().st_size, 'sha256': sha(destination)})\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--conversion-report', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path('configs/31b-ct-shared-positional-pilot.json'))
    p.add_argument('--owner', required=True)
    p.add_argument('--dataset-prefix', default='gemma4-particle')
    p.add_argument('--notebook-slug', required=True)
    p.add_argument('--title', default='Gemma 4 Particle Sliders candidate')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError('candidate preparation is immutable')
    for value in (args.owner, args.dataset_prefix, args.notebook_slug):
        if not re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*', value):
            raise ValueError('lowercase Kaggle owner/slug components required')
    if not 6<=len(args.title)<=50:
        raise ValueError('Kaggle title must have6 to50 characters')
    config = json.loads(args.config.read_text())
    receipt = json.loads((args.artifact/'build.json').read_text())
    archive_path = args.artifact/'submission.zip'
    if archive_path.is_symlink() or receipt.get('arm')!='distilled-particle' or sha256_file(archive_path)!=receipt['sha256']:
        raise ValueError('exact learned PEFT artifact required')
    adapter = args.artifact/'agent/adapters/particle_lora'
    spec = prepare_lora(config, adapter=adapter, conversion_report=args.conversion_report)
    native = json.loads((args.native/'native.json').read_text())
    cache = json.loads((args.cache/'index.json').read_text())
    identity = {'repo':config['model'], 'revision':config['model_revision'], 'dtype':'bfloat16',
        'attn_implementation':config['attn_implementation'], 'weight_format':config['frozen_quantization']}
    bindings = spec['provenance']['conversion_provenance']
    if (native.get('schema')!='gemma4-native-slider-v1' or type(native.get('completed_steps')) is not int
            or native['completed_steps']<=0 or native.get('weights')!='raw' or native.get('ema_decay') is not None
            or native['model_identity']!=identity or cache['metadata']['model_identity']!=identity
            or native['sha256']!=sha256_file(args.native/'native.safetensors')
            or cache['responses_sha256']!=sha256_file(args.cache/'responses.safetensors')
            or bindings['native_metadata_sha256']!=sha256_file(args.native/'native.json')
            or bindings['cache_index_sha256']!=sha256_file(args.cache/'index.json')):
        raise ValueError('learned native/cache bytes differ from measured conversion provenance')
    with ZipFile(archive_path) as archive:
        names = archive.namelist()
        if ('agent.yaml' not in names or len(set(names))!=len(names)
                or any(info.is_dir() for info in archive.infolist())
                or any(PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts for name in names)
                or set(names)!=set(receipt['files']) or archive.testzip() is not None
                or sum(info.file_size for info in archive.infolist())>=3*1024**3):
            raise ValueError('file-only root agent archive with exact listed files under3GiB required')
        for name in names:
            if hashlib.sha256(archive.read(name)).hexdigest()!=receipt['files'][name]:
                raise ValueError('archive member differs from its build receipt')
        for name, expected in spec['adapters']['particle_lora']['files'].items():
            if receipt['files'].get('adapters/particle_lora/'+name)!=expected:
                raise ValueError('packaged PEFT bytes differ from the measured conversion')
        if not {'LICENSE-code.txt','adapters/particle_lora/LICENSE-weights.txt','adapters/particle_lora/NOTICE.txt'}<=set(names):
            raise ValueError('code and upstream model notices required in learned bundle')
    slug = args.dataset_prefix+'-'+receipt['sha256'][:12]
    if not 6<=len(slug)<=50:
        raise ValueError('content-addressed dataset slug must have6 to50 characters')
    dataset, notebook = args.out/'dataset', args.out/'notebook'
    dataset.mkdir(parents=True); notebook.mkdir()
    wrapper = dataset/'agent_bundle.bin'
    with wrapper.open('xb') as destination, archive_path.open('rb') as source:
        destination.write(MAGIC); shutil.copyfileobj(source, destination)
    summary = {'schema':'gemma4-kaggle-candidate-payload-v1', 'artifact_sha256':receipt['sha256'],
        'wrapper_sha256':sha256_file(wrapper), 'adapter_identity':spec['adapters']['particle_lora'],
        'native_completed_steps':native['completed_steps'], 'native_metadata_sha256':bindings['native_metadata_sha256'],
        'cache_index_sha256':bindings['cache_index_sha256'], 'conversion_report_sha256':sha256_file(args.conversion_report),
        'export_strength':spec['provenance']['export_strength'], 'exact_merge':False,
        'official_score':None, 'scope':'local artifact preparation only; no uploaded dataset, executed notebook or official score'}
    (dataset/'candidate.json').write_text(json.dumps(summary,indent=2)+'\n')
    metadata = {'id':args.owner+'/'+slug,'title':args.title,'licenses':[{'name':'other'}],
        'description':'Gemma4-derived LoRA weights: Apache License2.0, as specified by the pinned Google model card. '
            'Agent and integration code: MIT. Both licenses and upstream notices are included inside the exact agent ZIP. '
            'https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct/blob/52f3f65bc7a02d555763bc923bd1d9094898219d/README.md '
            'https://github.com/HyperGAN/gemma-4-particle-sliders. Native routing is approximated by this PEFT conversion; '
            'preparation alone establishes no task accuracy or official score.',
        'resources':[{'path':'agent_bundle.bin','description':'Binary envelope containing exact SHA-bound agent ZIP'},
            {'path':'candidate.json','description':'Native/cache/conversion provenance hashes, without training episodes'}]}
    (dataset/'dataset-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    source = notebook_source(args.owner,slug,summary['wrapper_sha256'],receipt['sha256'])
    document = {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'}},
        'cells':[{'cell_type':'markdown','id':'purpose','metadata':{},'source':
            '# Gemma4 Particle Sliders candidate\n\nThis notebook emits a SHA-bound learned-adapter agent bundle from its declared dataset. '
            'It performs no model inference or training. The competition evaluates the agent separately. '
            'The nonlinear native-to-LoRA conversion is approximate; this notebook claims no official score.\n'},
            {'cell_type':'code','id':'emit-bundle','metadata':{},'source':source,'execution_count':None,'outputs':[]}]}
    (notebook/'candidate.ipynb').write_text(json.dumps(document,indent=2)+'\n')
    kernel = {'id':args.owner+'/'+args.notebook_slug,'title':args.title,'code_file':'candidate.ipynb',
        'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':False,'enable_tpu':False,
        'enable_internet':False,'competition_sources':['gemma-4-developer-agent'],
        'dataset_sources':[args.owner+'/'+slug],'kernel_sources':[],'model_sources':[]}
    (notebook/'kernel-metadata.json').write_text(json.dumps(kernel,indent=2)+'\n')
    summary.update(dataset=metadata['id'], notebook=kernel['id'],gpu_enabled=False,model_inference=False,
        source_sha256=sha256_file(Path(__file__)), notebook_sha256=sha256_file(notebook/'candidate.ipynb'))
    (args.out/'preparation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'prepared':str(args.out),'dataset':metadata['id'],'artifact_sha256':receipt['sha256'],'gpu_enabled':False}))


if __name__=='__main__':
    main()
