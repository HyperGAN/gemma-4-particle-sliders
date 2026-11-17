"""Select mirrored wheels compatible with a snapshot's declared core constraints.

This is a declared local environment repair. It does not modify repository code,
tests, the scorer, or original wheel files. Test-only dependency additions stay
in the explicitly named Docker image.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tomllib

from packaging.requirements import Requirement
from packaging.utils import parse_wheel_filename,canonicalize_name

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--snapshot',type=Path,required=True)
p.add_argument('--wheels',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
args=p.parse_args()
if args.out.exists(): raise FileExistsError('wheel cohort is immutable')
with tarfile.open(args.snapshot,'r:gz') as archive:
    info=next(i for i in archive.getmembers() if i.name.removeprefix('./')=='pyproject.toml')
    project=tomllib.loads(archive.extractfile(info).read().decode())['project']
requirements=[Requirement(r) for r in project['dependencies']]
constraints={canonicalize_name(r.name):r for r in requirements}
own_name=canonicalize_name(project['name'])
selected=[]
for wheel in sorted(args.wheels.glob('*.whl')):
    name,version,build,tags=parse_wheel_filename(wheel.name)
    if name==own_name: continue
    if name in constraints and version not in constraints[name].specifier: continue
    selected.append(wheel)
if any(not any(parse_wheel_filename(w.name)[0]==name for w in selected) for name in constraints):
    missing=[name for name in constraints if not any(parse_wheel_filename(w.name)[0]==name for w in selected)]
    raise ValueError('mirrored core dependencies unavailable: '+','.join(missing))
args.out.mkdir(parents=True)
for wheel in selected: shutil.copyfile(wheel,args.out/wheel.name)
receipt={'schema':'snapshot-core-constraint-wheels-v1','project':project['name'],
    'snapshot_sha256':hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
    'core_requirements':[str(r) for r in requirements],
    'policy':'exclude workspace distribution and incompatible declared-core wheel versions; preserve other mirrored wheels',
    'wheels':{w.name:hashlib.sha256(w.read_bytes()).hexdigest() for w in selected}}
(args.out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'project':project['name'],'core_requirements':receipt['core_requirements'],'selected_wheels':len(selected)},indent=2))
