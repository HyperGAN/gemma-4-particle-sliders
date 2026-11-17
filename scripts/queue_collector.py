"""Queue a source-pinned collector behind one verified live collector handle."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

from gemma_particle.data import digest
from gemma_particle.harness_identity import sha256_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous-pid',type=int,required=True)
    p.add_argument('--previous-progress',type=Path,required=True)
    p.add_argument('--executor',type=Path,required=True)
    p.add_argument('--executor-commit',required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--wheels',type=Path,required=True)
    p.add_argument('--server-identity',type=Path,required=True)
    p.add_argument('--endpoint',required=True)
    args=p.parse_args()
    args.root=args.root.resolve();args.executor=args.executor.resolve()
    args.previous_progress=args.previous_progress.resolve()
    plan_path=args.root/'plan.json';plan_sha=sha256_file(plan_path)
    plan=json.loads(plan_path.read_text())
    state=args.root/'queue.json'
    if state.exists():raise FileExistsError('queue handle is immutable; inspect it instead of restarting')
    previous=Path(f'/proc/{args.previous_pid}')
    if not previous.exists():raise ValueError('queue requires an observed live predecessor at creation')
    birth=(previous/'stat').read_text().split()[21]
    command=[os.fsdecode(x) for x in (previous/'cmdline').read_bytes().split(b'\0') if x]
    if 'scripts/collect_arms.py' not in command or '--root' not in command:
        raise ValueError('predecessor is not the specified collector')
    old_root=Path(command[command.index('--root')+1])
    if not old_root.is_absolute():old_root=(previous/'cwd').resolve()/old_root
    if old_root.resolve()!=args.previous_progress.parent:raise ValueError('predecessor progress path differs')
    def record(status,**fields):
        value={'state':status,'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'previous_pid':args.previous_pid,'plan_sha256':plan_sha,'executor_commit':args.executor_commit,**fields}
        temp=state.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(state)
        print(json.dumps(value),flush=True)
    record('waiting_for_live_predecessor')
    while True:
        try:observed=(previous/'stat').read_text().split()
        except FileNotFoundError:break
        if observed[2]=='Z' or observed[21]!=birth:break
        time.sleep(10)
    if not args.previous_progress.exists() or json.loads(args.previous_progress.read_text()).get('state')!='complete':
        record('failed_before_start',reason='predecessor ended without complete verification');return 2
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.executor,text=True).strip()
    if actual!=args.executor_commit or sha256_file(plan_path)!=plan_sha:
        record('failed_before_start',reason='executor or plan changed');return 2
    if any(sha256_file(args.executor/s)!=sha for s,sha in plan['execution_sources'].items()):
        record('failed_before_start',reason='executor source bytes changed');return 2
    if digest(json.loads(args.server_identity.read_text()))!=digest(plan['server_identity']):
        record('failed_before_start',reason='bound server identity changed');return 2
    env=dict(os.environ,PYTHONPATH=str(args.executor/'src'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    with (args.root/'collector.log').open('w') as log:
        child=subprocess.Popen([str(args.executor/'.harness-venv/bin/python'),'scripts/collect_arms.py',
            '--root',str(args.root),'--wheels',str(args.wheels.resolve()),
            '--server-identity',str(args.server_identity.resolve()),'--endpoint',args.endpoint],
            cwd=args.executor,env=env,stdout=log,stderr=subprocess.STDOUT)
        record('collecting',collector_pid=child.pid)
        code=child.wait()
    record('collector_complete' if code==0 else 'collector_failed',returncode=code)
    return code


if __name__=='__main__':raise SystemExit(main())
