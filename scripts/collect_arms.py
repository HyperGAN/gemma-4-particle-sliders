"""Finish a declared four-arm TRAIN cohort and verify every produced patch.

Existing complete arms can be reused; incomplete directories require explicit
reconciliation and are never restarted automatically. Each agent subprocess
sees sanitized tasks. Only the separate verifier reads reference-bearing data.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from gemma_particle.curation import select_contexts,instruction_set,DEFAULT_INSTRUCTION_PROTOCOL
from gemma_particle.data import read_jsonl,write_jsonl


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--visible-tasks',type=Path,default=Path('data/splits-v1/train.jsonl'))
    p.add_argument('--raw-tasks',type=Path,default=Path('data/kaggle-official/tasks.jsonl'))
    p.add_argument('--inputs',type=Path,default=Path('data/kaggle-official'))
    p.add_argument('--wheels',type=Path,required=True)
    p.add_argument('--image',default='gemma-particle-sandbox:fastapi-deps-v2')
    p.add_argument('--server-identity',type=Path,required=True)
    p.add_argument('--endpoint',required=True)
    p.add_argument('--stop-server-pid',type=int)
    args=p.parse_args()
    plan=json.loads((args.root/'plan.json').read_text())
    protocol=plan.get('instruction_protocol',DEFAULT_INSTRUCTION_PROTOCOL)
    instructions=instruction_set(protocol)
    if protocol!=DEFAULT_INSTRUCTION_PROTOCOL:
        if plan.get('target_instructions')!=instructions:raise ValueError('declared instruction text changed')
        sources=['src/gemma_particle/curation.py','scripts/build_submission.py','scripts/run_trajectories.py','scripts/collect_arms.py']
        observed={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sources}
        if plan.get('execution_sources')!=observed:raise ValueError('declared collection sources changed')
    identity=json.loads(args.server_identity.read_text())
    if plan['server_identity']!=identity or plan['arms']!=['off','reproduce','localize','verify']:
        raise ValueError('declared model identity or four-arm plan changed')
    visible=read_jsonl(args.visible_tasks)
    if any(r['split']!='train' for r in visible):raise ValueError('TRAIN manifest required')
    selected=plan['task_ids'];outcomes=[];summary=[]
    logs=args.root/'logs';logs.mkdir(exist_ok=True)
    def execute(command,log):
        with log.open('w') as stream:
            result=subprocess.run([sys.executable,*command],stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'subprocess failed ({result.returncode}); see {log}')
    try:
        for arm in plan['arms']:
            agent=args.root/arm
            if not agent.exists():
                command=['scripts/run_trajectories.py','--tasks',str(args.visible_tasks),'--arm',arm,
                    '--model',identity['served_model'],'--endpoint',args.endpoint,'--server-identity',str(args.server_identity),
                    '--inputs',str(args.inputs),'--wheels',str(args.wheels),'--image',args.image,'--out',str(agent)]
                if protocol!=DEFAULT_INSTRUCTION_PROTOCOL:command.extend(['--instruction-protocol',protocol])
                for task_id in selected:command.extend(['--task-id',task_id])
                execute(command,logs/f'{arm}-agent.log')
            manifest=json.loads((agent/'cohort.json').read_text())
            if ([r['instance_id'] for r in manifest['runs']]!=selected or manifest['arm']!=arm
                    or manifest['server_identity']!=identity
                    or manifest.get('instruction_protocol',DEFAULT_INSTRUCTION_PROTOCOL)!=protocol):
                raise ValueError('existing arm is incomplete or belongs to another cohort; reconcile it first')
            verified=args.root/(arm+'-verification')
            if not verified.exists():
                execute(['scripts/verify_trajectories.py','--cohort',str(agent),'--tasks',str(args.raw_tasks),
                    '--inputs',str(args.inputs),'--wheels',str(args.wheels),'--image',args.image,'--out',str(verified)],logs/f'{arm}-verify.log')
            rows=read_jsonl(verified/'outcomes.jsonl')
            if [r['instance_id'] for r in rows]!=selected:raise ValueError('incomplete verification arm')
            outcomes.extend(rows)
            for row in rows:
                summary.append({k:row[k] for k in ('instance_id','arm','resolved','agent_error','capture_complete','tool_calls','elapsed_seconds')})
            (args.root/'progress.json').write_text(json.dumps({'state':'collecting','finished':summary},indent=2)+'\n')
            print(json.dumps({'arm':arm,'verified':[{'task':r['instance_id'],'resolved':r['resolved']} for r in rows]}),flush=True)
        write_jsonl(args.root/'outcomes.jsonl',outcomes)
        pairs=select_contexts(visible,outcomes)
        write_jsonl(args.root/'contexts.jsonl',pairs)
        status={'state':'complete','finished':summary,'admitted_contexts':len(pairs),
            'scope':'TRAIN collection only; not DEV, holdout or an official Kaggle score'}
        (args.root/'progress.json').write_text(json.dumps(status,indent=2)+'\n')
        print(json.dumps(status),flush=True)
    finally:
        if args.stop_server_pid:
            process=Path(f'/proc/{args.stop_server_pid}/cmdline')
            command=process.read_bytes().split(b'\0') if process.exists() else []
            expected=str(args.server_identity.parent).encode()
            if any(script in command for script in (b'scripts/serve_native.py',b'scripts/serve_vllm.py')) and expected in command:
                os.kill(args.stop_server_pid,signal.SIGINT)


if __name__=='__main__':main()
