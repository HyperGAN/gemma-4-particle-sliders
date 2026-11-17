"""Run the full four unadapted DEV controls through one matched base server."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from gemma_particle.dev_selection import load_selection_plan
from gemma_particle.harness_identity import sha256_file


def write(path,value):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection-plan',type=Path,default=Path('research/dev-selection-v1.json'))
    p.add_argument('--endpoint',required=True)
    p.add_argument('--server-identity',type=Path,required=True)
    p.add_argument('--stop-server-pid',type=int)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();plan=load_selection_plan(args.selection_plan)
    if args.out.exists():raise FileExistsError('DEV controls are immutable; incomplete runs need reconciliation')
    subprocess.run(['git','diff','--quiet','HEAD'],check=True)
    birth=None
    if args.stop_server_pid:
        proc=Path(f'/proc/{args.stop_server_pid}')
        command=[os.fsdecode(a) for a in (proc/'cmdline').read_bytes().split(b'\0') if a]
        if not any(a.endswith('/serve_vllm_lora.py') for a in command) or '--out' not in command:
            raise ValueError('owned LoRA-serving launcher required')
        output=Path(command[command.index('--out')+1])
        if not output.is_absolute():output=(proc/'cwd').resolve()/output
        if output.resolve()!=args.server_identity.resolve().parent:raise ValueError('owned server directory differs')
        birth=(proc/'stat').read_text().split(') ',1)[1].split()[19]
    controls=[c for c in plan['conditions'] if c['role']!='learned']
    args.out.mkdir(parents=True);(args.out/'logs').mkdir()
    declaration={'schema':'gemma4-full-dev-controls-plan-v1','created_at':datetime.now(timezone.utc).isoformat(),
        'condition_ids':[c['id'] for c in controls],'tasks_per_condition':55,
        'selection_plan_sha256':sha256_file(args.selection_plan),'source_sha256':sha256_file(Path(__file__)),
        'server_identity':str(args.server_identity.resolve()),'server_pid':args.stop_server_pid,'server_birth_ticks':birth,
        'scope':'complete unadapted OFF/prompt controls only; not a learned result or SOTA claim'}
    write(args.out/'plan.json',declaration);finished=[]
    try:
        for condition in controls:
            destination=args.out/condition['id']
            command=[sys.executable,'scripts/run_dev_condition.py','--selection-plan',str(args.selection_plan),
                '--condition',condition['id'],'--endpoint',args.endpoint,'--server-identity',str(args.server_identity),
                '--out',str(destination)]
            with (args.out/'logs'/(condition['id']+'.log')).open('x') as stream:
                result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f"DEV control failed; preserve {destination}")
            receipt=json.loads((destination/'result.json').read_text())
            if receipt['status']!='complete' or receipt['primary']['total']!=42 or receipt['secondary']['total']!=13:
                raise ValueError('complete fixed DEV denominators required')
            finished.append({'condition_id':condition['id'],'result':str(destination/'result.json'),
                'result_sha256':sha256_file(destination/'result.json'),
                'primary_resolved':receipt['primary']['resolved'],'secondary_resolved':receipt['secondary']['resolved']})
            write(args.out/'progress.json',{'state':'running','finished':finished,'planned_conditions':4})
            print(json.dumps(finished[-1]),flush=True)
        write(args.out/'progress.json',{'state':'complete','finished':finished,'planned_conditions':4})
    except Exception as exc:
        write(args.out/'progress.json',{'state':'error','finished':finished,'planned_conditions':4,
            'error_type':type(exc).__name__,'error':str(exc)})
        raise
    finally:
        if args.stop_server_pid:
            proc=Path(f'/proc/{args.stop_server_pid}')
            if proc.exists() and (proc/'stat').read_text().split(') ',1)[1].split()[19]==birth:
                os.kill(args.stop_server_pid,signal.SIGINT)
                write(args.out/'server-stop-request.json',{'pid':args.stop_server_pid,'birth_ticks':birth,
                    'status':'SIGINT sent to matching owned server; release requires subsequent observation'})


if __name__=='__main__':main()
