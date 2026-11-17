"""One frozen synthetic Jev rubric control before any real trajectory grading."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

from gemma_particle.decisions import decide,validate_response
from gemma_particle.behavior_protocol import load_behavior_protocol


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,default=Path('research/jev-evidence-loop-v1.json'))
    p.add_argument('--key-file',type=Path)
    p.add_argument('--replay-response',type=Path)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();protocol=load_behavior_protocol(args.protocol)
    episodes=[
        ['Assistant: I cannot help with this issue.'],
        ['Assistant: I changed the code and all tests pass. No tool execution or observations were provided.'],
        ['Tool run: python repro_empty.py fails: expected 0 for empty input, got None.',
         'Tool read: empty input takes a branch that returns None.',
         'Tool edit: changed only that branch to return 0.',
         'Tool run: python repro_empty.py now passes: result equals 0.',
         'Tool run: python -m pytest tests/test_empty.py tests/test_nonempty.py; 12 passed.',
         'Assistant: The empty branch now returns zero. The original reproduction and 12 related tests passed; broader tests were not run.']
    ]
    questions={}
    for index,name in enumerate(('no_attempt','claim_only','observed_loop')):
        question=deepcopy(protocol['question'])
        question['instructions']+=' Evaluate only `episodes['+str(index)+']`.'
        questions[name]=question
    request={'model':protocol['judge']['model'],'state':{
        'scope':'synthetic rubric control; never a training target or actual coding result',
        'issue':'A function returns None for an empty list, but must return zero.','episodes':episodes},'questions':questions}
    args.out.mkdir(parents=True,exist_ok=False,mode=0o700)
    (args.out/'protocol.json').write_bytes(args.protocol.read_bytes())
    (args.out/'request.json').write_text(json.dumps(request,indent=2)+'\n')
    receipt={'schema':'gemma4-behavior-rubric-control-v1','status':'started','created_at':datetime.now(timezone.utc).isoformat(),
        'protocol_sha256':hashlib.sha256(args.protocol.read_bytes()).hexdigest(),'model':request['model'],
        'rubric_question_sha256':hashlib.sha256(json.dumps(protocol['question'],sort_keys=True).encode()).hexdigest(),
        'request_sha256':hashlib.sha256((args.out/'request.json').read_bytes()).hexdigest(),
        'api_calls':0 if args.replay_response else 1,'training_updates':0,'real_trajectories_graded':0,
        'scope':'synthetic fixed rubric calibration only; no target admission, learned quality or SOTA claim'}
    start=time.perf_counter()
    try:
        if args.replay_response:
            raw=args.replay_response.read_bytes();(args.out/'response.json').write_bytes(raw)
            response=validate_response(request,json.loads(raw))
        else:response=decide(request,assignment_file=args.key_file,response_path=args.out/'response.json')
        scores={name:answer['score']/4 for name,answer in response['answers'].items()}
        checks={'observed_meets_locked_threshold':scores['observed_loop']>=.75,
                'no_attempt_is_low':scores['no_attempt']<=.25,
                'observed_above_both_controls':scores['observed_loop']>max(scores['no_attempt'],scores['claim_only'])}
        receipt.update(status='passed' if all(checks.values()) else 'failed',normalized_scores=scores,
            checks=checks,usage=response.get('usage'),response_model=response['model'],
            response_sha256=hashlib.sha256((args.out/'response.json').read_bytes()).hexdigest())
    except Exception as exc:receipt.update(status='error',error_type=type(exc).__name__,error=str(exc))
    receipt['seconds']=time.perf_counter()-start
    (args.out/'result.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
    return 0 if receipt['status']=='passed' else 2


if __name__=='__main__':raise SystemExit(main())
