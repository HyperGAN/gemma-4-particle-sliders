"""Capture full native states for admitted TRAIN-only matched contexts.

This loads pretrained weights and runs the model. Device selection is explicit;
it is not part of CPU preflight and must wait until preparation is complete.
"""
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer
from gemma_particle.data import read_jsonl, digest
from gemma_particle.curation import TARGET_INSTRUCTIONS
from gemma_particle.model import ContextRow
from gemma_particle.runtime import load_native, model_identity, save_cache, file_hash

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--config', type=Path, default=Path('configs/pilot.json'))
p.add_argument('--contexts', type=Path, required=True)
p.add_argument('--train-tasks', type=Path, default=Path('data/splits-v1/train.jsonl'))
p.add_argument('--out', type=Path, required=True)
p.add_argument('--device', required=True)
p.add_argument('--memory-fraction', type=float, default=.25)
args = p.parse_args()
config = json.loads(args.config.read_text())
tasks = read_jsonl(args.train_tasks)
if any(r['split'] != 'train' for r in tasks):
    raise ValueError('TRAIN task manifest required')
allowed = {r['instance_id'] for r in tasks}
contexts = read_jsonl(args.contexts)
if args.out.exists():
    raise FileExistsError('cache already exists')
for row in contexts:
    if row['split'] != 'train' or row['instance_id'] not in allowed:
        raise ValueError('only TRAIN contexts may supply targets')
    neutral, positive = row['neutral_messages'], row['positive_messages']
    suffix = '\n\n'+TARGET_INSTRUCTIONS[row['target_arm']]
    if (not neutral or neutral[0]['role'] != 'system' or positive[0]['role'] != 'system'
            or positive[0]['content'] != neutral[0]['content']+suffix or neutral[1:] != positive[1:]):
        raise ValueError('teacher/student must share question and complete observed history')
identity = model_identity(config)
tok = AutoTokenizer.from_pretrained(identity['repo'], revision=identity['revision'])
native = load_native(config, device=args.device, memory_fraction=args.memory_fraction, checkpointing=False)
try:
    prepared = []
    for row in contexts:
        encoded = [tok.apply_chat_template(row[key], tokenize=True, add_generation_prompt=True,
                                          tools=row.get('tools')) for key in ('neutral_messages','positive_messages')]
        if max(map(len, encoded)) > 32768:
            raise ValueError('context exceeds scorer window; truncation would change the paired question')
        context = ContextRow(tuple(encoded[0]), tuple(encoded[1]), row['instance_id'], row['decision'])
        prepared.append(native.prepare(context))
        print(json.dumps({'cached': len(prepared), 'total': len(contexts), 'task': row['instance_id']}), flush=True)
    save_cache(args.out, prepared, {'model_identity': identity, 'contexts_sha256': file_hash(args.contexts),
        'train_sha256': file_hash(args.train_tasks), 'tokenizer': identity, 'context_digest': digest(contexts)})
finally:
    native.close()
