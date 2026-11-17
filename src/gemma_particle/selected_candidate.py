"""Bind a production payload to the archive already used in DEV/confirmation."""
import hashlib
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import yaml

from .adapters import adapter_identity
from .budgets import budget_values
from .conversion_handoff import read
from .harness_identity import sha256_file


def selected_evaluated_archive(scope, comparison):
    """Check actual local bytes, without selecting on confirmation outcomes."""
    selected = scope.receipt['selected_condition']['id']; packet = scope.receipt['selected_packet']
    results = comparison['condition_results']
    confirmation = read(results[selected]['path'])
    choice = read(scope.receipt['dev_choice'])
    dev = read(choice['condition_results'][selected]['path'])
    if (sha256_file(results[selected]['path']) != results[selected]['sha256']
            or sha256_file(choice['condition_results'][selected]['path']) != choice['condition_results'][selected]['sha256']
            or dev['condition'] != packet['condition'] or dev['status'] != 'complete'):
        raise ValueError('unchanged originally selected DEV and confirmation results required')
    artifact = Path(confirmation['finished'][0]['cohort']) / 'artifact'
    receipt = read(artifact / 'build.json'); matching = scope.plan['matching']
    if (receipt.get('arm') != 'distilled-particle'
            or receipt.get('prompt_profile') != matching['prompt_profile']
            or receipt.get('instruction_protocol') != matching['instruction_protocol']
            or receipt.get('budget_profile') != matching['budget_profile']):
        raise ValueError('exact evaluated learned prompt/tool/budget profiles required')
    for record in [*dev['finished'], *confirmation['finished']]:
        cohort = Path(record['cohort']); manifest = read(cohort / 'cohort.json')
        if manifest['artifact'] != receipt or read(cohort / 'artifact/build.json') != receipt:
            raise ValueError('one byte-identical production archive across every selected DEV/confirmation case required')
    archive = artifact / 'submission.zip'
    if archive.is_symlink() or sha256_file(archive) != receipt['sha256']:
        raise ValueError('original evaluated archive bytes required')
    with ZipFile(archive) as bundle:
        names = bundle.namelist()
        if (len(set(names)) != len(names) or set(names) != set(receipt['files'])
                or any(i.is_dir() for i in bundle.infolist())
                or any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts for n in names)
                or sum(i.file_size for i in bundle.infolist()) >= 3 * 1024**3 or bundle.testzip() is not None):
            raise ValueError('exact safe file-only evaluated archive required')
        for name in names:
            local = artifact / 'agent' / name
            if (local.is_symlink() or sha256_file(local) != receipt['files'][name]
                    or hashlib.sha256(bundle.read(name)).hexdigest() != receipt['files'][name]):
                raise ValueError('packaged and local evaluated agent files differ')
        if (receipt['files']['prompts/system.md'] != matching['common_system_sha256']
                or receipt['files']['configs/sampling.yaml'] != matching['sampling_file_sha256']
                or yaml.safe_load(bundle.read('eval_config.yaml')) != {'evaluation': budget_values(matching['budget_profile'])}
                or {'particle_lora': adapter_identity(artifact / 'agent/adapters/particle_lora')} != packet['adapters']):
            raise ValueError('production prompt,sampling,budget or selected adapter differs')
    return artifact, receipt
