"""Offline bindings and one-shot publication of an evaluated learned payload.

Local files establish consistency, not authenticity of an invented experiment.
The SDK is supplied by the caller only after complete experiment validation.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from urllib.parse import urlsplit

from requests.exceptions import ConnectionError, Timeout

from .conversion_handoff import read
from .harness_identity import sha256_file

COMPETITION = 'gemma-4-developer-agent'
MAGIC = b'GEMMA4_PARTICLE_AGENT_BUNDLE_V1\n'


def durable_json(path, value):
    """Exclusive durable intent/result; an existing operation is never replayed."""
    path = Path(path)
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def audit_payload(path, archive_sha256, source_function):
    path = Path(path).resolve(); preparation = read(path / 'preparation.json')
    dataset = path / 'dataset'; notebook = path / 'notebook'
    expected = {dataset: {'agent_bundle.bin', 'candidate.json', 'dataset-metadata.json'},
        notebook: {'candidate.ipynb', 'kernel-metadata.json'}}
    for directory, names in expected.items():
        if {p.name for p in directory.iterdir()} != names or any(p.is_symlink() or not p.is_file() for p in directory.iterdir()):
            raise ValueError('only the exact prepared model payload files may be uploaded')
    metadata = read(dataset / 'dataset-metadata.json'); kernel = read(notebook / 'kernel-metadata.json')
    candidate = read(dataset / 'candidate.json'); document = read(notebook / 'candidate.ipynb')
    candidate_keys = {'schema', 'artifact_sha256', 'wrapper_sha256', 'adapter_identity', 'native_completed_steps',
        'native_metadata_sha256', 'cache_index_sha256', 'conversion_report_sha256', 'export_strength',
        'exact_merge', 'official_score', 'scope'}
    owner, slug = preparation['dataset'].split('/'); notebook_owner, notebook_slug = preparation['notebook'].split('/')
    if (owner != notebook_owner or any(not re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*', v)
            for v in [owner, slug, notebook_slug]) or not 6 <= len(notebook_slug) <= 50
            or preparation.get('schema') != 'gemma4-kaggle-candidate-payload-v1'
            or preparation['artifact_sha256'] != archive_sha256 or not slug.endswith(archive_sha256[:12])
            or preparation.get('gpu_enabled') is not False or preparation.get('model_inference') is not False
            or preparation.get('official_score') is not None
            or metadata['id'] != preparation['dataset'] or metadata['licenses'] != [{'name': 'other'}]
            or set(metadata) != {'id', 'title', 'licenses', 'description', 'resources'}
            or len(metadata['resources']) != 2 or set(candidate) != candidate_keys
            or {r['path'] for r in metadata['resources']} != {'agent_bundle.bin', 'candidate.json'}
            or candidate != {k: preparation[k] for k in candidate}
            or kernel != {'id': preparation['notebook'], 'title': metadata['title'], 'code_file': 'candidate.ipynb',
                'language': 'python', 'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': False,
                'enable_tpu': False, 'enable_internet': False, 'competition_sources': [COMPETITION],
                'dataset_sources': [preparation['dataset']], 'kernel_sources': [], 'model_sources': []}
            or sha256_file(notebook / 'candidate.ipynb') != preparation['notebook_sha256']
            or document['nbformat'] != 4 or len(document['cells']) != 2
            or [cell['cell_type'] for cell in document['cells']] != ['markdown', 'code']
            or document['cells'][1]['source'] != source_function(owner, slug, preparation['wrapper_sha256'], archive_sha256)
            or document['cells'][1]['outputs'] != [] or document['cells'][1]['execution_count'] is not None):
        raise ValueError('exact evaluated payload,private offline CPU emitter and declared inputs required')
    wrapper = dataset / 'agent_bundle.bin'
    if sha256_file(wrapper) != preparation['wrapper_sha256']:
        raise ValueError('original prepared binary envelope required')
    with wrapper.open('rb') as stream:
        if stream.read(len(MAGIC)) != MAGIC or hashlib.file_digest(stream, 'sha256').hexdigest() != archive_sha256:
            raise ValueError('envelope must contain the exact evaluated archive bytes')
    files = [path / 'preparation.json', *dataset.iterdir(), *notebook.iterdir()]
    return {'payload': str(path), 'dataset': preparation['dataset'], 'notebook': preparation['notebook'],
        'owner': owner, 'artifact_sha256': archive_sha256, 'wrapper_sha256': preparation['wrapper_sha256'],
        'files': {str(p): sha256_file(p) for p in files}, 'kernel_metadata': kernel}


def normalized_status(value):
    return str(getattr(value, 'name', value)).split('.')[-1].upper()


def safe_error(exc):
    return {'error_type': type(exc).__name__, 'http_status': getattr(getattr(exc, 'response', None), 'status_code', None)}


def valid_url(value, kind, identifier):
    parsed = urlsplit(value)
    return (parsed.scheme == 'https' and parsed.netloc in ['www.kaggle.com', 'kaggle.com']
        and parsed.path.rstrip('/') == f'/{kind}/{identifier}' and not parsed.query and not parsed.fragment)


class MutationUncertain(RuntimeError):
    pass


def publish_once(api, candidate, out, verify, sleep=time.sleep, poll_seconds=60, declaration=None):
    """Publish privately and submit one exact completed notebook version.

    The caller's API implements dataset_details as an additional read-only
    adapter. Every mutation has an intent before its one SDK invocation.
    Read timeouts retain the same handles. No mutation is automatically retried.
    """
    out = Path(out)
    if out.exists(): raise FileExistsError('publication is immutable; reconcile existing intents and handles')
    out.mkdir(parents=True, mode=0o700)
    if declaration is not None: durable_json(out / 'plan.json', declaration)
    status = {'state': 'starting', 'dataset_create_calls': 0, 'notebook_push_calls': 0,
        'submission_calls': 0, 'submission_id': None, 'model_inference': False, 'gpu_enabled': False}
    def publish(state, **extra):
        status.update(state=state, as_of=datetime.now(timezone.utc).isoformat(), **extra)
        temporary = out / 'progress.json.tmp'
        temporary.write_text(json.dumps(status, indent=2) + '\n'); temporary.replace(out / 'progress.json')

    def observe(operation, function, *args, **kwargs):
        while True:
            verify()
            try: return function(*args, **kwargs)
            except Exception as exc:
                code = safe_error(exc)['http_status']
                if not (isinstance(exc, (ConnectionError, Timeout, TimeoutError))
                        or code in [408, 429] or isinstance(code, int) and 500 <= code <= 599):
                    raise
                publish('waiting_observation', operation=operation, **safe_error(exc))
                sleep(poll_seconds)

    def absent(operation, function, identifier):
        try: observe(operation, function, identifier)
        except Exception as exc:
            if safe_error(exc)['http_status'] == 404: return
            raise
        raise ValueError('existing remote artifact requires reconciliation: ' + operation)

    def mutate(operation, function, intent, capture):
        verify()
        if operation == 'dataset-create' and sha256_file(Path(candidate['payload']) / 'dataset/agent_bundle.bin') != candidate['wrapper_sha256']:
            raise ValueError('exact evaluated envelope required immediately before upload')
        durable_json(out / (operation + '-intent.json'), {'as_of': datetime.now(timezone.utc).isoformat(), **intent})
        status[{'dataset-create': 'dataset_create_calls', 'notebook-push': 'notebook_push_calls',
            'submission': 'submission_calls'}[operation]] += 1
        publish('one_mutation_call', operation=operation)
        try:
            value = function(); result = capture(value)
            durable_json(out / (operation + '-response.json'), result)
            return value, result
        except Exception as exc:
            durable_json(out / (operation + '-failure.json'), safe_error(exc))
            publish('mutation_requires_reconciliation', operation=operation, **safe_error(exc))
            raise MutationUncertain(operation + ' did not produce a verified response;retain intent and reconcile') from exc

    try:
        verify()
        if api.get_config_value(api.CONFIG_NAME_USER) != candidate['owner']:
            raise ValueError('authenticated Kaggle account must own the declared payload')
        absent('dataset-existence', api.dataset_status, candidate['dataset'])
        absent('notebook-existence', api.kernels_status, candidate['notebook'])
        _, created = mutate('dataset-create',
            lambda: api.dataset_create_new(str(Path(candidate['payload']) / 'dataset'), public=False,
                quiet=True, convert_to_csv=False, dir_mode='skip'),
            {'dataset': candidate['dataset'], 'private': True, 'files': candidate['files']},
            lambda r: {'ref': r.ref, 'url': r.url, 'status': r.status, 'api_error_present': bool(r.error)})
        if (created['api_error_present'] or normalized_status(created['status']) == 'ERROR'
                or created['ref'] not in ['', candidate['dataset']] or not valid_url(created['url'], 'datasets', candidate['dataset'])):
            raise MutationUncertain('dataset response differs;retain original create intent')
        while True:
            value = observe('dataset-ready', api.dataset_status, candidate['dataset'])
            publish('waiting_dataset', observed_status=normalized_status(value))
            if normalized_status(value) == 'READY': break
            if normalized_status(value) in ['ERROR', 'FAILED']: raise RuntimeError('actual dataset processing failed')
            sleep(poll_seconds)
        details = observe('dataset-details', api.dataset_details, candidate['dataset'])
        durable_json(out / 'dataset-details.json', details)
        if details != {'ref': candidate['dataset'], 'is_private': True, 'version': 1}:
            raise ValueError('new private dataset version1 must be observed before notebook publication')
        staging = out / 'notebook'; staging.mkdir()
        shutil.copyfile(Path(candidate['payload']) / 'notebook/candidate.ipynb', staging / 'candidate.ipynb')
        kernel = dict(candidate['kernel_metadata'])
        kernel['title'] = kernel['id'].split('/')[1].replace('-', ' ')
        kernel['dataset_sources'] = [candidate['dataset'] + '/1']
        durable_json(staging / 'kernel-metadata.json', kernel)
        durable_json(out / 'notebook-staging.json', {'original_metadata_sha256': candidate['files'][str(Path(candidate['payload']) / 'notebook/kernel-metadata.json')],
            'staged_metadata_sha256': sha256_file(staging / 'kernel-metadata.json'),
            'notebook_sha256': sha256_file(staging / 'candidate.ipynb'),
            'changes': {'title': kernel['title'], 'dataset_sources': kernel['dataset_sources']},
            'scope': 'bind title to declared slug and pin dataset1;emitter code and evaluated agent unchanged'})
        _, pushed = mutate('notebook-push', lambda: api.kernels_push(str(staging)),
            {'notebook': candidate['notebook'], 'metadata_sha256': sha256_file(staging / 'kernel-metadata.json'),
                'notebook_sha256': sha256_file(staging / 'candidate.ipynb'), 'private': True, 'GPU': False, 'internet': False},
            lambda r: {'ref': r.ref, 'url': r.url, 'version': r.version_number, 'kernel_id': r.kernel_id,
                'api_error_present': bool(r.error), 'invalid_sources_present': any(getattr(r, name) for name in
                    ['invalid_dataset_sources', 'invalid_competition_sources', 'invalid_kernel_sources', 'invalid_model_sources'])})
        if (pushed['api_error_present'] or pushed['invalid_sources_present']
                or pushed['ref'] not in ['', candidate['notebook']] or not valid_url(pushed['url'], 'code', candidate['notebook'])
                or type(pushed['version']) is not int or pushed['version'] != 1 or pushed['kernel_id'] <= 0):
            raise MutationUncertain('new notebook response differs;retain original push intent')
        handle = candidate['notebook'] + '/' + str(pushed['version'])
        while True:
            value = observe('notebook-status', api.kernels_status, handle)
            phase = normalized_status(value.status); publish('waiting_notebook', notebook_handle=handle, observed_status=phase)
            if phase == 'COMPLETE': break
            if phase in ['ERROR', 'CANCEL_ACKNOWLEDGED']: raise RuntimeError('actual notebook terminated unsuccessfully')
            sleep(poll_seconds)
        downloaded = out / 'downloaded'
        files, token = observe('notebook-output', api.kernels_output, handle, str(downloaded),
            file_pattern=r'^submission\.zip$', quiet=True)
        archive = downloaded / 'submission.zip'
        if token or str(archive) not in files or not archive.is_file() or archive.is_symlink() or sha256_file(archive) != candidate['artifact_sha256']:
            raise ValueError('downloaded exact notebook version must emit the original evaluated archive')
        durable_json(out / 'verified-notebook-output.json', {'kernel': candidate['notebook'], 'version': pushed['version'],
            'handle': handle, 'artifact_sha256': sha256_file(archive), 'path': str(archive)})
        message = 'Particle Sliders learned; locked DEV; sha256=' + candidate['artifact_sha256']
        while True:
            rows = observe('competition-submissions', api.competition_submissions, COMPETITION, page_size=100)
            if rows is None or any(row is None for row in rows) or len(rows) >= 100:
                raise ValueError('complete bounded account submission observation required;reconcile pagination')
            matching = [row.ref for row in rows if row.description == message]
            if matching:
                publish('existing_matching_submission_requires_reconciliation', matching_ids=matching); return status
            pending = [row.ref for row in rows if normalized_status(row.status) not in ['COMPLETE', 'ERROR']]
            limits = observe('competition-limits', api.competition_get_submission_limits, COMPETITION)
            if type(limits.num_allowed_now) is not int or limits.num_allowed_now < 0:
                raise ValueError('actual server submission allowance required')
            durable_json(out / ('quota-observation-' + str(len(list(out.glob('quota-observation-*.json')))) + '.json'),
                {'observed_at': datetime.now(timezone.utc).isoformat(), 'pending_ids': pending,
                    'num_allowed_now': limits.num_allowed_now, 'num_today': limits.num_today,
                    'num_total': limits.num_total, 'limited_by_total': limits.limited_by_total})
            if not pending and limits.num_allowed_now > 0: break
            if not pending and limits.num_allowed_now == 0 and limits.limited_by_total:
                raise RuntimeError('actual lifetime submission limit exhausted;preserve qualified payload')
            publish('waiting_competition_gate', pending_ids=pending, num_allowed_now=limits.num_allowed_now)
            sleep(poll_seconds)
        verify()
        if sha256_file(archive) != candidate['artifact_sha256']:
            raise ValueError('verified notebook output changed before competition request')
        _, submitted = mutate('submission',
            lambda: api.competition_submit_code(file_name='submission.zip', message=message, competition=COMPETITION,
                kernel=candidate['notebook'], kernel_version=pushed['version'], quiet=True),
            {'competition': COMPETITION, 'kernel': candidate['notebook'], 'version': pushed['version'],
                'file_name': 'submission.zip', 'artifact_sha256': candidate['artifact_sha256'], 'message': message},
            lambda r: {'submission_id': r.ref, 'message': r.message})
        if type(submitted['submission_id']) is not int or submitted['submission_id'] <= 0:
            raise MutationUncertain('actual positive submission ID required;retain intent without retry')
        publish('created', submission_id=submitted['submission_id'], notebook_handle=handle,
            official_score=None, scope='official evaluation queued;poll exact submission ID;fresh SOTA comparison remains required')
        return status
    except MutationUncertain:
        if status['state'] != 'mutation_requires_reconciliation': publish('mutation_requires_reconciliation')
        raise
    except BaseException as exc:
        publish('error', **safe_error(exc)); raise
