"""CPU payload and fake-service controls; no actual upload or learned model."""
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from requests.exceptions import HTTPError, Timeout

from gemma_particle.harness_identity import sha256_file
from gemma_particle.kaggle_publication import audit_payload, publish_once, MutationUncertain
from test_kaggle_candidate import fixture, module


def prepared(root, monkeypatch):
    artifact, native, cache, conversion = fixture(root)
    preparer = module('publication_payload_preparer', 'prepare_kaggle_candidate.py')
    out = root / 'payload'
    monkeypatch.setattr('sys.argv', ['prepare', '--artifact', str(artifact), '--native', str(native),
        '--cache', str(cache), '--conversion-report', str(conversion), '--owner', 'fixture-owner',
        '--notebook-slug', 'fixture-candidate', '--out', str(out)])
    preparer.main()
    candidate = audit_payload(out, sha256_file(artifact / 'submission.zip'), preparer.notebook_source)
    return candidate, artifact, preparer


class FakeAPI:
    CONFIG_NAME_USER = 'username'
    def __init__(self, candidate, archive, out, mode):
        self.candidate = candidate; self.archive = archive; self.out = out; self.mode = mode
        self.creates = self.pushes = self.submits = self.status_reads = self.quota_reads = 0
        self.output_handles = []; self.kernel_handles = []; self.rows_reads = 0
    def get_config_value(self, key):
        return 'wrong-owner' if self.mode == 'wrong_owner' else self.candidate['owner']
    def dataset_status(self, name):
        assert name == self.candidate['dataset']
        if not self.creates:
            error = HTTPError('CPU nonexistent dataset'); error.response = NS(status_code=404); raise error
        self.status_reads += 1
        if self.mode == 'read_timeout' and self.status_reads == 1: raise Timeout('CPU observation expired')
        return 'processing' if self.status_reads == 1 else 'ready'
    def kernels_status(self, handle):
        self.kernel_handles.append(handle)
        if '/' not in handle.removeprefix(self.candidate['notebook']):
            if handle == self.candidate['notebook']:
                if self.mode == 'existing_notebook': return NS(status='COMPLETE')
                error = HTTPError('CPU nonexistent notebook'); error.response = NS(status_code=404); raise error
        assert handle == self.candidate['notebook'] + '/1'
        count = len([value for value in self.kernel_handles if value == handle])
        return NS(status='NEW_SCRIPT' if count <= 2 else 'ERROR' if self.mode == 'notebook_error' else 'COMPLETE')
    def dataset_create_new(self, folder, **kwargs):
        self.creates += 1
        assert (self.out / 'plan.json').exists() and (self.out / 'dataset-create-intent.json').exists()
        assert kwargs == {'public': False, 'quiet': True, 'convert_to_csv': False, 'dir_mode': 'skip'}
        if self.mode == 'dataset_ambiguous': raise Timeout('CPU create result unknown')
        return NS(ref=self.candidate['dataset'], url='https://www.kaggle.com/datasets/' + self.candidate['dataset'], status='ok', error='')
    def dataset_details(self, identifier):
        return {'ref': identifier, 'is_private': self.mode != 'public_dataset', 'version': 1}
    def kernels_push(self, folder):
        self.pushes += 1
        assert (self.out / 'notebook-push-intent.json').exists()
        metadata = json.loads((Path(folder) / 'kernel-metadata.json').read_text())
        assert metadata['dataset_sources'] == [self.candidate['dataset'] + '/1']
        assert metadata['title'] == 'fixture candidate' and metadata['is_private'] and not metadata['enable_gpu'] and not metadata['enable_internet']
        assert sha256_file(Path(folder) / 'candidate.ipynb') == self.candidate['files'][str(Path(self.candidate['payload']) / 'notebook/candidate.ipynb')]
        if self.mode == 'push_ambiguous': raise Timeout('CPU push result unknown')
        return NS(ref=self.candidate['notebook'], url='https://www.kaggle.com/code/' + self.candidate['notebook'],
            version_number=1, kernel_id=123, error='', invalid_dataset_sources=[], invalid_competition_sources=[],
            invalid_kernel_sources=[], invalid_model_sources=[])
    def kernels_output(self, handle, destination, **kwargs):
        assert handle == self.candidate['notebook'] + '/1'
        self.output_handles.append(handle); out = Path(destination); out.mkdir(exist_ok=True)
        archive = out / 'submission.zip'; archive.write_bytes(self.archive.read_bytes() + (b'changed' if self.mode == 'corrupt_output' else b''))
        return [str(archive)], ''
    def competition_submissions(self, competition, **kwargs):
        self.rows_reads += 1
        if self.mode == 'matching':
            return [NS(ref=71, description='Particle Sliders learned; locked DEV; sha256=' + self.candidate['artifact_sha256'], status='COMPLETE')]
        return [NS(ref=42, description='existing OFF control', status='PENDING')] if self.rows_reads == 1 else []
    def competition_get_submission_limits(self, competition):
        self.quota_reads += 1
        return NS(num_allowed_now=0 if self.quota_reads <= 2 else 1, num_today=1, num_total=2, limited_by_total=False)
    def competition_submit_code(self, **kwargs):
        self.submits += 1
        assert (self.out / 'submission-intent.json').exists()
        assert kwargs['kernel'] == self.candidate['notebook'] and kwargs['kernel_version'] == 1 and kwargs['file_name'] == 'submission.zip'
        assert kwargs['competition'] == 'gemma-4-developer-agent'
        if self.mode == 'submit_ambiguous': raise Timeout('CPU submit response unknown')
        return NS(ref=701, message='CPU fixture accepted;not a real submission')


@pytest.mark.parametrize('damage', [None, 'extra_file', 'GPU', 'dataset', 'code', 'envelope', 'provenance'])
def test_only_exact_prepared_payload_and_canonical_offline_emitter_are_publishable(tmp_path, monkeypatch, damage):
    candidate, artifact, preparer = prepared(tmp_path, monkeypatch); root = Path(candidate['payload'])
    if damage == 'extra_file': (root / 'dataset/PRIVATE_EPISODES_MUST_NOT_UPLOAD').write_text('CPU fixture')
    elif damage in ['GPU', 'dataset']:
        path = root / 'notebook/kernel-metadata.json'; value = json.loads(path.read_text())
        if damage == 'GPU': value['enable_gpu'] = True
        else: value['dataset_sources'] = ['unrelated/data']
        path.write_text(json.dumps(value))
    elif damage == 'code':
        path = root / 'notebook/candidate.ipynb'; value = json.loads(path.read_text()); value['cells'][1]['source'] += '\nraise RuntimeError("changed")'
        path.write_text(json.dumps(value))
        summary_path = root / 'preparation.json'; summary = json.loads(summary_path.read_text())
        summary['notebook_sha256'] = sha256_file(path); summary_path.write_text(json.dumps(summary))
    elif damage == 'envelope':
        path = root / 'dataset/agent_bundle.bin'; path.write_bytes(path.read_bytes() + b'changed')
    elif damage == 'provenance': (root / 'dataset/candidate.json').write_text('{}')
    if damage:
        with pytest.raises(ValueError): audit_payload(root, candidate['artifact_sha256'], preparer.notebook_source)
    else:
        assert audit_payload(root, candidate['artifact_sha256'], preparer.notebook_source)['artifact_sha256'] == sha256_file(artifact / 'submission.zip')


@pytest.mark.parametrize('mode', ['complete', 'read_timeout', 'wrong_owner', 'existing_notebook', 'public_dataset',
    'dataset_ambiguous', 'push_ambiguous', 'notebook_error', 'corrupt_output', 'matching', 'submit_ambiguous'])
def test_one_shot_private_publication_and_exact_notebook_version(tmp_path, monkeypatch, mode):
    candidate, artifact, _ = prepared(tmp_path, monkeypatch); out = tmp_path / 'publication'
    api = FakeAPI(candidate, artifact / 'submission.zip', out, mode); pauses = []; verifications = []
    def verify(): verifications.append(True)
    if mode in ['complete', 'read_timeout', 'matching']:
        result = publish_once(api, candidate, out, verify, sleep=lambda seconds: pauses.append(seconds), declaration={'CPU-fixture': True})
    else:
        with pytest.raises((ValueError, RuntimeError)):
            publish_once(api, candidate, out, verify, sleep=lambda seconds: pauses.append(seconds), declaration={'CPU-fixture': True})
        result = json.loads((out / 'progress.json').read_text())
    assert api.creates <= 1 and api.pushes <= 1 and api.submits <= 1 and verifications
    if mode in ['complete', 'read_timeout']:
        assert result['state'] == 'created' and result['submission_id'] == 701 and api.submits == 1
        assert api.quota_reads == 3 and api.output_handles == [candidate['notebook'] + '/1']
        assert len([h for h in api.kernel_handles if h.endswith('/1')]) == 3 and pauses
        assert json.loads((out / 'verified-notebook-output.json').read_text())['artifact_sha256'] == candidate['artifact_sha256']
    elif mode == 'matching':
        assert api.submits == 0 and result['state'] == 'existing_matching_submission_requires_reconciliation'
    elif mode.endswith('ambiguous'):
        assert result['state'] == 'mutation_requires_reconciliation'
        before = (api.creates, api.pushes, api.submits)
        with pytest.raises(FileExistsError): publish_once(api, candidate, out, verify)
        assert before == (api.creates, api.pushes, api.submits)
    else:
        assert api.submits == 0
        if mode in ['wrong_owner', 'existing_notebook']: assert api.creates == 0


def test_dataset_sdk_retry_override_is_restored_after_ambiguous_result():
    publisher = module('publication_sdk_adapter', 'publish_selected_candidate.py')
    original_calls = []; mutations = []
    def original(function):
        original_calls.append(True); return function
    class API:
        with_retry = staticmethod(original)
        def dataset_create_new(self):
            def create(): mutations.append(True); raise Timeout('CPU ambiguous create')
            return self.with_retry(create)()
    api = API(); before = api.with_retry
    with pytest.raises(Timeout): publisher.KagglePublicationClient(api).dataset_create_new()
    assert mutations == [True] and original_calls == [] and api.with_retry is before
