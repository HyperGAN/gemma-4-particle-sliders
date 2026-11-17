"""Publish and submit one exact, fully evaluated learned candidate privately.

Plan-only performs no account import or protected input read. Actual mode
reaudits the entire selected experiment before authentication or publication.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
from importlib.metadata import version, distribution
import json
from pathlib import Path
import subprocess

from gemma_particle.harness_identity import sha256_file
from gemma_particle.kaggle_publication import publish_once, safe_error
from gemma_particle.selected_publication import audit_selected_publication


class KagglePublicationClient:
    """Small adapter around the installed authenticated SDK; no credential output."""
    def __init__(self, api):
        self.api = api

    def __getattr__(self, name):
        return getattr(self.api, name)

    def dataset_details(self, identifier):
        from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
        request = ApiGetDatasetRequest(); request.owner_slug, request.dataset_slug = identifier.split('/')
        with self.api.build_kaggle_client() as client:
            value = client.datasets.dataset_api_client.get_dataset(request)
        return {'ref': value.ref, 'is_private': value.is_private, 'version': value.current_version_number}

    def dataset_create_new(self, *args, **kwargs):
        # The SDK wraps CreateDataset in with_retry. Preserve the one-call
        # mutation rule while retaining resumable byte-upload behavior.
        original = self.api.with_retry
        try:
            self.api.with_retry = lambda function: function
            return self.api.dataset_create_new(*args, **kwargs)
        finally:
            self.api.with_retry = original


def validate_runtime(path):
    runtime = json.loads(Path(path).read_text())
    if (runtime.get('schema') != 'gemma4-kaggle-publication-runtime-v1'
            or any(version(name) != value for name, value in runtime['versions'].items())):
        raise ValueError('exact tested publication SDK versions required')
    pins = {}
    for package, files in runtime['source_sha256'].items():
        for name, expected in files.items():
            actual = Path(distribution(package).locate_file(name)).resolve()
            if sha256_file(actual) != expected: raise ValueError('tested publication API interface source changed')
            pins[str(actual)] = expected
    return runtime, pins


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['readiness', 'access', 'environments', 'pair', 'comparison', 'out']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--plan-only', action='store_true')
    args = p.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path): setattr(args, key, value.resolve())
    root = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != root: raise ValueError('run inside the frozen publication checkout')
    if args.out.exists(): raise FileExistsError('publication is immutable;reconcile existing intent and handles')
    files = ['scripts/publish_selected_candidate.py', 'scripts/prepare_kaggle_candidate.py',
        'research/kaggle-publication-runtime-v1.json',
        'src/gemma_particle/kaggle_publication.py', 'src/gemma_particle/selected_publication.py',
        'src/gemma_particle/confirmation.py', 'src/gemma_particle/confirmation_environments.py',
        'src/gemma_particle/confirmation_results.py', 'src/gemma_particle/selected_candidate.py',
        'src/gemma_particle/learned_effect.py']
    sources = {str(root / name): sha256_file(root / name) for name in files}
    declaration = {'schema': 'gemma4-learned-publication-plan-v1', 'created_at': datetime.now(timezone.utc).isoformat(),
        'sources': sources, 'model_inference': False, 'GPU': False,
        'mutation_policy': 'exclusive durable intent before one SDK invocation;ambiguous result stops for reconciliation',
        'scope': 'private payload and exact offline CPU notebook version,one learned competition request;score and fresh SOTA remain separate observations'}
    if args.plan_only:
        args.out.mkdir(parents=True)
        (args.out / 'plan.json').write_text(json.dumps(declaration, indent=2) + '\n')
        (args.out / 'progress.json').write_text(json.dumps({'state': 'prepared', 'protected_inputs_opened': False,
            'account_imported': False, 'account_requests': 0, 'model_inference': False}) + '\n')
        print(json.dumps({'state': 'prepared', 'protected_inputs_opened': False, 'account_requests': 0})); return
    spec = importlib.util.spec_from_file_location('fixed_payload_preparer', root / 'scripts/prepare_kaggle_candidate.py')
    preparer = importlib.util.module_from_spec(spec); spec.loader.exec_module(preparer)
    candidate = audit_selected_publication(args.readiness, args.access, args.environments, args.pair,
        args.comparison, root, preparer.notebook_source)
    subprocess.run(['git', 'diff', '--quiet', 'HEAD'], cwd=root, check=True)
    small = {path: value for path, value in candidate['files'].items() if not path.endswith('/agent_bundle.bin')}
    def verify():
        if any(sha256_file(path) != value for path, value in {**sources, **candidate['experiment_pins'], **small}.items()):
            raise ValueError('original publication source,experiment or prepared metadata changed')
    verify()
    # Authentication happens only after the complete scientific and payload audit.
    runtime, sdk_pins = validate_runtime(root / 'research/kaggle-publication-runtime-v1.json')
    sources.update(sdk_pins)
    from kaggle import api
    sdk_path = Path(__import__('kaggle.api.kaggle_api_extended', fromlist=['KaggleApi']).__file__).resolve()
    if str(sdk_path) not in sdk_pins or sha256_file(sdk_path) != sdk_pins[str(sdk_path)]:
        raise ValueError('authenticated SDK import must match the exact tested implementation')
    declaration.update(sdk_version=runtime['versions']['kaggle'], sdk_dependency_version=runtime['versions']['kagglesdk'], sdk_source_sha256=sha256_file(sdk_path),
        readiness_sha256=sha256_file(args.readiness), artifact_sha256=candidate['artifact_sha256'],
        dataset=candidate['dataset'], notebook=candidate['notebook'])
    if sha256_file(Path(candidate['payload']) / 'dataset/agent_bundle.bin') != candidate['wrapper_sha256']:
        raise ValueError('evaluated binary payload changed before publication')
    result = publish_once(KagglePublicationClient(api), candidate, args.out, verify, declaration=declaration)
    print(json.dumps({key: result[key] for key in ['state', 'submission_id', 'dataset_create_calls', 'notebook_push_calls', 'submission_calls']}))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # SDK exceptions may contain signed download URLs. Keep those out of
        # stdout; the durable operation record preserves safe failure metadata.
        print(json.dumps({'state': 'error', **safe_error(exc)}))
        raise SystemExit(1)
