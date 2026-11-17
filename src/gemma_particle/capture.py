"""Immutable model-boundary journals. Transport credentials are never selected."""
from copy import deepcopy
import json
from pathlib import Path

from .data import digest

GENERATION_FIELDS = frozenset({'temperature','top_p','top_k','seed','max_tokens','max_completion_tokens',
    'stop','response_format','presence_penalty','frequency_penalty','tool_choice','parallel_tool_calls',
    'extra_body','reasoning_effort','thinking','chat_template_kwargs'})


class RequestJournal:
    def __init__(self, directory, *, instance_id, arm):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True,exist_ok=False,mode=0o700)
        self.instance_id, self.arm, self.next_index = instance_id, arm, 0

    def begin(self, *, model, messages, tools, options):
        if options.get('stream'):
            raise ValueError('this capture cohort requires non-streaming model calls')
        # Deliberately exclude api_key, headers, api_base and provider internals.
        # Only model input and generation controls enter this research artifact.
        request = {'model':model, 'messages':deepcopy(messages), 'tools':deepcopy(tools),
                   'generation':{k:deepcopy(v) for k,v in options.items() if k in GENERATION_FIELDS}}
        def reject_credentials(value):
            if isinstance(value,dict):
                if set(value) & {'api_key','authorization','Authorization','access_token','password','secret','headers'}:
                    raise ValueError('transport credentials must not appear in generation controls')
                for v in value.values(): reject_credentials(v)
            elif isinstance(value,list):
                for v in value: reject_credentials(v)
        reject_credentials(request['generation'])
        packet = {'schema':'gemma4-model-request-v1', 'instance_id':self.instance_id,
                  'arm':self.arm, 'decision':self.next_index, 'request':request, 'request_sha256':digest(request)}
        path = self.directory/f'{self.next_index:04d}-request.json'
        path.write_text(json.dumps(packet,ensure_ascii=False,indent=2)+'\n')
        path.chmod(0o600)
        self.next_index += 1
        return packet

    def complete(self, packet, response):
        # Keep only the actual assistant completion and usage, not hidden
        # provider metadata that may contain credentials or HTTP headers.
        if hasattr(response,'model_dump'):
            response = response.model_dump(mode='json')
        message_fields = {'role','content','tool_calls','function_call','reasoning_content','reasoning','refusal'}
        choices = [{'index':c.get('index',i), 'finish_reason':c.get('finish_reason'),
                    'message':{k:deepcopy(v) for k,v in c['message'].items() if k in message_fields}}
                   for i,c in enumerate(response['choices'])]
        selected = {'choices':choices, 'usage':deepcopy(response.get('usage'))}
        result = {'schema':'gemma4-model-response-v1', 'request_sha256':packet['request_sha256'],
                  'response':selected, 'response_sha256':digest(selected)}
        path = self.directory/f'{packet["decision"]:04d}-response.json'
        if path.exists():
            raise FileExistsError('completion already recorded')
        path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        path.chmod(0o600)


def load_requests(directory):
    result = []
    for path in sorted(Path(directory).glob('*-request.json')):
        packet = json.loads(path.read_text())
        if packet['decision'] != len(result) or packet['request_sha256'] != digest(packet['request']):
            raise ValueError('request journal order or identity changed')
        response = path.with_name(path.name.replace('-request.json','-response.json'))
        if not response.is_file():
            raise ValueError('incomplete model request')
        completed = json.loads(response.read_text())
        if (completed['request_sha256'] != packet['request_sha256']
                or completed['response_sha256'] != digest(completed['response'])):
            raise ValueError('completion journal identity changed')
        result.append(packet)
    if not result:
        raise ValueError('no complete model requests')
    return result
