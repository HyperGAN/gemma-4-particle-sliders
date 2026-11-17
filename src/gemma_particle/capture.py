"""Immutable model-boundary journals. Transport credentials are never selected."""
from copy import deepcopy
import json
from pathlib import Path

from .data import digest

GENERATION_FIELDS = frozenset({'temperature','top_p','top_k','seed','max_tokens','max_completion_tokens',
    'stop','response_format','presence_penalty','frequency_penalty','tool_choice','parallel_tool_calls',
    'extra_body','reasoning_effort','thinking','chat_template_kwargs'})
LEGACY_CAPTURE_PROTOCOL = 'complete-or-timeout-v3'
CONTEXT_CAPTURE_PROTOCOL = 'context-budget-off-v1'
CAPTURE_PROTOCOLS = (LEGACY_CAPTURE_PROTOCOL,CONTEXT_CAPTURE_PROTOCOL)


class RequestJournal:
    def __init__(self, directory, *, instance_id, arm):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True,exist_ok=False,mode=0o700)
        self.instance_id, self.arm, self.next_index = instance_id, arm, 0

    def begin(self, *, model, messages, tools, options, purpose=None):
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
        if purpose is not None:
            if purpose!='history_summary':raise ValueError('unknown capture purpose')
            packet['purpose']=purpose
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


def _request(path,index):
    packet=json.loads(path.read_text())
    if packet['decision']!=index or packet['request_sha256']!=digest(packet['request']):
        raise ValueError('request journal order or identity changed')
    return packet


def _response(path,packet):
    completed=json.loads(path.read_text())
    if (completed['request_sha256']!=packet['request_sha256']
            or completed['response_sha256']!=digest(completed['response'])):
        raise ValueError('completion journal identity changed')


def load_requests(directory):
    result = []
    for path in sorted(Path(directory).glob('*-request.json')):
        packet=_request(path,len(result))
        response = path.with_name(path.name.replace('-request.json','-response.json'))
        if not response.is_file():
            raise ValueError('incomplete model request')
        _response(response,packet)
        result.append(packet)
    if not result:
        raise ValueError('no complete model requests')
    return result


def load_terminal_request(directory):
    """Validate a complete prefix followed by exactly one unanswered request."""
    directory=Path(directory);paths=sorted(directory.glob('*-request.json'))
    if not paths:raise ValueError('terminal request required')
    prefix=[]
    for index,path in enumerate(paths):
        packet=_request(path,index)
        response=path.with_name(path.name.replace('-request.json','-response.json'))
        if index<len(paths)-1:
            if not response.exists():raise ValueError('incomplete prefix before terminal request')
            _response(response,packet);prefix.append(packet)
        elif response.exists():raise ValueError('terminal request already has a response')
    if len(list(directory.glob('*-response.json')))!=len(prefix):
        raise ValueError('unexpected response outside the complete prefix')
    if any(p['instance_id']!=packet['instance_id'] or p['arm']!=packet['arm'] for p in prefix):
        raise ValueError('mixed request task or arm')
    return prefix,packet
