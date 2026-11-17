import io
import zipfile
import pytest
from gemma_particle.remote_zip import RangeFile


class Response:
    def __init__(self, content, start, end, *, ignored=False):
        self.status_code = 200 if ignored else 206
        self.headers = {} if ignored else {'Content-Range':f'bytes {start}-{end}/{len(content)}'}
        self.raw = io.BytesIO(content if ignored else content[start:end+1])
    def __enter__(self): return self
    def __exit__(self, *args): pass


def test_zip_seek_and_range_refusal(monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer,'w') as archive:
        archive.writestr('tasks.jsonl', '{"instance_id":"task"}\n')
    content = buffer.getvalue()
    reader = RangeFile('https://example.invalid/archive',len(content),block_size=32)
    def get(url, *, params, headers, stream, timeout):
        start,end = map(int,headers['Range'].removeprefix('bytes=').split('-'))
        return Response(content,start,end)
    monkeypatch.setattr(reader.session,'get',get)
    with zipfile.ZipFile(reader) as archive:
        assert archive.read('tasks.jsonl').startswith(b'{')
    assert reader.transferred <= len(content)
    refused = RangeFile('https://example.invalid/archive',len(content))
    monkeypatch.setattr(refused.session,'get',lambda *a,**kw:Response(content,0,len(content)-1,ignored=True))
    with pytest.raises(IOError, match='exact Range'):
        refused.read(1)
