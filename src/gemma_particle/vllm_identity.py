"""Source-bound request guard for the unmodified official vLLM server."""
import json
import os
from pathlib import Path

from starlette.responses import JSONResponse

from .data import digest


async def guard(request, call_next):
    identity=json.loads(Path(os.environ['GEMMA_SERVER_IDENTITY']).read_text())
    expected=digest(identity)
    if request.url.path.startswith('/v1/') and request.headers.get('X-Gemma-Identity')!=expected:
        return JSONResponse({'detail':'request must name this exact server identity'},status_code=412)
    response=await call_next(request)
    if request.url.path=='/health' and response.status_code==200:
        return JSONResponse({'status':'ok','identity_sha256':expected,'model_identity':identity['model_identity']})
    response.headers['X-Gemma-Identity']=expected
    return response
