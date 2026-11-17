"""Execution isolation for the mirrored sandbox's unkeyed wheel-tar cache."""
import os
from pathlib import Path
import tempfile
import json
import shlex

from swegemma.sandbox import ContainerManager,ExecResult


def isolate_sandbox_cache(directory):
    directory=Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=False,mode=0o700)
    # The mirror names its archive sp_base.tar without including wheel identity.
    # Use a private temp root; never delete another session's shared cache.
    os.environ['TMPDIR']=str(directory)
    tempfile.tempdir=str(directory)
    return directory


class SourceBoundManager(ContainerManager):
    """Refuse to score an installed package in place of the edited workspace."""
    def __init__(self,config,*,package):
        super().__init__(config)
        self.package=package
        self.origin_checks=[]

    def exec(self,container_id,command,*args,**kwargs):
        if 'python3 -s -m pytest ' in command:
            code=('import importlib.util,json,pathlib; s=importlib.util.find_spec('+repr(self.package)+'); '
                  'p=str(pathlib.Path(s.origin).resolve()) if s and s.origin else None; '
                  'print(json.dumps(dict(package='+repr(self.package)+',origin=p,workspace=bool(p and p.startswith("/workspace/")))))')
            observed=super().exec(container_id,'cd /workspace && PYTHONSAFEPATH=1 PYTHONNOUSERSITE=1 python3 -s -c '+shlex.quote(code))
            try:
                record=json.loads(observed.stdout)
            except (ValueError,TypeError):
                record={'package':self.package,'origin':None,'workspace':False}
            self.origin_checks.append(record)
            if observed.exit_code or not record['workspace']:
                return ExecResult(exit_code=1,stderr='Workspace import ownership check failed: '+json.dumps(record))
        return super().exec(container_id,command,*args,**kwargs)
