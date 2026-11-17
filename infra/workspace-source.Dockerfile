# Source-import repair for local public-data controls. No test/scorer changes.
FROM gemma-particle-sandbox:requests-deps-v1
COPY workspace-source.pth /usr/local/lib/python3.13/site-packages/zz_workspace_source.pth
