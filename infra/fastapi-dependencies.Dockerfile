# Adds public test dependencies missing from the mirrored task wheel directory.
FROM gemma-particle-sandbox:source-v1
RUN python -m pip install --no-cache-dir typing-inspection dirty-equals inline-snapshot 'python-multipart>=0.0.18,<1' \
    && python -m pip freeze > /fastapi-control-packages.txt
