FROM gemma-particle-sandbox:fastapi-deps-v2
COPY *.whl /tmp/json-test-wheels/
RUN python -m pip install --no-index --no-deps /tmp/json-test-wheels/*.whl && rm -rf /tmp/json-test-wheels
