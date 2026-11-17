# A declared local environment derivative, not a production scorer image.
# Adds dependencies named by Requests' own requirements-dev.txt. No test,
# network, clock, certificate-validation or scoring changes are made.
FROM gemma-particle-sandbox:mirror-v30
RUN python -m pip install --no-cache-dir \
    pytest-httpbin==2.0.0 'httpbin~=0.10.0' trustme pytest-cov PySocks pytest-mock \
    && python -m pip freeze > /requests-control-packages.txt
