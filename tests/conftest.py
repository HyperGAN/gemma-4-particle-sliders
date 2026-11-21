from pathlib import Path

import pytest
import torch


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture(scope="session")
def archived_evidence():
    from gemma_particle.archive_evidence import ARCHIVED_FILES, restore_file

    checkout = Path(__file__).resolve().parents[1]
    for relative in ARCHIVED_FILES:
        restore_file(checkout, relative, allow_network=False)
