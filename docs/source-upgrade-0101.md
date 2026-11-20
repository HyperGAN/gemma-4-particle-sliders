# ParticleGAN 0.10.1 source cohort

This historical qualification used a source lock that pinned ParticleGAN `53f5ff3757334f02341df66d107d27db900f4a5d` (0.10.1) and Particle Sliders `119d7e0559bacdc49847961c1b243ac08daa013c`. The previous [0.10.0 lock](../research/sources-0.10.0-af0764b1.lock.json) and bridge patch remain available for historical receipts. Existing collection processes use their frozen checkout and original environment.

The upstream runtime change selects EMPF when `get_recipe()` has no name. Both task recipes already use explicit names. The task's deterministic R2 particle initializer now explicitly requests the RHBI prior holder used previously, so a default change cannot alter the task architecture. The explicit RHBI and EMPF implementations are unchanged. The latest pristine Sliders tree matches the previous tree; the new overlay changes only source/version metadata and explanatory strings in the shared conditional engine. Normalizing those strings reproduces the previous engine exactly.

Training uses a separate `.train-venv` and separate dependency checkouts. Run `scripts/bootstrap.py` with that environment. This host reuses ordinary third-party packages through a plain site-packages path, without processing the old environment's editable-package hooks. Editable ParticleGAN, Sliders and this project belong to the new environment. A clean installation can instead install `requirements-model.txt` directly, as shown in the README. Native construction checks the version, commit, shared engine, manifest and all 87 ParticleGAN Python source hashes before model loading or mutation.

The [qualification receipt](../reports/pg0101-source-qualification-v1.json) records:

- 57 CPU checks and successful official submission compilation.
- Exact tensor equality for initialization and one optimizer update, separately for `rhbi_optimized` and `empf`, between the old and new source cohorts. Each comparison covers 386 tensors and matching numeric metrics on the same random Gemma fixture.
- Rejection of a full 0.10.0 checkpoint by each new game, with every live state value unchanged after rejection.
- Successful bootstrap of the new environment while the old environment continues to import 0.10.0.

These are source and continuation controls on random CPU fixtures. They supply no training targets or pretrained task score. A new trained cohort must start fresh under 0.10.1; old optimizer/RNG checkpoints are never relabeled or migrated.
