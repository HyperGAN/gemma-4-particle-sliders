# Checkpoint input lifetime

The original CPU-packed/nested profile failed its first gradient-enabled full-history forward at both [.40](../reports/cpu-first-packed-040-forward-error-v1.json) and [.50](../reports/cpu-first-packed-050-forward-error-v1.json) allocator caps. The frozen OFF forward completed. Neither failed run completed the pair of ON forwards, a full gradient qualification or an optimizer update.

Inspection of the installed PyTorch checkpoint implementation shows that positional tensor inputs enter its saved-tensor mechanism, while keyword inputs remain in the recomputation closure. Gemma calls attention with `hidden_states` as a keyword. Nesting the public checkpoint around that attention therefore retains a normalized hidden tensor per layer even when an outer saved-tensor hook copies decoder inputs to CPU.

A [CPU lifetime control](../reports/positional-checkpoint-cpu-v1.json) reproduces this distinction with a nonleaf tensor: the ordinary keyword call retains the input object after the caller releases it; a positional call releases it. Both produce exactly the same input gradient.

The distinct [positional profile](../configs/31b-ct-shared-positional-pilot.json) binds the original forward signature before entering the unchanged public `checkpoint_with_adapter_scale` wrapper. Declared positional-or-keyword arguments become positional checkpoint inputs. Their values and order at the original function are unchanged. Container objects, including shared-KV `UserDict` instances, keep their identity and mutations; they are not flattened or copied. Actual keyword-only/variadic keyword fields retain their original calling convention.

The wrapper has an owned, reversible close handle. It preserves the public scale-capturing recomputation contexts and adds the new profile to native metadata and the full checkpoint signature. Old block and original nested profiles remain separate and executable for reproduction. No PG objective, optimizer, adapter architecture, token, batch or hidden-vector dimension changes.

All [91 CPU checks and the official compiler](../reports/positional-checkpoint-preflight-v1.json) pass. Six [tiny CUDA controls](../reports/positional-packed-cuda-v1.json) cover CPU/device packed storage and direct/original-nested/positional-nested checkpoints, with exact output and full adapter-gradient equality against public dense decompression. Shared-KV CPU gradients, deferred strength, both recipes' exact updates/continuation, incompatible-profile rejection and wrapper cleanup also pass.

The next full-history probe uses the original .40 cap, two complete 28,672-token sequences, CPU packed constants and CPU saved inputs. Its outcome remains pending. It is a synthetic derivative/resource check, performs zero optimizer updates and cannot supply targets or a coding score.
