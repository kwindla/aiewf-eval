"""Narrow SGLang shim: honor an OpenAI request seed without deterministic mode.

The pinned SGLang 0.5.15.post1 image only populates
``SamplingBatchInfo.sampling_seed`` when the
server-wide ``enable_deterministic_inference`` option is true.  That option
also enables batch-invariant matrix kernels and disables insertion of finished
requests into the radix cache.  The latter makes it unsuitable for an
experiment whose treatment includes a warm radix-cache arm.

This shim changes the view of the server arguments in exactly one SGLang
module.  The sampler receives the per-request seed and uses SGLang's PyTorch
position-keyed sampling path, while the model, scheduler, and cache retain
their ordinary production behavior.  ``server.sh`` explicitly selects the
PyTorch sampling backend whenever it enables this shim.
"""

from __future__ import annotations

import os


if os.environ.get("SGLANG_HONOR_REQUEST_SEED_WITHOUT_BATCH_INVARIANCE") == "1":
    from sglang.srt.sampling import sampling_batch_info

    _real_get_global_server_args = sampling_batch_info.get_global_server_args

    class _SamplingArgsView:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        @property
        def enable_deterministic_inference(self) -> bool:
            return True

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    def _get_seed_aware_server_args():
        return _SamplingArgsView(_real_get_global_server_args())

    sampling_batch_info.get_global_server_args = _get_seed_aware_server_args
