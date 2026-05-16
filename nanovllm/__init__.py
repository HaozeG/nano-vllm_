import os

# Keep normal users and benchmark/eval jobs from accidentally inheriting
# ablation or profiling flags from a dirty shell. These flags are read by
# submodules at import time, so this must happen before importing LLM. Set
# NANOVLLM_ALLOW_ABLATION_ENV=1 for intentional ablation/profiling runs.
_NANOVLLM_ABLATION_ENV_VARS = (
    "NANOVLLM_DISABLE_CUDA_GRAPH",
    "NANOVLLM_DISABLE_TRITON_RMSNORM",
    "NANOVLLM_DISABLE_TRITON_MOE",
    "NANOVLLM_DISABLE_VECTOR_GATHER",
    "NANOVLLM_PROFILE_MOE",
    "NANOVLLM_PROFILE_LAYER",
    "NANOVLLM_PROFILE_ATTN",
    "NANOVLLM_PROFILE_ATTN_DETAIL",
    "NANOVLLM_PROFILE_SYNC",
)
if os.getenv("NANOVLLM_ALLOW_ABLATION_ENV", "0") != "1":
    for _name in _NANOVLLM_ABLATION_ENV_VARS:
        os.environ.pop(_name, None)

from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams
