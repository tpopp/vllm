#!/usr/bin/env bash
# Populate vllm_dsv41_mono/kernels/ with the kernel package of vllm#60397
# (vllm/models/deepseek_v41/amd/mono/, adapted from ROCm/ATOM 97359d6, MIT),
# unchanged except for its absolute import path. Run from a vLLM checkout that
# has the PR's branch fetched, e.g.:
#   git fetch https://github.com/vllm-project/vllm pull/60397/head:pr60397
#   ./vendor_kernels.sh pr60397
set -euo pipefail
ref="${1:-pr60397}"
here="$(cd "$(dirname "$0")" && pwd)"
dst="$here/vllm_dsv41_mono/kernels"
src="vllm/models/deepseek_v41/amd/mono"
rm -rf "$dst" && mkdir -p "$dst"
git ls-tree -r --name-only "$ref" "$src" | while read -r f; do
  out="$dst/${f#"$src"/}"
  mkdir -p "$(dirname "$out")"
  git show "$ref:$f" \
    | sed 's/vllm\.models\.deepseek_v41\.amd\.mono/vllm_dsv41_mono.kernels/g' > "$out"
done
# The package must not import vLLM: that is the property that makes it a
# provider and not a fork. #60397's package already satisfies it.
if grep -rnE '^\s*(from|import) vllm(\.|\s|$)' "$dst"; then
  echo "kernel package imports vllm" >&2; exit 1
fi
echo "vendored $(find "$dst" -name '*.py' | wc -l) files from $ref"
