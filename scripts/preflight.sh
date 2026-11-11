#!/usr/bin/env bash
# Run before every commit:  bash scripts/preflight.sh
# Checks the STAGED changes. Stage first with named paths (never git add -A).
#
# Extra patterns that must stay out of the repo (for example employer or
# university names) go in .git/info/preflight-patterns, one extended-regex
# pattern per line. That file is local and never committed.
set -u
fail=0

echo "== staged files"
git diff --staged --stat

echo "== banned file types or private paths"
bad=$(git diff --staged --name-only --diff-filter=ACMR | grep -E \
  '(\.(png|jpe?g|gif|bmp|zip|tar|gz|pt|pth|onnx|task|h5|ckpt|safetensors)$)|(^data/private/[^R])|(^data/raw/)|(^data/cache/)|(^private/)' \
  || true)
if [ -n "$bad" ]; then
  echo "BLOCKED (images, archives, model files or private data):"
  echo "$bad"
  fail=1
fi

echo "== large staged files (over 1 MB)"
for f in $(git diff --staged --name-only --diff-filter=ACMR); do
  if [ -f "$f" ] && [ "$(wc -c < "$f")" -gt 1048576 ]; then
    echo "LARGE: $f"
    fail=1
  fi
done

echo "== sensitive words in the staged diff"
pattern='secret|token|api_key|password'
if [ -f .git/info/preflight-patterns ]; then
  extra=$(paste -sd'|' .git/info/preflight-patterns)
  [ -n "$extra" ] && pattern="$pattern|$extra"
else
  echo "note: .git/info/preflight-patterns not found, so only the generic words are checked"
fi
# This script lists the generic words itself, so it is left out of the scan.
hits=$(git diff --staged -- . ':(exclude)scripts/preflight.sh' | grep -v '^+++ ' | grep '^+' | grep -inwE "$pattern" || true)
if [ -n "$hits" ]; then
  echo "BLOCKED (review these lines):"
  echo "$hits"
  fail=1
fi

if [ "$fail" -eq 0 ]; then echo "preflight passed"; else echo "preflight FAILED"; fi
exit "$fail"
