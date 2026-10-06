#!/usr/bin/env bash
# Top-level launcher for the UltraFast v16b stack: it hands off to
# scripts/serve-ultrafast.sh, which loads config/ultrafast/env and builds the
# docker run command for the v16b image.
#
# Paths below are derived from this script's own location (REPO_ROOT), so the
# SAME committed file works on every machine with no local path edits: the
# checkpoints live under $REPO_ROOT/models on both the host clone and the
# Thor device clone.
#
# The served model name defaults to qwen38-flash-next for API compatibility
# with callers that already address that name; override SERVED_NAME (or
# MODEL_DIR / TABLE_DIR) in the environment to change any of them, since the
# ${VAR:-default} form keeps caller overrides.
#
# The previous, non-ultrafast stack remains available via
# scripts/serve-intel-ar.sh if you need it.
#
#   ./serve.sh --print    # print the command + env, launch nothing
#   ./serve.sh            # expand the vocab, then launch
cd "$(dirname "$0")"
REPO_ROOT="$PWD"

export MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-mtpdense-g32}"
export TABLE_DIR="${TABLE_DIR:-$REPO_ROOT/models/ple-table-fp8}"
export SERVED_NAME="${SERVED_NAME:-qwen38-flash-next}"

# Fail early with a clear message when the required artifacts are missing,
# except for --print: that path only reports the command and must work on a
# clone that has no checkpoints yet.
case " $* " in
  *" --print "*)
    ;;
  *)
    if [ ! -d "$MODEL_DIR" ]; then
      echo "serve.sh: MODEL_DIR not found: $MODEL_DIR" >&2
      echo "serve.sh: build the prepared checkpoint with tools/ultrafast-build/build.sh first" >&2
      exit 1
    fi
    if [ ! -d "$TABLE_DIR" ]; then
      echo "serve.sh: TABLE_DIR not found: $TABLE_DIR" >&2
      echo "serve.sh: fetch the PLE/ngram table as described in the README" >&2
      exit 1
    fi
    ;;
esac

exec scripts/serve-ultrafast.sh "$@"
