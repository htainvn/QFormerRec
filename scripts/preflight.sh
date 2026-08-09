#!/usr/bin/env bash
# What is actually about to run / what actually ran.
#
# Every "my change did not take effect" in this project traced to one of three
# things: Colab sitting on an older checkout, a log file that was never
# refreshed, or a config edit that was proposed but never applied. This prints
# all three in one shot so none of them costs a training run.
#
#   bash scripts/preflight.sh train_configs/stage2_qformer_ml1m.yaml
#   bash scripts/preflight.sh train_configs/stage2_qformer_ml1m.yaml /content/logs/stage2_ml1m/log.txt
#
# With a log path it also prints the runtime markers -- those are the only
# authority, because the config on disk is not necessarily what the job loaded.

set -uo pipefail
CFG="${1:-}"
LOG="${2:-}"

if [ -z "$CFG" ]; then
    echo "usage: bash scripts/preflight.sh <config.yaml> [logfile]" >&2
    exit 2
fi

hr() { printf '=== %s %s\n' "$1" "$(printf '%.0s-' $(seq 1 $((56 - ${#1}))))"; }

hr "checkout"
git log --oneline -1 2>/dev/null || echo "  (not a git checkout)"
if git rev-parse --abbrev-ref @{u} >/dev/null 2>&1; then
    behind=$(git rev-list --count HEAD..@{u} 2>/dev/null || echo 0)
    ahead=$(git rev-list --count @{u}..HEAD 2>/dev/null || echo 0)
    if [ "$behind" != "0" ]; then
        echo "  !! $behind commit(s) BEHIND remote -- run: git pull"
    elif [ "$ahead" != "0" ]; then
        echo "  !! $ahead commit(s) ahead of remote -- push before running elsewhere"
    else
        echo "  up to date with remote"
    fi
fi
dirty=$(git status --porcelain -- "$CFG" prompts/ 2>/dev/null)
[ -n "$dirty" ] && echo "  !! uncommitted changes:" && echo "$dirty" | sed 's/^/     /'

hr "config: $CFG"
grep -nE '^  (arch|prompt_path|n_titles_kept|freeze_rec|freeze_proj|freeze_lora|init_lr|min_lr|max_epoch|iters_per_epoch|warmup_steps|patience):' "$CFG"
echo "  --- loss ---"
grep -nE '^    lambda_[a-z]+:' "$CFG"
echo "  --- lr_scale ---"
grep -nE '^    (qformer|proj|proto|rec|lora):' "$CFG"

PROMPT=$(grep -E '^  prompt_path:' "$CFG" | sed -E 's/.*"(.*)".*/\1/')
if [ -n "$PROMPT" ] && [ -f "$PROMPT" ]; then
    hr "prompt: $PROMPT"
    n_pref=$(grep -c '<PrefTokens>' "$PROMPT" || true)
    nq=$(grep -E '^    n_query:' "$CFG" | grep -oE '[0-9]+' | head -1)
    ptn=$(grep -E '^  proj_token_num:' "$CFG" | grep -oE '[0-9]+' | head -1)
    expect=0
    [ "$n_pref" != "0" ] && expect=$((expect + ${nq:-0}))
    grep -q '<UserID>' "$PROMPT"       && expect=$((expect + ${ptn:-0}))
    grep -q '<TargetItemID>' "$PROMPT" && expect=$((expect + ${ptn:-0}))
    head -1 "$PROMPT" | fold -s -w 76 | sed 's/^/  /'
    echo "  -> expect exactly $expect <unk> in the runtime 'prompt example:' line"
fi

if [ -n "$LOG" ]; then
    hr "runtime (the only authority)"
    if [ ! -f "$LOG" ]; then
        echo "  !! no such log: $LOG"
        exit 1
    fi
    echo "  log mtime: $(date -r "$LOG" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || stat -c %y "$LOG")"
    grep -m1 "trainable parameters:" "$LOG" | sed 's/^/  /'
    grep -m1 "^lr_scale:" "$LOG" | sed 's/^/  /'
    ex=$(grep -m1 "prompt example:" "$LOG")
    if [ -n "$ex" ]; then
        got=$(printf '%s' "$ex" | grep -o '<unk>' | wc -l | tr -d ' ')
        echo "  prompt example has $got <unk>"
        printf '%s\n' "$ex" | fold -s -w 76 | sed 's/^/    /'
    fi
    grep -oE 'prompt_tokens: [0-9.]+' "$LOG" | head -1 | sed 's/^/  /'
fi
