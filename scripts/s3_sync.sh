#!/usr/bin/env bash
# Push / pull QFormerRec artifacts to S3-compatible storage.
#
#   export ENDPOINT_URL=https://<your-endpoint>
#   export BUCKET=s3://qformerrec
#   export RUN=ml1m-pit-k50            # run tag, like the `sella-aug` slot
#
#   scripts/s3_sync.sh up              # push ckpt/ + logs/
#   scripts/s3_sync.sh down            # restore both after a runtime restart
#   scripts/s3_sync.sh up   stage2     # just one stage
#   scripts/s3_sync.sh down ckpt
#   DRYRUN=1 scripts/s3_sync.sh up     # show what would move
#
# Deliberately never touches the Vicuna weights (13.5 GB, re-downloadable) or the
# datasets. Sync BETWEEN stages, not mid-training: checkpoint_best.pth is rewritten
# in place whenever validation UAUC improves, so a mid-write copy can be truncated.
set -euo pipefail

: "${ENDPOINT_URL:?set ENDPOINT_URL}"
: "${BUCKET:=s3://qformerrec}"
: "${RUN:?set RUN, e.g. RUN=ml1m-pit-k50}"
CKPT_DIR="${CKPT_DIR:-/content/ckpt}"
LOGS_DIR="${LOGS_DIR:-/content/logs}"
DRY=""; [ "${DRYRUN:-0}" = "1" ] && DRY="--dryrun"

DIR=${1:-up}
WHAT=${2:-all}
S3="aws --endpoint-url=$ENDPOINT_URL s3"
EXCL=(--exclude "*/result/*" --exclude "*.tmp" --exclude "*/vicuna*/*")

sync_pair() {           # $1 local  $2 remote-suffix
    local local_path="$1" remote="$BUCKET/$2/$RUN/"
    if [ "$DIR" = "up" ]; then
        [ -e "$local_path" ] || { echo "  skip (absent): $local_path"; return 0; }
        echo "  UP   $local_path  ->  $remote"
        $S3 sync "$local_path" "$remote" "${EXCL[@]}" $DRY
    else
        echo "  DOWN $remote  ->  $local_path"
        mkdir -p "$local_path"
        $S3 sync "$remote" "$local_path" $DRY
    fi
}

echo "endpoint=$ENDPOINT_URL bucket=$BUCKET run=$RUN direction=$DIR what=$WHAT${DRY:+ (dry run)}"
case "$WHAT" in
    all)   sync_pair "$CKPT_DIR" ckpt; sync_pair "$LOGS_DIR" logs ;;
    ckpt)  sync_pair "$CKPT_DIR" ckpt ;;
    logs)  sync_pair "$LOGS_DIR" logs ;;
    stage1|stage2|stage3|eval)
        # one stage at a time, both datasets if present
        for ds in ml1m amazon; do
            d="$LOGS_DIR/${WHAT}_${ds}"
            [ "$DIR" = "down" ] || [ -e "$d" ] || continue
            sync_pair "$d" "logs/${WHAT}_${ds}"
        done ;;
    *) echo "unknown target '$WHAT' (all|ckpt|logs|stage1|stage2|stage3|eval)"; exit 2 ;;
esac
echo "done"
