#!/usr/bin/env bash
# Run the whole pipeline on this machine, with nothing uploaded anywhere. Outputs stay under out/.
#
#   ./run_local.sh extract sampled            # the production sample: every 2nd day of the last 12 months
#   ./run_local.sh extract pilot 2026-09      # every day of one month
#   ./run_local.sh extract dates 2026-09-01,2026-09-05
#   ./run_local.sh pilot 2026-09              # sampled-vs-full check on that month -> out/pilot/sampling_check.json
#   ./run_local.sh grid 2026-09               # build the sample, validate against the pilot of that month
#   ./run_local.sh upload                     # push out/grid/<latest> to the private bucket (needs FLIGHT_GRID_S3_*)
#
# Extraction skips days already in out/extract, so it can be stopped and resumed. A laptop does
# about one day in 6 to 8 minutes, so JOBS=3 (default) gives roughly 25 days an hour if the
# connection holds. STRIDE=4 and JOBS=3 can be overridden in the environment.
set -euo pipefail
cd "$(dirname "$0")"

STRIDE="${STRIDE:-4}"
JOBS="${JOBS:-3}"
PY=(uv run python)

# macOS ships bash 3.2: no mapfile, and "${empty[@]}" is an error under set -u.
read_days() { days=(); while IFS= read -r line; do if [ -n "$line" ]; then days+=("$line"); fi; done < <("$@"); }

extract_days() {
  local todo=()
  for d in "$@"; do
    [ -f "out/extract/${d:0:4}/$d.parquet" ] || todo+=("$d")
  done
  echo "${#todo[@]} of $# days to extract (JOBS=$JOBS)"
  [ "${#todo[@]}" -eq 0 ] && return 0
  # Each day gets its own out dir (and manifest) so parallel jobs never race on days_manifest.json;
  # the results are folded into out/extract and out/days_manifest.json afterwards.
  printf '%s\n' "${todo[@]}" | xargs -P "$JOBS" -I{} bash -c '
    d={}; tmp="out/tmp/$d"; mkdir -p "$tmp"
    if uv run python extract_day.py --date "$d" --out "$tmp" --workers 4 > "$tmp/log.txt" 2>&1; then
      mkdir -p "out/extract/${d:0:4}" out/extract/manifests
      mv "$tmp/extract/${d:0:4}/$d.parquet" "out/extract/${d:0:4}/$d.parquet"
      mv "$tmp/days_manifest.json" "out/extract/manifests/$d.json"
      echo "done   $d"
    else
      echo "FAILED $d (see $tmp/log.txt)"
    fi'
  "${PY[@]}" merge_manifests.py out/extract/manifests out/days_manifest.json
}

case "${1:-}" in
  extract)
    case "${2:-}" in
      sampled) read_days python3 plan_days.py sampled --stride "$STRIDE" --lines ;;
      pilot)   read_days python3 plan_days.py pilot --month "${3:?month YYYY-MM}" --lines ;;
      dates)   read_days python3 plan_days.py dates "${3:?comma-separated dates}" --lines ;;
      *) echo "usage: $0 extract sampled|pilot MONTH|dates LIST" >&2; exit 2 ;;
    esac
    extract_days "${days[@]}"
    ;;
  pilot)
    "${PY[@]}" sample_check.py --month "${2:?month YYYY-MM}" --stride "$STRIDE" --extract-dir out/extract --work out/pilot
    ;;
  grid)
    read_days python3 plan_days.py sampled --stride "$STRIDE" --lines
    present=()
    for d in "${days[@]}"; do [ -f "out/extract/${d:0:4}/$d.parquet" ] && present+=("$d"); done
    echo "${#present[@]} of ${#days[@]} sampled days are extracted"
    [ "${#present[@]}" -gt 0 ] || { echo "nothing to build; run: $0 extract sampled" >&2; exit 1; }
    rm -rf out/grid-build; mkdir -p out/grid-build
    "${PY[@]}" build_grid.py --extract-dir out/extract --out out/grid-build --cache out/agg \
      --days "$(IFS=,; echo "${present[*]}")" --day-stride "$STRIDE" --workers 4
    grid="$(ls -d out/grid-build/*/ | tail -1)"; grid="${grid%/}"
    args=(--extract-dir out/extract)
    [ -f out/pilot/sampling_check.json ] && args+=(--sampling-check out/pilot/sampling_check.json)
    if "${PY[@]}" validate_grid.py "$grid" "${args[@]}"; then
      mkdir -p out/grid && rm -rf "out/grid/$(basename "$grid")" && mv "$grid" out/grid/
      echo "validated build: out/grid/$(basename "$grid")"
    else
      echo "validation FAILED; the build stays in $grid and is not eligible for upload" >&2
      exit 1
    fi
    ;;
  upload)
    grid="$(ls -d out/grid/*/ | tail -1)"; grid="${grid%/}"
    "${PY[@]}" - "$grid" <<'PYEOF'
import json, sys
v = json.load(open(sys.argv[1] + "/validation.json"))
sys.exit(0 if v.get("passed") is True else "refusing to upload: validation.json is not a pass")
PYEOF
    "${PY[@]}" store.py put-dir "$grid" "grid/$(basename "$grid")"
    ;;
  *)
    sed -n '2,13p' "$0" >&2
    exit 2
    ;;
esac
