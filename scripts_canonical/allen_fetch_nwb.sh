#!/bin/bash
# fetch the 32 Allen VC Neuropixels session NWBs used by the paper into raw/allen_nwb (never backed up).
# Public S3; curl with --max-time per file and 3 retries; at most 4 in parallel; skips files already present.
cd "$(dirname "$0")/.."
D=raw/allen_nwb; mkdir -p "$D"
S3=https://allen-brain-observatory.s3.us-west-2.amazonaws.com/visual-coding-neuropixels/ecephys-cache
fetch() {
  local s=$1
  [ -f "$D/session_$s.nwb" ] && { echo "HAVE $s"; return; }
  if curl -sS --fail --retry 3 --max-time 3600 -o "$D/session_$s.nwb.part" "$S3/session_$s/session_$s.nwb"; then
    mv "$D/session_$s.nwb.part" "$D/session_$s.nwb"; echo "DONE $s"
  else echo "FAIL $s"; fi
}
for s in $(python3 -c "import json;print(' '.join(str(s['session']) for s in json.load(open('data_canonical/allen_expansion_all_sessions.json'))['results']))"); do
  while [ "$(jobs -r | wc -l)" -ge 4 ]; do sleep 2; done
  fetch "$s" &
done
wait
echo ALL_FETCHED
