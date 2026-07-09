#!/bin/bash
# Download the 10 verified-indoor SCAND Spot bags (~9.4 GB) from TDL Dataverse.
# The server 403s non-browser user agents; -C - makes downloads resumable.
# Selection: indoor-confirmed via preview videos, 2026-07-08 (see TASKS.md T13).
UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
cd "$(dirname "$0")/../data/scand_bags" || { mkdir -p "$(dirname "$0")/../data/scand_bags" && cd "$(dirname "$0")/../data/scand_bags"; }
while read -r id name; do
  if [ ! -f "$name.done" ]; then
    curl -sL -A "$UA" -C - -o "$name" "https://dataverse.tdl.org/api/access/datafile/$id" && touch "$name.done"
  fi
  echo "$(date +%H:%M:%S) done $name"
done <<'LIST'
138144 A_Spot_JCL_JCL_Wed_Nov_10_65.bag
138153 A_Spot_JCL_JCL_Wed_Nov_10_66.bag
138158 A_Spot_Dobie_Dobie_Thu_Nov_11_73.bag
138132 A_Spot_Union_Union_Wed_Nov_10_53.bag
138152 A_Spot_Union_Union_Wed_Nov_10_67.bag
136676 B_Spot_SAC_SAC_Wed_Nov_10_49.bag
138146 A_Spot_Jester_Jester_Wed_Nov_10_63.bag
138147 A_Spot_Jester_Jester_Wed_Nov_10_64.bag
138156 B_Spot_JCL_JCL_Thu_Nov_11_76.bag
138193 B_Spot_SAC_SAC_Mon_Nov_15_109.bag
LIST
echo ALL_DONE
