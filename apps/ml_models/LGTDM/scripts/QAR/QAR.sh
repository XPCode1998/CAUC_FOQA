#!/bin/bash
m=0.1

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --m) m="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [[ "$m" == "0.1" || "$m" == "0.2" || "$m" == "0.5" || "$m" == "0.9" ]]; then
    bash ./scripts/QAR/BRITS.sh --missing_ratio $m &&
    bash ./scripts/QAR/SAITS.sh --missing_ratio $m &&
    bash ./scripts/QAR/CSDI.sh --missing_ratio $m 
else
    echo "Invalid missing ratio: $m"
    exit 1
fi
