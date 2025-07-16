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
    bash ./scripts/ADSB/LGTDM.sh --missing_ratio $m &&
    bash ./scripts/ADSB/BRITS.sh --missing_ratio $m &&
    bash ./scripts/ADSB/GAIN.sh --missing_ratio $m &&
    bash ./scripts/ADSB/SAITS.sh --missing_ratio $m &&
    bash ./scripts/ADSB/CSDI.sh --missing_ratio $m &&
    bash ./scripts/ADSB/SSSD.sh --missing_ratio $m
else
    echo "Invalid missing ratio: $m"
    exit 1
fi
