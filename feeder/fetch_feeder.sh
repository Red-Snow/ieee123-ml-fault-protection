#!/usr/bin/env bash
# Download the IEEE 123-node test feeder (EPRI OpenDSS model, BSD licence) from the
# DSS-Extensions test-case repository at a pinned commit, and verify checksums.
set -euo pipefail
cd "$(dirname "$0")"
C=3b208397160213cae4a9e2d0a7d1aa3528ce26e1
B=https://raw.githubusercontent.com/dss-extensions/electricdss-tst/$C
for f in IEEE123Master.dss IEEE123Regulators.DSS IEEE123Loads.DSS BusCoords.dat; do
  curl -sfSL -o "$f" "$B/Version8/Distrib/IEEETestCases/123Bus/$f"
done
curl -sfSL -o IEEELineCodes.DSS "$B/Version8/Distrib/IEEETestCases/IEEELineCodes.DSS"
curl -sfSL -o LICENSE-OpenDSS.txt "$B/License.txt"
sha256sum -c <<'SUMS'
0b409c0e5d99f365c0140bf38280c1d64932032f1ecb8ef83ae54d4ed0d2716f  BusCoords.dat
fd28f52836205506a1980b52c5a7e12ca585aa626c52a6249502c1c3b2fc2255  IEEE123Loads.DSS
c92a69d9b218b1b2646ec7911783826229309038e72f16b848304c0457c0a54d  IEEE123Master.dss
cd6ad77980dc018631ce39faab528afce40b0b9a50a6bc1c9ac8049ffb8207d7  IEEE123Regulators.DSS
71bc16ef9c6084b9e3bab74fbdf967080b79624683465e1483b787ff160f58b3  IEEELineCodes.DSS
f13543277f32a6437d4f583200858f0988d3acc9598a6a1a292d710009ec21ef  LICENSE-OpenDSS.txt
SUMS
echo "IEEE 123-node feeder files ready in $(pwd)"
