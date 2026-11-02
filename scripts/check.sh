#!/usr/bin/env bash
set -euo pipefail
python -m pytest -q
python -m onegpu_posttrain.cli validate-example-data data/examples
