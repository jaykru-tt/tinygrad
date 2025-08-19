#!/usr/bin/env bash
export TT_METAL_SIMULATOR=/home/j/host/ttsim
export TT_METAL_REPO=/home/j/host/tt-metal
export TT_METAL_SLOW_DISPATCH_MODE=1

# # For WH:
cp $TT_METAL_SIMULATOR/src/_out/release_wh/ttsim "$TT_METAL_SIMULATOR"/run.sh
cp "$TT_METAL_REPO"/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml "$TT_METAL_SIMULATOR"/soc_descriptor.yaml

source .venv/bin/activate

uv run python test_ttnn_basic.py

