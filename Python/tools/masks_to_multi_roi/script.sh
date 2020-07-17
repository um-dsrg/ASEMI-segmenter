#!/bin/bash

python masks_to_multi_roi.py \
    --input \
        "../../example_volume/output/segment/air" \
        "../../example_volume/output/segment/bones" \
        "../../example_volume/output/segment/tissues" \
    --output "output" \
    --soft \
    --processes 1
