#!/bin/bash

python multi_roi_to_masks.py \
    --input "../masks_to_multi_roi/output" \
    --output "output" \
    --labels_fullfname "../masks_to_multi_roi/output/labels.txt" \
    --bits 8 \
    --image_extension "tiff"
