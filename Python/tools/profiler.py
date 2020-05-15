#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2020 Marc Tanti

import cProfile
import pstats
import os
import io
import textwrap
import tqdm
from asemi_segmenter import listener
from asemi_segmenter import segment

TEXT_WIDTH = 100

#########################################
class ProgressListener(listener.ProgressListener):

    #########################################
    def __init__(self):
        self.prog = None
        self.prog_prev_value = 0

    #########################################
    def log_output(self, text):
        if text == '':
            print()
        else:
            for (i, line) in enumerate(textwrap.wrap(text, TEXT_WIDTH)):
                if i == 0:
                    print(line)
                else:
                    print('   '+line)

    #########################################
    def current_progress_start(self, start, total):
        self.prog = tqdm.tqdm(initial=start, total=total)
        self.prog_prev_value = start

    #########################################
    def current_progress_update(self, curr):
        self.prog.update(curr - self.prog_prev_value)
        self.prog_prev_value = curr

    #########################################
    def current_progress_end(self):
        self.prog.close()


config = {
    "soft_segmentation": False,
    "as_masks": True,
    "image_extension": "tiff",
    "bits": 8
}

print('Starting profiler...')
print()
pr = cProfile.Profile()
pr.enable()

segment.main(
    segmenter=os.path.join('..', 'example_volume', 'output', 'train', 'segmenter.pkl'),
    preproc_volume_fullfname=os.path.join('..', 'example_volume', 'output', 'preprocess', 'volume.hdf'),
    config=config,
    results_dir=os.path.join('..', 'example_volume', 'output', 'segment'),
    checkpoint_fullfname=None,
    checkpoint_namespace='segment',
    reset_checkpoint=False,
    checkpoint_init=dict(),
    max_processes=1,
    max_batch_memory=1.0,
    listener=ProgressListener(),
    debug_mode=True
    )

pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats()
with open('profiler.txt', 'w', encoding='utf-8') as f:
    lines = s.getvalue().strip().split('\n')
    lines[2] = lines[2].strip()
    for i in range(4, len(lines)):
        lines[i] = '\t'.join(lines[i].strip().split(None, 5))
    print('\n'.join(lines), file=f)
print()
print('Profiling ended.')
