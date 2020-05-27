import sys
import os
import shutil
from asemi_segmenter.lib import files

print('Deleting...')
print(' output/checkpoint.json')
files.delete('output/checkpoint.json')
print(' output/log.txt')
files.delete('output/log.txt')
for dir in os.listdir('output'):
    dir = 'output/{}'.format(dir)
    if os.path.isdir(dir):
        print('', dir)
        for fname in os.listdir(dir):
            if fname != '.gitignore':
                files.delete('{}/{}'.format(dir, fname))
print('Ready.')