import sys
import os
import shutil

if len(sys.args) == 1 or sys.args[1] != 'keep_checkpoint':
    keep_checkpoint = False
else:
    keep_checkpoint = True

print('Deleting...')
if not keep_checkpoint:
    os.remove('output/checkpoint.json')
os.remove('output/log.txt')
for dir in os.listdir('output'):
    for fname in os.listdir('output/{}'.format(dir)):
        if fname != '.gitignore':
            path = 'output/{}/{}'.format(dir, fname)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
print('Ready.')