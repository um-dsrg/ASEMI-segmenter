import os
import shutil

print('Deleting...')
for dir in os.listdir('output'):
    for fname in os.listdir('output/{}'.format(dir)):
        if fname != '.gitignore':
            path = 'output/{}/{}'.format(dir, fname)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
print('Ready.')