import urllib.request
import re
import json

with urllib.request.urlopen(
    'http://godsnotwheregodsnot.blogspot.com/2013/11/kmeans-color-quantization-seeding.html'
    ) as f:
    html = f.read().decode('utf-8')

colours_section = re.search(r'new String\[\]\{([^}]*)\}', html).group(1)
colour_strings = re.findall('#[0-9A-F]{6}', colours_section)
colour_rgbs = [
    [int(colour_string[2*i+1:2*i+3], 16) for i in range(3)]
    for colour_string in colour_strings
    ]

greys = list()
non_greys = list()
for colour_rgb in colour_rgbs:
    if max(
        abs(colour_rgb[0] - colour_rgb[1]),
        abs(colour_rgb[0] - colour_rgb[2]),
        abs(colour_rgb[1] - colour_rgb[2])
        ) <= 1:
        greys.append(colour_rgb)
    else:
        non_greys.append(colour_rgb)

output = [
    [channel/255 for channel in rgb]
    for rgb in (non_greys + greys)[:253]
    ]
with open('colours.json', 'w', encoding='utf-8') as f:
    json.dump(output, f)