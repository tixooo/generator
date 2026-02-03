import os, tarfile, shutil
from PIL import Image

TAR_PATH = './images.tar'
EXTRACT = './data/extracted'
OUT = './data/dogs/dogs'
os.makedirs(OUT, exist_ok=True)

if os.path.exists(TAR_PATH):
    with tarfile.open(TAR_PATH) as t:
        t.extractall(EXTRACT)
    print('Extracted.')

count = 0
for root, dirs, files in os.walk(EXTRACT):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            src = os.path.join(root, f)
            try:
                img = Image.open(src).convert('RGB')
                img.save(os.path.join(OUT, f'dog_{count:06d}.jpg'))
                count += 1
            except:
                pass

print(f'Prepared {count} images -> {OUT}')
