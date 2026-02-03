import torch
import torch.nn as nn
from torchvision.utils import save_image
import os

NZ=100; NGF=64; NC=3; NUM_IMAGES=50; MODELPATH='./models'; OUTPATH='./generated/dogs'
os.makedirs(OUTPATH, exist_ok=True)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(NZ, NGF*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(NGF*8), nn.ReLU(True),
            nn.ConvTranspose2d(NGF*8, NGF*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF*4), nn.ReLU(True),
            nn.ConvTranspose2d(NGF*4, NGF*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF*2), nn.ReLU(True),
            nn.ConvTranspose2d(NGF*2, NGF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF), nn.ReLU(True),
            nn.ConvTranspose2d(NGF, NC, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        return self.net(x.view(-1, NZ, 1, 1))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G = Generator().to(device)

checkpoints = sorted([f for f in os.listdir(MODELPATH) if f.startswith('G_')])
if checkpoints:
    G.load_state_dict(torch.load(f'{MODELPATH}/{checkpoints[-1]}', map_location=device))
    print(f'Loaded {checkpoints[-1]}')

G.eval()
with torch.no_grad():
    z = torch.randn(NUM_IMAGES, NZ).to(device)
    imgs = G(z)
    for i in range(NUM_IMAGES):
        save_image(imgs[i], f'{OUTPATH}/dog_{i:03d}.png', normalize=True)

print(f'Generated {NUM_IMAGES} images -> {OUTPATH}/')
