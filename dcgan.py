import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os, argparse

BATCH=64; NZ=100; NGF=64; NDF=64; NC=3; EPOCHS=25; LR=2e-4; BETA=0.5
IMGSIZE=64; DATAPATH='./data/dogs'; OUTPATH='./generated'; MODELPATH='./models'

os.makedirs(OUTPATH, exist_ok=True)
os.makedirs(MODELPATH, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((IMGSIZE, IMGSIZE)),
    transforms.CenterCrop(IMGSIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = datasets.ImageFolder(DATAPATH, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=2, drop_last=True)

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)

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

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(NC, NDF, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(NDF, NDF*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF*2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(NDF*2, NDF*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF*4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(NDF*4, NDF*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF*8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(NDF*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).view(-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G = Generator().to(device).apply(weights_init)
D = Discriminator().to(device).apply(weights_init)
criterion = nn.BCELoss()
optG = optim.Adam(G.parameters(), lr=LR, betas=(BETA, 0.999))
optD = optim.Adam(D.parameters(), lr=LR, betas=(BETA, 0.999))

fixed_noise = torch.randn(64, NZ).to(device)

for epoch in range(EPOCHS):
    for i, (imgs, _) in enumerate(loader):
        imgs = imgs.to(device)
        real_label = torch.ones(imgs.size(0)).to(device)
        fake_label = torch.zeros(imgs.size(0)).to(device)

        D.zero_grad()
        out_real = D(imgs)
        loss_real = criterion(out_real, real_label)
        z = torch.randn(imgs.size(0), NZ).to(device)
        fake = G(z)
        out_fake = D(fake.detach())
        loss_fake = criterion(out_fake, fake_label)
        lossD = loss_real + loss_fake
        lossD.backward()
        optD.step()

        G.zero_grad()
        out = D(fake)
        lossG = criterion(out, real_label)
        lossG.backward()
        optG.step()

        if i % 50 == 0:
            print(f'[{epoch}/{EPOCHS}][{i}/{len(loader)}] lossD={lossD.item():.4f} lossG={lossG.item():.4f}')

    with torch.no_grad():
        samples = G(fixed_noise)
        save_image(samples, f'{OUTPATH}/epoch_{epoch:03d}.png', nrow=8, normalize=True)

    torch.save(G.state_dict(), f'{MODELPATH}/G_epoch_{epoch:03d}.pth')
    torch.save(D.state_dict(), f'{MODELPATH}/D_epoch_{epoch:03d}.pth')

print('Training done.')
