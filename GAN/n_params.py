import torch
from models.cProGAN import Generator

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    D_layers=[8, 16, 32, 64, 128, 256]
    G_layers=[256, 128, 64, 32, 16, 8]
    G = Generator(17, 17, 17, G_layers).to(device)

    for i, layer in enumerate(G.layers):
        n = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        print(f"{i}: {n}")

if __name__ == "__main__":
    main()