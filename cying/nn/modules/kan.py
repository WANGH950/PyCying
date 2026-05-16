import torch
import torch.nn as nn

class KANEncoder(nn.Module):
    def __init__(self, in_features, k=3, G=20):
        super().__init__()
        self.in_features = in_features
        self.k = k
        self.G = G
        self.h = 1/G
        self.register_buffer('g', torch.linspace(-0.5-self.h*self.k,0.5+self.h*self.k,G+2*k+1))
        self.encoder = nn.ModuleList([nn.Linear(self.G+k,in_features*2+1,bias=False) for _ in range(in_features)])

    def basis(self, x):
        return (torch.relu(x-self.g[:-self.k-1])*torch.relu(self.g[self.k+1:]-x)*4/(self.h*self.k)**2)**self.k

    def forward(self, input):
        return torch.stack([self.encoder[i](self.basis(input[:,i:i+1])) for i in range(self.in_features)]).sum(dim=0)


class KANDecoder(nn.Module):
    def __init__(self, in_features, out_features, hid_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hid_features = hid_features
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1,hid_features),
                nn.GELU(),
                nn.Linear(hid_features,out_features)
            ) for _ in range(in_features*2+1)
        ])

    def forward(self, encoder):
        return sum([self.decoder[i](encoder[:,i:i+1]) for i in range(self.in_features*2+1)])


class KAN(nn.Module):
    def __init__(self, in_features, out_features, hid_features, k=3, G=20):
        super().__init__()
        self.encoder = KANEncoder(
            in_features=in_features,
            k=k,
            G=G
        )
        self.decoder = KANDecoder(
            in_features=in_features,
            out_features=out_features,
            hid_features=hid_features
        )

    def forward(self, input):
        encoder = self.encoder(input)
        decoder = self.decoder(encoder)
        return decoder