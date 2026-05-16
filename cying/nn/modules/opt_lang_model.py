import torch
import torch.nn as nn
from .opt_lang_layer import OptLangLayer


class OptLangModel(nn.Module):
    def __init__(
        self,
        voc_size,
        d_model,
        n_head,
        hid_width,
        num_layers
    ):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.hid_width = hid_width
        self.num_layers = num_layers

        self.embedding = nn.Embedding(voc_size,d_model)
        
        self.model = nn.ModuleList([
            OptLangLayer(d_model, n_head, hid_width) for _ in range(num_layers)
        ])

        self.w = nn.Parameter(torch.zeros([num_layers*2,d_model]))
        self.norm = nn.RMSNorm(d_model,elementwise_affine=False)

        self.linear_out = nn.Linear(d_model,voc_size)

    def forward(
        self,
        token_seq,
        padding_mask=None,
        casual_mask=None
    ):
        token_seq = [self.embedding(token_seq)]
        
        for i in range(self.num_layers):
            token_seq.append(self.model[i]._multi_head_attention(self.norm(token_seq[-1]),padding_mask,casual_mask))
            s = torch.stack([self.norm(token_seq[j])@self.w[i*2] for j in range(i*2+2)],dim=-1).softmax(dim=-1)
            token_seq[-1] = sum([token_seq[j]*s[...,j:j+1] for j in range(i*2+2)])

            token_seq.append(self.model[i].nonlinear(self.norm(token_seq[-1])))
            s = torch.stack([self.norm(token_seq[j])@self.w[i*2+1] for j in range(i*2+3)],dim=-1).softmax(dim=-1)
            token_seq[-1] = sum([token_seq[j]*s[...,j:j+1] for j in range(i*2+3)])

        return self.linear_out(self.norm(token_seq[-1]))