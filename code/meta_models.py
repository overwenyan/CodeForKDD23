import torch.nn.functional as F
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, embedding_dim, reg):
        super(MLP, self).__init__()
        self.embedding1 = nn.Embedding(4, embedding_dim)
        self.embedding2 = nn.Embedding(2, embedding_dim)
        self.embedding3 = nn.Embedding(2, embedding_dim)
        self.embedding4 = nn.Embedding(5, embedding_dim)
        self.embedding5 = nn.Embedding(3, embedding_dim)
        self.embedding6 = nn.Embedding(4, embedding_dim)
        self.embedding7 = nn.Embedding(4, embedding_dim)
        self.W = nn.Linear(embedding_dim * 7, 1, bias=False)
        self.reg = reg # what is reg?

    def forward(self, input):
        [e1, e2, e3, e4, e5, e6, e7] = input
        emb1 = self.embedding1(e1)
        emb2 = self.embedding2(e2)
        emb3 = self.embedding3(e3)
        emb4 = self.embedding4(e4)
        emb5 = self.embedding5(e5)
        emb6 = self.embedding6(e6)
        emb7 = self.embedding7(e7)

        mlp_out = torch.cat([emb1, emb2, emb3, emb4, emb5, emb6, emb7], dim=-1)
        inference = self.W(torch.tanh(mlp_out)).sum(dim=1, keepdim=True)
        regs = self.reg * (torch.norm(emb1) +
                           torch.norm(emb2) +
                           torch.norm(emb3) +
                           torch.norm(emb4) +
                           torch.norm(emb5) +
                           torch.norm(emb6) +
                           torch.norm(emb7))
        return inference  # , regs

    def forward_pair(self, input_pos, input_neg):
        inference_pos = self.forward(input_pos)
        inference_neg = self.forward(input_neg)
        inference = inference_pos - inference_neg
        return inference
    
class FM(nn.Module):
    def __init__(self, embedding_dim, reg):
        super(FM, self).__init__()
        self.embedding1 = nn.Embedding(4, embedding_dim)
        self.embedding2 = nn.Embedding(2, embedding_dim)
        self.embedding3 = nn.Embedding(2, embedding_dim)
        self.embedding4 = nn.Embedding(5, embedding_dim)
        self.embedding5 = nn.Embedding(3, embedding_dim)
        self.embedding6 = nn.Embedding(4, embedding_dim)
        self.embedding7 = nn.Embedding(4, embedding_dim)
        self.reg = reg

    def forward_old(self, e1, e2, e3, e4, e5, e6, e7):

        emb1 = self.embedding1(e1)
        emb2 = self.embedding2(e2)
        emb3 = self.embedding3(e3)
        emb4 = self.embedding4(e4)
        emb5 = self.embedding5(e5)
        emb6 = self.embedding6(e6)
        emb7 = self.embedding7(e7)

        fm_out = torch.pow(emb1+emb2+emb3+emb4+emb5+emb6+emb7, 2) - torch.pow(emb1,2) - torch.pow(emb2,2) - torch.pow(emb3,2) - torch.pow(emb4,2) - torch.pow(emb5,2) - torch.pow(emb6,2)- torch.pow(emb7,2)
        inference = fm_out.sum(dim=1, keepdim=True)
        regs = self.reg * (torch.norm(emb1) +
                           torch.norm(emb2) +
                           torch.norm(emb3) +
                           torch.norm(emb4) +
                           torch.norm(emb5) +
                           torch.norm(emb6) +
                           torch.norm(emb7) )
        return inference #, regs

    def forward(self, input):
        [e1, e2, e3, e4, e5, e6, e7] = input
        emb1 = self.embedding1(e1)
        emb2 = self.embedding2(e2)
        emb3 = self.embedding3(e3)
        emb4 = self.embedding4(e4)
        emb5 = self.embedding5(e5)
        emb6 = self.embedding6(e6)
        emb7 = self.embedding7(e7)

        fm_out = torch.pow(emb1+emb2+emb3+emb4+emb5+emb6+emb7, 2) - torch.pow(emb1,2) - torch.pow(emb2,2) - torch.pow(emb3,2) - torch.pow(emb4,2) - torch.pow(emb5,2) - torch.pow(emb6,2)- torch.pow(emb7,2)
        inference = fm_out.sum(dim=1, keepdim=True)
        regs = self.reg * (torch.norm(emb1) +
                           torch.norm(emb2) +
                           torch.norm(emb3) +
                           torch.norm(emb4) +
                           torch.norm(emb5) +
                           torch.norm(emb6) +
                           torch.norm(emb7) )
        return inference  # ,regs