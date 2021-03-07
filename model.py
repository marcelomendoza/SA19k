import numpy as np
import torch
from torch.nn.init import xavier_normal_


class TuckER(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(TuckER, self).__init__()

        self.E = torch.nn.Embedding(len(d.entities), d1)
        self.R = torch.nn.Embedding(len(d.relations), d2)
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)), 
                                    dtype=torch.float, device="cuda", requires_grad=True))

        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)
        

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, e1_idx, r_idx):
        e1 = self.E(e1_idx)  # bs x de
        x = self.bn0(e1)  # bs x de
        x = self.input_dropout(x) # bs x de
        x = x.view(-1, 1, e1.size(1)) # bs x 1 x de

        r = self.R(r_idx) # bs x dr
        W_mat = torch.mm(r, self.W.view(r.size(1), -1)) # (bs x dr) x (dr x (de x de)) -> bs x (de x de)
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1)) # bs x de x de
        W_mat = self.hidden_dropout1(W_mat) # bs x de x de

        x = torch.bmm(x, W_mat) # bs x 1 x de
        x = x.view(-1, e1.size(1)) # bs x de      
        x = self.bn1(x) # bs x de
        x = self.hidden_dropout2(x) # bs x de
        x = torch.mm(x, self.E.weight.transpose(1,0)) # (bs x de) x (de x ne) -> bs x ne
        pred = torch.sigmoid(x) # bs x ne
        return pred

