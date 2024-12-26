import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, 0.6, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GATLayer(nfeat, nhid, alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GATLayer(nhid * nheads, nclass, alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x


class KAG(nn.Module):
    def __init__(self, d_k, d_r, d_h):
        super(KAG, self).__init__()
        self.d_k = d_k
        self.d_r = d_r
        self.d_h = d_h

        self.W_q = nn.Linear(d_k + d_r + d_h, d_k)
        self.W_h = nn.Linear(d_k + d_r + d_h, d_h)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, h_t, x_t, r_t, prev_h):
        combined = torch.cat((x_t, r_t, prev_h), dim=1)
        g_t = self.sigmoid(self.W_q(combined))
        candidate_h = self.tanh(self.W_h(combined))
        h_t = (1 - g_t) * prev_h + g_t * candidate_h
        return h_t


class OutputLayer(nn.Module):
    def __init__(self, d_k, d_e, d_h):
        super(OutputLayer, self).__init__()
        self.W_p = nn.Linear(d_k + d_e + d_h, d_h)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h_t, e_t, kc_t):
        combined = torch.cat((h_t, e_t, kc_t), dim=1)
        y_t = self.sigmoid(self.W_p(combined))
        return y_t


class CEKT(nn.Module):
    def __init__(self, d_k, d_e, d_r, d_h, alpha, nheads):
        super(CEKT, self).__init__()
        self.gat = GAT(d_k, d_h, d_h, 0.6, alpha, nheads)
        self.kag = KAG(d_k, d_r, d_h)
        self.output_layer = OutputLayer(d_k, d_e, d_h)

    def forward(self, x, adj, e_t, r_t, prev_h):
        h_t = self.gat(x, adj)
        h_t = self.kag(h_t, e_t, r_t, prev_h)
        y_t = self.output_layer(h_t, e_t, x)
        return y_t


# 超参数
alpha = 0.2
nheads = 3
learning_rate = 0.001
num_epochs = 10

# 初始化模型
model = CEKT(d_k, d_e, d_r, d_h, alpha, nheads)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()

# 假设我们有邻接矩阵 adj
adj = torch.eye(d_k)

# 训练循环
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # 模拟输入数据
    x = torch.randn((d_k, d_k))
    e_t = torch.randn((1, d_e))
    r_t = torch.randn((1, d_r))
    prev_h = torch.zeros((1, d_h))

    # 前向传播
    y_t = model(x, adj, e_t, r_t, prev_h)

    # 计算损失
    target = torch.tensor([[1.0]])  # 假设目标输出
    loss = criterion(y_t, target)
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

