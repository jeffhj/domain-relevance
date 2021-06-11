import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression


class CFL(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(CFL, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)
    
def train_cfl(model, X, A, split_idx, split_y, optimizer):
    model.train()
    optimizer.zero_grad()
    
    out = model(X, A)[split_idx["train"]]
    loss = F.nll_loss(out, split_y["train"])
    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def test_cfl(model, X, A, split_idx, split_y):
    model.eval()

    out = model(X, A)
    y_scores = F.softmax(out,dim=1)[:,1]
    
    aucs = []
    for key in split_idx:
        y_true_ = split_y[key].cpu()    
        y_scores_ = y_scores[split_idx[key]].cpu()
        auc = roc_auc_score(y_true_, y_scores_)
        aucs.append(auc)
        
    aps = []
    for key in split_idx:
        y_true_ = split_y[key].cpu()    
        y_scores_ = y_scores[split_idx[key]].cpu()
        ap = average_precision_score(y_true_, y_scores_)
        aps.append(ap)
    
    return aucs, aps

@torch.no_grad()
def predict_cfl(model, X, A):
    model.eval()

    out = model(X, A)
    y_scores = F.softmax(out,dim=1)[:,1]
    
    return y_scores


class HiCFL(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_hierarchy, dropout):
        super(HiCFL, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        self.fcs_g = torch.nn.ModuleList()
        self.bns_g = torch.nn.ModuleList()
        self.fcs_g.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.bns_g.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_hierarchy-1):
            self.fcs_g.append(torch.nn.Linear(hidden_channels*2, hidden_channels))
            self.bns_g.append(torch.nn.BatchNorm1d(hidden_channels))
            
        self.fcs_l = torch.nn.ModuleList()
        self.bns_l = torch.nn.ModuleList()
        for _ in range(num_hierarchy):
            self.fcs_l.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns_l.append(torch.nn.BatchNorm1d(hidden_channels))
            
        self.out_g = torch.nn.Linear(hidden_channels, out_channels)
        
        self.out_l = torch.nn.ModuleList()
        for _ in range(num_hierarchy):
            self.out_l.append(torch.nn.Linear(hidden_channels, out_channels))
            
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs_l:
            fc.reset_parameters()
        for fc in self.fcs_g:
            fc.reset_parameters()
        self.out_g.reset_parameters()
        for fc in self.out_l:
            fc.reset_parameters()
        for bn in self.bns_g:
            bn.reset_parameters()
        for bn in self.bns_l:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        hg = []
        xg = self.fcs_g[0](x)
        xg = self.bns_g[0](xg)
        xg = F.relu(xg)
        xg = F.dropout(xg, p=self.dropout, training=self.training)
        hg.append(xg)
        for i, fc in enumerate(self.fcs_g[1:]):
            xg = torch.cat([hg[i],x],-1)
            xg = fc(xg)
            xg = self.bns_g[i+1](xg)
            xg = F.relu(xg)
            xg = F.dropout(xg, p=self.dropout, training=self.training)
            hg.append(xg)
            
        og = self.out_g(hg[-1])
        
        ol = []
        for i in range(len(self.fcs_l)):
            hl = self.fcs_l[i](hg[i])
            hl = self.bns_l[i](hl)
            hl = F.relu(hl)
            hl = F.dropout(hl, p=self.dropout, training=self.training)
            ol.append(self.out_l[i](hl))
            
        return og.log_softmax(dim=-1), [o.log_softmax(dim=-1) for o in ol]
    
def train_hicfl(model, X, A, split_idx, list_split_y, optimizer):
    model.train()
    optimizer.zero_grad()
    
    og, ol = model(X, A)

    loss = F.nll_loss(og[split_idx["train"]], list_split_y[-1]["train"])
    for i,o in enumerate(ol):
        loss += F.nll_loss(o[split_idx["train"]], list_split_y[i]["train"])
    
    loss.backward()
    optimizer.step()

    return loss.item()

def train_hicfl_pu(model, X, A, split_idx, split_idx_pu, list_split_y, optimizer):
    model.train()
    optimizer.zero_grad()
    
    og, ol = model(X, A)

    loss = F.nll_loss(og[split_idx_pu["train"]], list_split_y[-1]["train"])
    for i,o in enumerate(ol[:-1]):
        loss += F.nll_loss(o[split_idx["train"]], list_split_y[i]["train"])

    loss += F.nll_loss(ol[-1][split_idx_pu["train"]], list_split_y[-1]["train"])
    
    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def test_hicfl(model, X, A, split_idx, split_y, alpha=0.5):
    model.eval()

    og, ol = model(X, A)
    
    y_ols = [F.softmax(o,dim=1)[:,1] for o in ol]
    y_ol = y_ols[0]
    for y_ in y_ols[1:]:
        y_ol *= y_
    
    y_og = F.softmax(og,dim=1)[:,1]
    
    y_scores = alpha*y_og + (1-alpha)*y_ol
    
    aucs = []
    for key in split_idx:
        y_true_ = split_y[key].cpu()    
        y_scores_ = y_scores[split_idx[key]].cpu()
        auc = roc_auc_score(y_true_, y_scores_)
        aucs.append(auc)
        
    aps = []
    for key in split_idx:
        y_true_ = split_y[key].cpu()    
        y_scores_ = y_scores[split_idx[key]].cpu()
        ap = average_precision_score(y_true_, y_scores_)
        aps.append(ap)
    
    return aucs, aps

@torch.no_grad()
def predict_hicfl(model, X, A, alpha=0.5):
    model.eval()

    og, ol = model(X, A)
    
    y_ols = [F.softmax(o,dim=1)[:,1] for o in ol]
    y_ol = y_ols[0]
    for y_ in y_ols[1:]:
        y_ol *= y_
    
    y_og = F.softmax(og,dim=1)[:,1]
    y_scores = alpha*y_og + (1-alpha)*y_ol
    
    return y_scores


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()

        self.fcs = torch.nn.ModuleList()
        self.fcs.append(torch.nn.Linear(in_channels, hidden_channels))
        
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        for _ in range(num_layers - 2):
            self.fcs.append(
                torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.fcs.append(torch.nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for fc in self.fcs:
            fc.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()            

    def forward(self, x):
        for i, fc in enumerate(self.fcs[:-1]):
            x = fc(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fcs[-1](x)
        return x.log_softmax(dim=-1)
    
def train_mlp(model, X, split_idx, split_y, optimizer):
    model.train()
    optimizer.zero_grad()
    
    out = model(X)[split_idx["train"]]
    loss = F.nll_loss(out, split_y["train"])
    loss.backward()
    optimizer.step()

    return loss.item()

def train_mlp_minibatch(model, train_loader, optimizer):
    model.train()
    
    total_loss = 0
    for x, y in train_loader:
        # x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.nll_loss(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test_mlp(model, X, split_idx, split_y):
    model.eval()

    out = model(X)
    y_scores = F.softmax(out,dim=1)[:,1]
    
    aucs = []
    for key in split_idx:
        y_true_ = split_y[key].cpu()    
        y_scores_ = y_scores[split_idx[key]].cpu()
        auc = roc_auc_score(y_true_, y_scores_)
        aucs.append(auc)
        
    aps = []
    for key in split_idx:
        y_true_ = split_y[key].cpu()    
        y_scores_ = y_scores[split_idx[key]].cpu()
        ap = average_precision_score(y_true_, y_scores_)
        aps.append(ap)
    
    return aucs, aps

@torch.no_grad()
def predict_mlp(model, X):
    model.eval()

    out = model(X)
    y_scores = F.softmax(out,dim=1)[:,1]
    
    return y_scores


class MC(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(MC, self).__init__()

        self.fc1 = torch.nn.Linear(in_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(in_channels, hidden_channels)
        self.fc3 = torch.nn.Linear(in_channels, hidden_channels)
        self.fc_o = torch.nn.Linear(hidden_channels*3, out_channels)
        
        self.bns1 = torch.nn.BatchNorm1d(hidden_channels)
        self.bns2 = torch.nn.BatchNorm1d(hidden_channels)
        self.bns3 = torch.nn.BatchNorm1d(hidden_channels)
        self.bns4 = torch.nn.BatchNorm1d(hidden_channels)
        
        self.dropout = dropout

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc_o.reset_parameters()
        
        self.bns1.reset_parameters()
        self.bns2.reset_parameters()
        self.bns3.reset_parameters()
        self.bns4.reset_parameters()
        
    def forward(self, x, x_general):
        h1 = self.fc1(x)
        h1 = self.bns1(h1)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        
        h2 = self.fc2(x_general)
        h2 = self.bns2(h2)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        
        h3a = self.fc3(x)
        h3a = self.bns3(h3a)
        h3a = F.relu(h3a)
        h3b = self.fc3(x_general)
        # h3b = self.bns3(h3b)
        h3b = self.bns4(h3b)
        h3b = F.relu(h3b)
        d = torch.abs(h3a-h3b)
        d = F.dropout(d, p=self.dropout, training=self.training)

        h_c = torch.cat([h1,h2,d],-1)
        h_o = self.fc_o(h_c)
        
        return h_o.log_softmax(dim=-1)

def train_mc(model, X, X_general, split_idx, split_y, optimizer):
    model.train()
    optimizer.zero_grad()
    
    out = model(X, X_general)[split_idx["train"]]
    loss = F.nll_loss(out, split_y["train"])
    loss.backward()
    optimizer.step()

    return loss.item()

def train_mc_minibatch(model, train_loader, optimizer):
    model.train()
    
    total_loss = 0
    for xl, xg, y in train_loader:
        # x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(xl, xg)
        loss = F.nll_loss(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xl.size(0)
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test_mc(model, X, X_general, split_idx, split_y):
    model.eval()

    out = model(X, X_general)
    y_scores = F.softmax(out,dim=1)[:,1]
    
    aucs = []
    for key in split_idx:
        y_true_ = split_y[key].cpu()    
        y_scores_ = y_scores[split_idx[key]].cpu()
        auc = roc_auc_score(y_true_, y_scores_)
        aucs.append(auc)
        
    aps = []
    for key in split_idx:
        y_true_ = split_y[key].cpu()    
        y_scores_ = y_scores[split_idx[key]].cpu()
        ap = average_precision_score(y_true_, y_scores_)
        aps.append(ap)
    
    return aucs, aps


def train_test_lr(X, split_idx, split_y):
    clf = LogisticRegression(solver='lbfgs').fit(X[split_idx['train']].cpu(), split_y['train'].cpu())
    y_proba = clf.predict_proba(X.cpu())[:,1]
    
    aucs = []
    for key in split_idx:
        y_true_ = split_y[key].cpu()    
        y_proba_ = y_proba[split_idx[key]]
        auc = roc_auc_score(y_true_, y_proba_)
        aucs.append(auc)
        
    aps = []
    for key in split_idx:
        y_true_ = split_y[key].cpu()    
        y_proba_ = y_proba[split_idx[key]]
        ap = average_precision_score(y_true_, y_proba_)
        aps.append(ap)

    return aucs[-1], aps[-1]