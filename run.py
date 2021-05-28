import numpy as np
import argparse
import random
from tqdm import tqdm
import torch

from model import *
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="Run baselines.")
    
    parser.add_argument('--domain', choices=['cs', 'phy', 'math'],
                        help='The target domain.')
    parser.add_argument('--narrow', action='store_true',
                        help='Training and evaluating on the corresponding subdomains.')
    parser.add_argument('--method', choices=['lr', 'mlp', 'mc', 'cfl', 'hicfl'],
                        help='The learning method.')
    parser.add_argument('--pu', action='store_true',
                        help='PU setting.')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Hyperparameter to balance the global and local information (HiCFL).')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--device', type=int, default=0)
    return parser.parse_args()
    

def main(args):
    print("Domain:", args.domain)
    print("Method:", args.method)
    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    domain = args.domain

    # load seed terms
    phrase_id, phrases = get_phrases(f"term-candidates/arxiv-phrases-filtering-{domain}.txt")
    
    # load label information (automatic annotation)
    cp=f"wikipedia/core-categories/phrase-core-categories-{domain}.txt"
    cm=f"wikipedia/core-categories/phrase-core-categories-media-{domain}.txt"
    path_gold_subcategories = "wikipedia/gold-subcategories"
    if domain == "cs":
        seed_labels_1 = get_core_phrase_label("computer science", f"{path_gold_subcategories}/wikipedia-category-Subfields_of_computer_science-3.txt", phrase_id, cp, cm)
        seed_labels_2 = get_core_phrase_label("artificial intelligence", f"{path_gold_subcategories}/wikipedia-category-Artificial_intelligence-2.txt", phrase_id, cp, cm)
        seed_labels_3 = get_core_phrase_label("machine learning", f"{path_gold_subcategories}/wikipedia-category-Machine_learning-2.txt", phrase_id, cp, cm)
    elif domain == "phy":
        seed_labels_1 = get_core_phrase_label("physics", f"{path_gold_subcategories}/wikipedia-category-Subfields_of_physics-3.txt", phrase_id, cp, cm)
        seed_labels_2 = get_core_phrase_label("mechanics", f"{path_gold_subcategories}/wikipedia-category-Mechanics-2.txt", phrase_id, cp, cm)
        seed_labels_3 = get_core_phrase_label("quantum mechanics", f"{path_gold_subcategories}/wikipedia-category-Quantum_mechanics-2.txt", phrase_id, cp, cm)
    elif domain == "math":
        seed_labels_1 = get_core_phrase_label("mathematics", f"{path_gold_subcategories}/wikipedia-category-Fields_of_mathematics-3.txt", phrase_id, cp, cm)
        seed_labels_2 = get_core_phrase_label("algebra", f"{path_gold_subcategories}/wikipedia-category-Algebra-2.txt", phrase_id, cp, cm)
        seed_labels_3 = get_core_phrase_label("abstract algebra", f"{path_gold_subcategories}/wikipedia-category-Abstract_algebra-2.txt", phrase_id, cp, cm)
        
    if args.narrow: # narrow domains: ml/qm/aa
        list_seed_labels = [seed_labels_1, seed_labels_2, seed_labels_3]
        seed_labels = list_seed_labels[-1]
    else:  # broad domains: cs/phy/math
        list_seed_labels = [seed_labels_1]
        seed_labels = list_seed_labels[0]
    
    # load train/valid/test split
    split_idx, split_y = load_train_valid_test_split(seed_labels, domain)
    
    if args.pu:  # PU setting
        assert len(list_seed_labels) >= 2
        pu_positives = []
        with open(f"train-valid-test/{domain}/pu_positives.txt") as f:
            for line in f:
                pu_positives.append(int(line))
        pu_idx, pu_y = train_test_split_for_pu(split_idx["train"], split_y["train"], list_seed_labels[-2], pu_positives)
        split_idx_pu, split_y_pu = split_idx.copy(), split_y.copy()
        split_idx_pu["train"], split_y_pu["train"] = pu_idx, pu_y
    
    for key,value in split_y.items():
        split_y[key] = torch.LongTensor(value).to(device)
    if args.pu:
        for key,value in split_y_pu.items():
            split_y_pu[key] = torch.LongTensor(value).to(device)
    
    # process train/valid/test split for HiCFL
    if args.method=="hicfl":
        list_split_y = []
        num_hierarchy = len(list_seed_labels)
        
        for d in list_seed_labels:
            idx_, y_ = load_train_valid_test_split(d, domain)
            for key,value in y_.items():
                y_[key] = torch.LongTensor(value).to(device)
            list_split_y.append(y_)
    
        if args.pu:
            list_split_y[-1] = split_y_pu
    
    if args.method == "mc":
        X = load_embeddings(f'features/{domain}.wordvectors', phrases)
    else:
        X = load_embeddings('features/general.wordvectors', phrases)  # G
        # X = load_embeddings(f'features/{domain}.wordvectors', phrases)  # S
        # X = torch.cat([load_embeddings(f'features/{domain}.wordvectors', phrases),\
                       # load_embeddings('features/general.wordvectors', phrases)],dim=-1)  # SG

    X = X.to(device)
    num_features = X.size()[1]
    
    if args.method=="mc":
        X_general = load_embeddings('features/general.wordvectors', phrases)
        X_general = X_general.to(device)
    
    # build core-anchored semantic graph for CFL/HiCFL
    if args.method in ["cfl","hicfl"]:
        A = get_term_graph(split_idx["train"], phrase_id, domain)
        A = A.to(device)

        
    test_aucs = []
    test_aps = []
    best_auc_epochs = []
    best_ap_epochs = []
    
    for run in range(1,args.runs+1):
        print("Run:", run)
        
        best_valid_auc = 0
        best_valid_ap = 0
        best_test_auc = 0
        best_test_ap = 0
        
        best_auc_epoch = 0
        best_ap_epoch = 0

        if args.method=="lr":
            if args.pu:
                rets = train_test_lr(X, split_idx_pu, split_y_pu)
            else:
                rets = train_test_lr(X, split_idx, split_y)
            best_test_auc, best_test_ap = rets
            
        elif args.method=="mlp":
            model = MLP(num_features, args.hidden_channels, args.num_classes, args.num_layers, args.dropout).to(device)
            model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            for epoch in tqdm(range(1, 1 + args.epochs)):
                if args.pu:
                    loss = train_mlp(model, X, split_idx_pu, split_y_pu, optimizer)
                else:
                    loss = train_mlp(model, X, split_idx, split_y, optimizer)
                aucs,aps = test_mlp(model, X, split_idx, split_y)
                train_auc, valid_auc, test_auc = aucs
                train_ap, valid_ap, test_ap = aps
                
                if valid_auc > best_valid_auc:
                    best_valid_auc, best_test_auc = valid_auc, test_auc
                    best_auc_epoch = epoch
                if valid_ap > best_valid_ap:
                    best_valid_ap, best_test_ap = valid_ap, test_ap
                    best_ap_epoch = epoch
                    
            y_scores = predict_mlp(model, X, split_idx, split_y)
            y_scores = y_scores.cpu().numpy()                    

        elif args.method=="mc":
            model = MC(num_features, args.hidden_channels, args.num_classes, args.num_layers, args.dropout).to(device)
            model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            for epoch in tqdm(range(1, 1 + args.epochs)):
                if args.pu:
                    loss = train_mc(model, X, X_general, split_idx_pu, split_y_pu, optimizer)
                else:
                    loss = train_mc(model, X, X_general, split_idx, split_y, optimizer)
                aucs,aps = test_mc(model, X, X_general, split_idx, split_y)
                train_auc, valid_auc, test_auc = aucs
                train_ap, valid_ap, test_ap = aps
                
                if valid_auc > best_valid_auc:
                    best_valid_auc, best_test_auc = valid_auc, test_auc
                    best_auc_epoch = epoch
                if valid_ap > best_valid_ap:
                    best_valid_ap, best_test_ap = valid_ap, test_ap
                    best_ap_epoch = epoch
                                    
        elif args.method=="cfl":
            model = CFL(num_features, args.hidden_channels, args.num_classes, args.num_layers, args.dropout).to(device)
            model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            for epoch in tqdm(range(1, 1 + args.epochs)):
                if args.pu:
                    loss = train_cfl(model, X, A, split_idx_pu, split_y_pu, optimizer)
                else:
                    loss = train_cfl(model, X, A, split_idx, split_y, optimizer)
                aucs,aps = test_cfl(model, X, A, split_idx, split_y)
                train_auc, valid_auc, test_auc = aucs
                train_ap, valid_ap, test_ap = aps
                
                if valid_auc > best_valid_auc:
                    best_valid_auc, best_test_auc = valid_auc, test_auc
                    best_auc_epoch = epoch
                if valid_ap > best_valid_ap:
                    best_valid_ap, best_test_ap = valid_ap, test_ap
                    best_ap_epoch = epoch
                    
            y_scores = predict_cfl(model, X, A, split_idx, split_y)
            y_scores = y_scores.cpu().numpy()
        
        elif args.method=="hicfl":
            model = HiCFL(num_features, args.hidden_channels, args.num_classes, args.num_layers, num_hierarchy, args.dropout).to(device)
            model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            for epoch in tqdm(range(1, 1 + args.epochs)):
                if args.pu:
                    loss = train_hicfl_pu(model, X, A, split_idx, split_idx_pu, list_split_y, optimizer)
                else:
                    loss = train_hicfl(model, X, A, split_idx, list_split_y, optimizer)        
                aucs,aps = test_hicfl(model, X, A, split_idx, split_y, args.alpha)
                train_auc, valid_auc, test_auc = aucs
                train_ap, valid_ap, test_ap = aps
                
                if valid_auc > best_valid_auc:
                    best_valid_auc, best_test_auc = valid_auc, test_auc
                    best_auc_epoch = epoch
                if valid_ap > best_valid_ap:
                    best_valid_ap, best_test_ap = valid_ap, test_ap
                    best_ap_epoch = epoch
                    
            y_scores = predict_hicfl(model, X, A, split_idx, split_y, args.alpha)
            y_scores = y_scores.cpu().numpy()
            
        print("ROC-AUC:", "%.3f"%np.mean(best_test_auc), end="; ")
        print("PR-AUC:", "%.3f"%np.mean(best_test_ap))
        print("Epoch:", best_auc_epoch, best_ap_epoch)
        test_aucs.append(best_test_auc)
        test_aps.append(best_test_ap)        
        
    print("ROC-AUC:", "%.3f (%.3f)"%(np.mean(test_aucs),np.std(test_aucs)))
    print("PR-AUC: ", "%.3f (%.3f)"%(np.mean(test_aps),np.std(test_aps)))
    

if __name__ == "__main__":
    args = parse_args()
    main(args)


