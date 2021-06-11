import numpy as np
import argparse
import random
from tqdm import tqdm
import torch
from lemmatizer import NLTKLemmatizer
import wikipedia
import os

from model import *
from utils import *


def get_wiki_search_result(term, mode=0):
    if mode==0:
        return wikipedia.search(f"\"{term}\"")
    else:
        return wikipedia.search(term)

def get_wiki_search_result_batch(terms, mode=0):
    dirs = "tmp"
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    with open(f"{dirs}/phrase-wiki-search-results-{mode}-query.txt", 'w') as fw:    
        for w in terms:
            rets = []
            try:
                rets = get_wiki_search_result(w, mode)
            except:
                pass
            print(w, '\t'.join(rets), sep='\t', file=fw)

def print_results(terms, scores, phrase_id):
    for w in terms:
        print(f"{w}: {scores[phrase_id[w]]}")

# build core-anchored semantic graph with queries
def get_term_graph_with_query(core_nodes, phrase_id, domain, max_in_degree=5, additional_link=True):
    lemmatizer = NLTKLemmatizer()
    
    core_nodes = set(core_nodes)
    phrase_link_tmp_store = {}

    if additional_link:
        files = [f"wikipedia/ranking-results/phrase-wiki-search-results-1-{domain}.txt", \
                    f"tmp/phrase-wiki-search-results-1-query.txt"]
        for filename in files:
            f = open(filename)
            for line in f:
                line = line.split('\t')
                w1 = line[0]
                k = 0
                for w2 in line[1:]:
                    if k>=max_in_degree:
                        break
                    w2 = process_category(w2, lemmatizer)
                    if w1!=w2 and w2 in phrase_id and phrase_id[w2] in core_nodes:
                        if w1 in phrase_link_tmp_store:
                            phrase_link_tmp_store[w1].append(w2)
                        else:
                            phrase_link_tmp_store[w1] = [w2]
                        k+=1
            f.close()

    row = []
    col = []
    files = [f"wikipedia/ranking-results/phrase-wiki-search-results-0-{domain}.txt", \
                    f"tmp/phrase-wiki-search-results-0-query.txt"]
    for filename in files:
        f = open(filename)
        for line in f.readlines():
            line = line.split('\t')
            w1 = line[0]
            k = 0
            
            # add self-link
            row.append(phrase_id[w1])
            col.append(phrase_id[w1])
            
            for w2 in line[1:]:
                if k>=max_in_degree:
                    break
                w2 = process_category(w2, lemmatizer)
                if w1!=w2 and w2 in phrase_id and phrase_id[w2] in core_nodes:
                    row.append(phrase_id[w2])
                    col.append(phrase_id[w1])
                    k+=1
            if additional_link and k<5:
                if w1 in phrase_link_tmp_store:
                    for w2 in phrase_link_tmp_store[w1]:
                        if k>=max_in_degree:
                            break
                        if w2 not in line[1:]:
                            row.append(phrase_id[w2])
                            col.append(phrase_id[w1])
                            k+=1
        f.close()
                            
    A = SparseTensor(row=torch.LongTensor(row), col=torch.LongTensor(col))
    A = A.to_symmetric()
    return A


def parse_args():
    parser = argparse.ArgumentParser(description="Run baselines.")
    
    parser.add_argument('--domain', choices=['cs', 'phy', 'math'],
                        help='The target domain.')
    parser.add_argument('--narrow', action='store_true',
                        help='Training and evaluating on the corresponding subdomains.')
    parser.add_argument('--method', choices=['cfl', 'hicfl'],
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
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--device', type=int, default=0)
    return parser.parse_args()
    

def main(args):
    query_terms = [
        "machine learning",
        "few-shot learning",
        "long-short term memory",
        "social network",
        "frequency assignment problem",
        "data sparseness",
        "large neighborhood search",
        "multi-hop wireless networks",
        "signal prediction",
        "molecule",
        "gravity",
        "animism",
        "backflow",
        "calcite",
        "supply and demand",
        "hellbent on compromise",
        "anatahan"
    ]

    print("Domain:", args.domain)
    print("Method:", args.method)
    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    domain = args.domain
    
    lemmatizer = NLTKLemmatizer()
    for i,term in enumerate(query_terms):
        query_terms[i] = process_category(term, lemmatizer)

    # load seed terms
    phrase_id, phrases = get_phrases(f"term-candidates/arxiv-phrases-filtering-{domain}.txt")

    # include query terms
    TID = len(phrases)
    tid = TID
    for w in query_terms:
        if w not in phrase_id:
            phrase_id[w] = tid
            phrases.append(w)
            tid += 1

    get_wiki_search_result_batch(phrases[TID:], mode=0)
    get_wiki_search_result_batch(phrases[TID:], mode=1)
    
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
    

    # split
    def train_test_split_full(seed_labels, seed=0):
        random.seed(seed)
        
        split_idx = {}
        split_y = {}
        
        candicates = list(seed_labels.keys())
        random.shuffle(candicates)
        n = len(candicates)
        
        # use all core terms for training, ugly implementation
        split_idx["train"] = candicates[:]
        split_idx["valid"] = candicates[:]
        split_idx["test"] = candicates[:]

        split_y["train"] = [seed_labels[i] for i in split_idx["train"]]
        split_y["valid"] = [seed_labels[i] for i in split_idx["valid"]]
        split_y["test"] = [seed_labels[i] for i in split_idx["test"]]
        
        return split_idx, split_y

    # load train/valid/test split
    # split_idx, split_y = load_train_valid_test_split(seed_labels, domain) # normal train/valid/test split
    split_idx, split_y = train_test_split_full(seed_labels) # use all core terms for training
    
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
            # idx_, y_ = load_train_valid_test_split(d, domain)
            idx_, y_ = train_test_split_full(seed_labels) # use all core terms for training
            for key,value in y_.items():
                y_[key] = torch.LongTensor(value).to(device)
            list_split_y.append(y_)
    
        if args.pu:
            list_split_y[-1] = split_y_pu
    
    # load compositional GloVe embeddings
    X = load_embeddings_glove('features/glove.6B.100d.txt', phrases)  # C
    
    X = X.to(device)
    num_features = X.size()[1]
    
    # build core-anchored semantic graph for CFL/HiCFL
    # A = get_term_graph_with_query(split_idx["train"], phrase_id, domain)
    A = get_term_graph_with_query(seed_labels.keys(), phrase_id, domain)
    A = A.to(device)

    dirs = "model"
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    
    if args.method=="cfl":
        model = CFL(num_features, args.hidden_channels, args.num_classes, args.num_layers, args.dropout).to(device)

        if os.path.exists(f'model/cfl-{domain}-c.pt'):
            model.load_state_dict(torch.load(f'model/cfl-{domain}-c.pt'))
        else:
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
                
                print(f"Epoch: {epoch}, AUC-ROC: {aucs}, AUC-PR: {aps}")

        torch.save(model.state_dict(), f"model/cfl-{domain}-c.pt")

        y_scores = predict_cfl(model, X, A)
        y_scores = y_scores.cpu().numpy()
    
    elif args.method=="hicfl":
        model = HiCFL(num_features, args.hidden_channels, args.num_classes, args.num_layers, num_hierarchy, args.dropout).to(device)
        
        if os.path.exists(f'model/hicfl-{domain}-c.pt'):
            model.load_state_dict(torch.load(f"model/hicfl-{domain}-c.pt"))
        else:
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
                
                print(f"Epoch: {epoch}, AUC-ROC: {aucs}, AUC-PR: {aps}")

        torch.save(model.state_dict(), f"model/hicfl-{domain}-c.pt")
        
        y_scores = predict_hicfl(model, X, A, args.alpha)
        y_scores = y_scores.cpu().numpy()
    
    print_results(query_terms, y_scores, phrase_id)


if __name__ == "__main__":
    args = parse_args()
    main(args)


