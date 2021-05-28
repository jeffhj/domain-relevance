import numpy as np
import random
from lemmatizer import NLTKLemmatizer
from gensim.models import KeyedVectors
import torch
from torch_sparse.tensor import SparseTensor


# load term candidates
def get_phrases(path):
    phrase_id = {}
    phrases = []
    with open(path) as fr:
        tid = 0
        for line in fr.readlines():
            w,v = line.split('\t')
            phrase_id[w] = tid
            phrases.append(w)
            tid += 1
    return phrase_id, phrases

# load word2vec embeddings
def load_embeddings(path, phrases):
    wv = KeyedVectors.load(path)
    X = []
    dim = len(wv[phrases[0].replace(' ', '_')])
    
    for phrase in phrases:
        phrase = phrase.replace(' ', '_')
        if phrase in wv:
            X.append(wv[phrase])
        else:
            X.append(np.random.rand(dim))
    
    X = torch.FloatTensor(X)
    return X

# load GloVe embeddings
def load_embeddings_glove(path, phrases):
    gloveModel = {}
    with open(path) as f:
        for line in f:
            line = line.split()
            word = line[0]
            emb = np.array([float(v) for v in line[1:]])
            gloveModel[word] = emb
            
    dim = len(emb)
    X = []
    for phrase in phrases:
        ws = phrase.split(' ')
        emb = np.zeros(dim)
        
        for w in ws:
            if w in gloveModel:
                emb += gloveModel[w]
        X.append(emb)
    X = torch.FloatTensor(X)
    return X

# load train/valid/test split
def load_train_valid_test_split(seed_labels, domain):
    def load_ids(path):
        ids = []
        with open(path) as f:
            for line in f:
                ids.append(int(line.strip()))
        return ids
    
    split_idx = {}
    split_y = {}
    
    split_idx["train"] = load_ids(f'train-valid-test/{domain}/train.txt')
    split_idx["valid"] = load_ids(f'train-valid-test/{domain}/valid.txt')
    split_idx["test"] = load_ids(f'train-valid-test/{domain}/test.txt')
    
    split_y["train"] = [seed_labels[i] for i in split_idx["train"]]
    split_y["valid"] = [seed_labels[i] for i in split_idx["valid"]]
    split_y["test"] = [seed_labels[i] for i in split_idx["test"]]
    
    return split_idx, split_y

# train/valid/test split for pu learning
def train_test_split_for_pu(idx, y, core_labels_p, positives=None, k=20, seed=10):
    random.seed(seed)
    
    if positives==None:
        idx_pos = []
        for i,c in enumerate(y):
            if c:
                idx_pos.append(idx[i])
        idxs_pos_sample = set(random.sample(idx_pos, k))
    else:
        idxs_pos_sample = set(positives)
    
    ret_idx = []
    ret_y = []
    for wid in idx:
        if wid in idxs_pos_sample:
            ret_idx.append(wid)
            ret_y.append(1)
        elif not core_labels_p[wid]:
            ret_idx.append(wid)
            ret_y.append(0)
            
    return ret_idx, ret_y


def process_category(c, lemmatizer):
    if '(' in c:
        c = c[:c.find('(')-1]
    c = lemmatizer.lemmatize_phrase(c.lower())
    return c

# get labels of core terms
def get_core_phrase_label(root, wc_path, phrase_id, category_pedia, category_media, seed_option="combine"):
    # root = "computer science"
    # wc_path = "wikipedia-category-Subfields_of_computer_science-3.txt"
    assert seed_option in ["media", "category", "combine"]
    
    lemmatizer = NLTKLemmatizer()
    true_category_count = {}
    
    gold_terms = set()
    root = process_category(root, lemmatizer)
    gold_terms.add(root)
    with open(wc_path) as f:
        for line in f:
            level,w = line.split('#')
            w = process_category(w, lemmatizer)
            gold_terms.add(w)

    pedia_label = {}
    pedia_label[phrase_id[root]] = 1
    with open(category_pedia) as fr:
        for line in fr:
            line = line.split('\t')
            w = line[0]
            categories = line[1:]
            
            label = 0
            if w in gold_terms:
                label = 1
            else:
                for c in categories:
                    c = process_category(c, lemmatizer)
                    if c in gold_terms:
                        true_category_count[c] = true_category_count.get(c,0)+1
                        label = 1
                        break
            pedia_label[phrase_id[w]] = label
        
    media_label = {}
    media_label[phrase_id[root]] = 1
    with open(category_media) as fr:
        for line in fr:
            line = line.split('\t')
            w = line[0]
            categories = line[1:]
            
            label = 0
            if w in gold_terms:
                label = 1
            else:
                for c in categories:
                    c = process_category(c, lemmatizer)
                    if c in gold_terms:
                        label = 1
                        break
            media_label[phrase_id[w]] = label
    
    seed_labels = {}
    if seed_option == "media":
        seed_labels = media_label
    elif seed_option == "category":
        seed_labels = pedia_label
    elif seed_option == "combine":
        for w,c1 in pedia_label.items():
            if w in media_label:
                seed_labels[w] = c1 or media_label[w]
            else:
                seed_labels[w] = c1   

    return seed_labels

# build core-anchored semantic graph
def get_term_graph(core_nodes, phrase_id, domain, max_in_degree=5, additional_link=True):
    lemmatizer = NLTKLemmatizer()
    
    core_nodes = set(core_nodes)
    phrase_link_tmp_store = {}

    if additional_link:
        with open(f"wikipedia/ranking-results/phrase-wiki-search-results-1-{domain}.txt") as f:
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

    row = []
    col = []
    with open(f"wikipedia/ranking-results/phrase-wiki-search-results-0-{domain}.txt") as f:
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
                            
    A = SparseTensor(row=torch.LongTensor(row), col=torch.LongTensor(col))
    A = A.to_symmetric()
    return A
