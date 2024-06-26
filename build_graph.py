# import torch
# import torch.nn as nn
# import torch.nn.functional as F
from tqdm.auto import tqdm
import scipy.sparse as sp
from math import log
import numpy as np
import torch
import math
from torch import nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
from torch.optim import AdamW

def ordered_word_pair(a, b):
    if a > b:
        return b, a
    else:
        return a, b

class LSTM_classifier(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, num_labels, dropout) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(input_size=emb_size, hidden_size=hidden_size,num_layers=1, batch_first=True, dropout=dropout, bidirectional=False)
        self.classifier = nn.Linear(hidden_size, num_labels)
    def forward(self, inputs):
        emb = self.embedding(inputs)
        output, (h_n, c_n) = self.lstm(emb)
        inter_output = torch.mean(output, dim=1)
        res = self.classifier(inter_output)
        return output, res

def trans_corpus_to_ids(corpus, word_id_map, max_len):
    new_corpus = []
    for text in corpus:
        word_list = text.split()
        if len(word_list) > max_len:
            word_list = word_list[:max_len]
        new_corpus.append([word_id_map[w] + 1 for w in word_list]) # + 1 for padding
    # padding
    for i, one in enumerate(new_corpus):
        if len(one) < max_len:
            new_corpus[i] = one + [0]*(max_len-len(one))
    new_corpus = np.asarray(new_corpus, dtype=np.int32)
    return new_corpus

def lstm_eval(model, dataloader, device):
    model.eval()
    all_preds, all_labels,all_outs = [],[],[]
    for batch in dataloader:
        batch = [one.to(device) for one in batch]
        x, y = batch
        with torch.no_grad():
            output, pred = model(x)
            all_outs.append(output.cpu().numpy())
            pred_ids = torch.argmax(pred, dim=-1)
            all_preds += pred_ids.tolist()
            all_labels += y.tolist()
    acc = np.mean(np.asarray(all_preds) == np.asarray(all_labels))

    all_outs = np.concatenate(all_outs, axis=0)

    model.train()
    return acc, all_outs

def train_lstm(corpus, word_id_map, train_size, valid_size, labels, emb_size, hidden_size, dropout, batch_size, epochs, lr, weight_decay, num_labels,device,max_len):
    vocab_size = len(word_id_map) + 1
    corpus_ids = trans_corpus_to_ids(corpus, word_id_map, max_len)
    model = LSTM_classifier(vocab_size, emb_size, hidden_size, num_labels, dropout)
    model.to(device)
    train_data = corpus_ids[:train_size,:]
    dev_data = corpus_ids[train_size:train_size+valid_size,:]

    test_data = corpus_ids[train_size+valid_size:,:]
    train_label = labels[:train_size]
    dev_label = labels[train_size:train_size+valid_size]
    test_label = labels[train_size+valid_size:]
    train_x = torch.tensor(train_data, dtype=torch.long)
    train_y = torch.tensor(train_label, dtype=torch.long)
    dev_x = torch.tensor(dev_data, dtype=torch.long)
    dev_y = torch.tensor(dev_label, dtype=torch.long)
    test_x = torch.tensor(test_data, dtype=torch.long)
    test_y = torch.tensor(test_label, dtype=torch.long)
    train_dataset = TensorDataset(train_x, train_y)
    dev_dataset = TensorDataset(dev_x, dev_y)
    test_dataset = TensorDataset(test_x, test_y)
    train_sampler = RandomSampler(train_dataset)
    train_dev_sampler = SequentialSampler(train_dataset)
    dev_sampler = SequentialSampler(dev_dataset)
    test_sampler = SequentialSampler(test_dataset)
    train_dataloader = DataLoader(train_dataset,batch_size,sampler=train_sampler)
    train_dev_dataloader = DataLoader(train_dataset,batch_size,sampler=train_dev_sampler)
    dev_dataloader = DataLoader(dev_dataset,batch_size,sampler=dev_sampler)
    test_dataloader = DataLoader(test_dataset,batch_size,sampler=test_sampler)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    loss_func = nn.CrossEntropyLoss(reduction='mean')
    best_acc = 0.0
    for ep in range(epochs):
        for batch in tqdm(train_dataloader):
            batch = [one.to(device) for one in batch]
            x, y = batch
            output, pred = model(x)
            loss = loss_func(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # import pudb;pu.db
        acc, all_outs = lstm_eval(model, dev_dataloader, device)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'lstm.bin')
            print("current best acc={:4f}".format(acc))
    # model.load_state_dict(torch.load('lstm.bin'))
    acc, all_outs_train = lstm_eval(model, train_dev_dataloader, device)
    acc, all_outs_dev = lstm_eval(model, dev_dataloader, device)
    acc, all_outs_test = lstm_eval(model, test_dataloader, device)
    # import pudb;pu.db
    all_outs = np.concatenate([all_outs_train, all_outs_dev, all_outs_test], axis=0)
    return model, all_outs, corpus_ids
#
#
#
def get_adj(tokenize_sentences, train_size, word_id_map, word_list, args, train_labels, train_sentences, device):
    window_size = 20
    total_W = 0
    word_occurrence = {}
    word_pair_occurrence = {}

    node_size = train_size + len(word_list)
    vocab_length = len(word_list)

    def update_word_and_word_pair_occurrence(q):
        unique_q = list(set(q))
        for i in unique_q:
            try:
                word_occurrence[i] += 1
            except:
                word_occurrence[i] = 1
        for i in range(len(unique_q)):
            for j in range(i + 1, len(unique_q)):
                word1 = unique_q[i]
                word2 = unique_q[j]
                word1, word2 = ordered_word_pair(word1, word2)
                try:
                    word_pair_occurrence[(word1, word2)] += 1
                except:
                    word_pair_occurrence[(word1, word2)] = 1

    if not args.easy_copy:
        print("Calculating PMI")
    for ind in range(train_size):
        words = tokenize_sentences[ind]

        q = []
        # push the first (window_size) words into a queue
        for i in range(min(window_size, len(words))):
            q += [word_id_map[words[i]]]
        # update the total number of the sliding windows
        total_W += 1
        # update the number of sliding windows that contain each word and word pair
        update_word_and_word_pair_occurrence(q)

        now_next_word_index = window_size
        # pop the first word out and let the next word in, keep doing this until the end of the document
        while now_next_word_index < len(words):
            q.pop(0)
            q += [word_id_map[words[now_next_word_index]]]
            now_next_word_index += 1
            # update the total number of the sliding windows
            total_W += 1
            # update the number of sliding windows that contain each word and word pair
            update_word_and_word_pair_occurrence(q)

    # calculate PMI for edges
    row = []
    col = []
    weight = []

    corpus = train_sentences
    train_size2 = train_size
    valid_size = int(0.1 * train_size)
    test_size = int(0.3 * train_size)
    train_size = train_size-valid_size-test_size

    le = LabelEncoder()
    le.fit(train_labels)
    labels = le.transform(train_labels)
#
#
    embed_size=200
    hidden_size = 200
    dropout = 0
    batch_size = 32
    epochs = 20 #20
    lr = 0.001
    weight_decay = 1e-6
    max_len = 512 #512
    num_labels = 2
    thres = 0.05 #0.05
    # import pudb;pu.db
    model, all_outs, corpus_ids = train_lstm(corpus, word_id_map, train_size, valid_size, labels, embed_size,
                                             hidden_size, dropout, batch_size, epochs, lr,
                                             weight_decay, num_labels, device, max_len)
    num_docs = all_outs.shape[0]
    cos_simi_count = {}
    for i in tqdm(range(num_docs)):
        text = corpus[i]
        word_list2 = text.split()
        max_len = len(word_list2) if len(word_list2) < max_len else max_len
        x = all_outs[i, :, :]
        x_norm = np.linalg.norm(x, ord=2, axis=-1, keepdims=True)
        simi_mat = np.dot(x, x.T) / np.dot(x_norm, x_norm.T)  # L * L
        for k in range(max_len):
            for j in range(k + 1, max_len):
                word_k_id = word_id_map[word_list2[k]]
                word_j_id = word_id_map[word_list2[j]]
                simi = simi_mat[k, j]
                if word_k_id == word_j_id:
                    continue
                if simi > thres:
                    word_pair_str = str(word_k_id) + ',' + str(word_j_id)
                    # import pudb;pu.db
                    if word_pair_str in cos_simi_count:
                        cos_simi_count[word_pair_str] += 1
                    else:
                        cos_simi_count[word_pair_str] = 1
                    # two orders
                    word_pair_str = str(word_j_id) + ',' + str(word_k_id)
                    if word_pair_str in cos_simi_count:
                        cos_simi_count[word_pair_str] += 1
                    else:
                        cos_simi_count[word_pair_str] = 1
    max_count = 0
    min_count = 1000000

    for v in cos_simi_count.values():
        if v < min_count:
            min_count = v
        if v > max_count:
            max_count = v

    for key in cos_simi_count:
        temp = key.split(',')
        # if temp[0] not in word_id_map or temp[1] not in word_id_map:
        #     continue
        i = int(temp[0])
        j = int(temp[1])
        w = (cos_simi_count[key] - min_count) / (max_count - min_count)
        row.append(train_size2 + i)
        col.append(train_size2 + j)
        # import pudb;pu.db
        weight.append(w)
        row.append(train_size2 + j)
        col.append(train_size2 + i)
        weight.append(w)






    # for word_pair in word_pair_occurrence:
    #     i = word_pair[0]
    #     j = word_pair[1]
    #     count = word_pair_occurrence[word_pair]
    #     word_freq_i = word_occurrence[i]
    #     word_freq_j = word_occurrence[j]
    #     pmi = log((count * total_W) / (word_freq_i * word_freq_j))
    #     if pmi <= 0:
    #         continue
    #     row.append(train_size + i)
    #     col.append(train_size + j)
    #     weight.append(pmi)
    #     row.append(train_size + j)
    #     col.append(train_size + i)
    #     weight.append(pmi)
    # if not args.easy_copy:
    #     print("PMI finished.")

    # get each word appears in which document

    word_doc_list = {}
    for word in word_list:
        word_doc_list[word] = []

    for i in range(train_size2):
        doc_words = tokenize_sentences[i]
        unique_words = set(doc_words)
        for word in unique_words:
            # import pudb;pu.db
            exsit_list = word_doc_list[word]
            exsit_list.append(i)
            word_doc_list[word] = exsit_list

    # document frequency
    word_doc_freq = {}
    for word, doc_list in word_doc_list.items():
        word_doc_freq[word] = len(doc_list)

    # term frequency
    doc_word_freq = {}

    for doc_id in range(train_size2):
        words = tokenize_sentences[doc_id]
        for word in words:
            word_id = word_id_map[word]
            doc_word_str = str(doc_id) + ',' + str(word_id)
            if doc_word_str in doc_word_freq:
                doc_word_freq[doc_word_str] += 1
            else:
                doc_word_freq[doc_word_str] = 1

    doc_emb = np.zeros((train_size2, vocab_length))

    for i in range(train_size2):
        words = tokenize_sentences[i]
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            j = word_id_map[word]
            key = str(i) + ',' + str(j)
            freq = doc_word_freq[key]
            row.append(i)
            col.append(train_size2 + j)
            idf = log(1.0 * train_size2 / word_doc_freq[word_list[j]])
            w = freq * idf
            weight.append(w)
            doc_word_set.add(word)
            doc_emb[i][j] = w / len(words)


    adj = sp.csr_matrix((weight, (row, col)), shape=(node_size, node_size))

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return adj, doc_emb, word_doc_freq


    # import pudb;pu.db
    # packet_list = []
    #
    # for i in range(train_size):
    #     packet = ' '
    #     for k in range(len(tokenize_sentences[i])):
    #         if len(tokenize_sentences[i][k])==8:
    #             packet = packet + tokenize_sentences[i][k]
    #     packet_list.append(packet.strip())
        # import pudb;pu.db

    # flow_emb = np.zeros((train_size, train_size))
    # for i in range(train_size):
    #     for j in range(train_size):
    #
    #         # flow_emb
    #         if i != j:
    #
    #             a = levenshteinDistance(doc_emb[i], doc_emb[j])
    #             row.append(i)
    #             col.append(j)
    #             weight.append(a)
    #             import pudb;pu.db



