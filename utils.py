import os, re, sys
import pickle
import pandas as pd
import numpy as np
from Bio import SeqIO
import itertools
from collections import Counter
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit,StratifiedKFold,KFold


def read_fasta_to_df(file_path):
    records = list(SeqIO.parse(file_path, 'fasta'))
    df = pd.DataFrame({"Description":["|".join(str(record.description).split("|")[:2]) for record in records], 'Sequence': [str(record.seq) for record in records], 'Label': [",".join(str(record.id).split('|')[-1].split(',')) for record in records]})
    return df

def read_nucleotide_sequences(file):
    if os.path.exists(file) == False:
        print('Error: file %s does not exist.' % file)
        sys.exit(1)
    with open(file) as f:
        records = f.read()
    if re.search('>', records) == None:
        print('Error: the input file %s seems not in FASTA format!' % file)
        sys.exit(1)
    records = records.split('>')[1:]
    fasta_sequences = []
    for fasta in records:
        array = fasta.split('\n')
        header, sequence = array[0].split()[0], re.sub('[^ACGTU]', '-', ''.join(array[1:]).upper())
        header_array = header.split('|')
        name = header_array[0]
        label = header_array[-1] if len(header_array) >= 2 else '0'
        label_train = header_array[-1] if len(header_array) >= 3 else 'training'
        sequence = re.sub('U', 'T', sequence)
        fasta_sequences.append([name, sequence, label, label_train])
    return fasta_sequences


def print_label_counts(train_df, test_df, labels_column):
    print(f"Train set contains {len(train_df)} sequences, and labels: ")
    train_labels = []
    for row in df.itertuples():
        locations = str(row.Label).split(',')
        train_labels.extend(locations)
    train_counts = Counter(train_labels)
    print(train_counts)

    print(f"Test set contains {len(test_df)} sequences, and labels: ")
    test_labels = []
    for row in test_df.itertuples():
        locations = str(row.Label).split(',')
        test_labels.extend(locations)
    test_counts = Counter(test_labels)
    print(test_counts)

def kmerArray(sequence, k):
    kmer = []
    for i in range(len(sequence) - k + 1):
        kmer.append(sequence[i:i + k])
    return kmer

def Kmer(fastas, k=2, type="DNA", upto=False, normalize=True, **kw):
    # encoding = []
    fastas = read_nucleotide_sequences(fastas)
    encoding = {}
    header = ['#', 'label']
    NA = 'ACGT'
    if type in ("DNA", 'RNA'):
        NA = 'ACGT'
    else:
        NA = 'ACDEFGHIKLMNPQRSTVWY'
    if k < 1:
        print('Error: the k-mer value should larger than 0.')
        return 0
    if upto == True:
        for tmpK in range(1, k + 1):
            for kmer in itertools.product(NA, repeat=tmpK):
                header.append(''.join(kmer))
        # encoding.append(header)
        for i in fastas:
            seq_key=i[1]
            name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
            count = Counter()
            for tmpK in range(1, k + 1):
                kmers = kmerArray(sequence, tmpK)
                count.update(kmers)
                if normalize == True:
                    for key in count:
                        if len(key) == tmpK:
                            count[key] = count[key] / len(kmers)
            # code = [name, label]
            code=[]
            for j in range(2, len(header)):
                if header[j] in count:
                    code.append(count[header[j]])
                else:
                    code.append(0)
            encoding[seq_key]=code
            # encoding.append(code)
    else:
        for kmer in itertools.product(NA, repeat=k):
            header.append(''.join(kmer))
        # encoding.append(header)
        for i in fastas:
            seq_key=i[1]
            name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
            kmers = kmerArray(sequence, k)
            count = Counter()
            count.update(kmers)
            if normalize == True:
                for key in count:
                    count[key] = count[key] / len(kmers)
            # code = [name, label]
            code = []
            for j in range(2, len(header)):
                if header[j] in count:
                    code.append(count[header[j]])
                else:
                    code.append(0)
            # encoding.append(code)
            encoding[seq_key]=code
    return encoding

def linear_fold(sequences, ids, out_fasta_name):
    for seq, id in zip(sequences, ids):
        # if not id in processed_ids:
        print(f"{id} is processing...")
        with open("tmp.fasta", "w") as ofile: 
            ofile.write(f">{id}\n{seq}\n")
        os.system(f"cat tmp.fasta | ./linearfold_v > tmp.dot") 
        in_lines = open("tmp.dot","r").readlines()
        with open("clean_tmp.dot","w") as out_file:
            for line in in_lines:
                if ">" in line: # extract just the ID
                    out_file.write(':'.join(line.split(":")[1:]).strip() + "\n")
                    # out_file.write(line+ "\n")
                else:
                    out_file.write(line)
        os.system("cat " + "clean_tmp.dot" + " >> " + out_fasta_name + ".fasta") 

def compute_DACC(fasta_path, lag=2, seq_type='RNA', ilearn_base_dir=None):
    """
    Compute DACC (autocorrelation) encoding for sequences using iLearn; returns dict[sequence -> feature vector].
    By default uses DACC files under this repo's reference/ilearn.
    """
    if ilearn_base_dir is None:
        _cur_dir = os.path.dirname(os.path.abspath(__file__))
        ilearn_base_dir = os.path.join(_cur_dir, 'reference', 'ilearn')
    if not os.path.isdir(ilearn_base_dir):
        raise FileNotFoundError(f"iLearn directory not found: {ilearn_base_dir}. Set ilearn_base_dir or place DACC files under reference/ilearn.")
    if ilearn_base_dir not in sys.path:
        sys.path.insert(0, ilearn_base_dir)
    from pubscripts.read_fasta_sequences import read_nucleotide_sequences
    from descnucleotide.ACC import make_acc_vector
    from descnucleotide import check_parameters
    myPropertyName = check_parameters.myDictDefault['DACC'][seq_type]
    dataFile = check_parameters.myDataFile['DACC'][seq_type]
    data_path = os.path.join(ilearn_base_dir, 'data')
    with open(os.path.join(data_path, dataFile), 'rb') as f:
        myProperty = pickle.load(f)
    myPropertyValue = {k: myProperty[k] for k in myPropertyName}
    kmer = check_parameters.myKmer['DACC']
    fastas = read_nucleotide_sequences(fasta_path)
    encodings = make_acc_vector(fastas, myPropertyName, myPropertyValue, lag, kmer)
    encoding_dict = {}
    for enc in encodings[1:]:
        seq_key = enc[1]
        encoding_dict[seq_key] = [float(x) for x in enc[2:]]
    return encoding_dict


def dot_fasta_to_pkl(file, out_pkl):
    with open(file) as f:
        records=f.read()
    if re.search('>', records) == None:
        print('Error: the input file %s seems not in FASTA format!' % file)
        sys.exit(1)
    records = records.split('>')[1:]
    # print(records)
    seq_dotbracket = {} # 
    for fasta in records:
        valueList=[]
        array = fasta.split('\n')
        sequence,dot_bracket =array[1],array[2]
        sequence = re.sub('U', 'T', sequence)
        if 'N' in sequence:
            sequence = re.sub('N', 'G', sequence)
            print(array[0])
        dot_bracket_list=dot_bracket.split()
        # print(dot_bracket_list)
        # break
        ev=float(dot_bracket_list[1].split('(')[1].split(')')[0]) #.replace('\U00002013', '-')
        valueList.append(dot_bracket_list[0])
        valueList.append(ev)
        seq_dotbracket[sequence]=valueList
    with open(out_pkl, 'wb') as handle:
        pickle.dump(seq_dotbracket, handle)


def split_dataset_ensure_label(df, labels_column, locations, k=None, test_size=0.2, min_samples=10, random_state=42):

    df = df.copy() 
    X = df['Sequence'].values

    def encode_labels(label_string):
        labels = label_string.split(',')
        return [1 if label in labels else 0 for label in locations]

    y = np.array([encode_labels(label) for label in df[labels_column]])
    n_labels = y.shape[1]

    y_indices = {i: np.where(y[:, i] == 1)[0] for i in range(n_labels)}

    def _ensure_min_samples(indices):
        indices = set(indices)
        for i in range(n_labels):
            label_samples = np.intersect1d(list(indices), y_indices[i])
            missing_count = min_samples - len(label_samples)

            if missing_count > 0:
                additional_samples = set(y_indices[i]) - indices
                indices.update(list(additional_samples)[:missing_count])

        return np.array(list(indices))

    if k is not None:
        kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
        folds = []

        for train_index, test_index in kf.split(X):
            test_index = _ensure_min_samples(test_index) 
            train_index = np.setdiff1d(train_index, test_index) 

            folds.append((train_index, test_index))

        return folds

    else:
        test_indices = set()
        df['label_list'] = df[labels_column].str.split(',')

        for label in locations:
            label_samples = df[df['label_list'].apply(lambda x: label in x)]
            selected_samples = (
                label_samples.index if len(label_samples) <= min_samples 
                else label_samples.sample(n=min_samples, random_state=random_state).index
            )
            test_indices.update(selected_samples)

        remaining_samples = df.drop(index=test_indices)
        extra_train, extra_test = train_test_split(
            remaining_samples, test_size=test_size, random_state=42
        )

        test_indices.update(extra_test.index)
        train_indices = np.setdiff1d(df.index, list(test_indices)) 

        return np.array(train_indices), np.array(list(test_indices))

import numpy as np
from scipy.stats import fisher_exact

def build_R_from_Y(Y, min_cooccur=5, pval_thresh=0.05, tau=0.0, beta=1.0, prior=None):
    # Y: (N, L) binary numpy
    N, L = Y.shape
    n = Y.sum(axis=0)                # [L]
    # co-occurrence matrix
    n_ij = Y.T.dot(Y)                # [L, L]
    P = n / N
    Pij = n_ij / N + 1e-12

    # PMI and NPMI
    PMI = np.log(Pij / (P[:,None]*P[None,:] + 1e-12) + 1e-12)
    NPMI = PMI / (-np.log(Pij + 1e-12) + 1e-12)
    NPMI = np.nan_to_num(NPMI)       # replace possible nans

    # significance mask via Fisher exact test (symmetric)
    sig_mask = np.zeros((L,L), dtype=float)
    for i in range(L):
        for j in range(i+1, L):
            a = int(n_ij[i,j])                         # both
            b = int(n[i] - a)                         # i only
            c = int(n[j] - a)                         # j only
            d = int(N - (a+b+c))                      # neither
            # fisher
            _, p = fisher_exact([[a,b],[c,d]], alternative='two-sided')
            if (p <= pval_thresh) and (a >= min_cooccur):
                sig_mask[i,j] = sig_mask[j,i] = 1.0

    # apply mask and threshold
    M = NPMI * sig_mask
    # optional thresholding to remove tiny values
    M[np.abs(M) < tau] = 0.0

    # confidence weight (optional)
    conf = n_ij / (n_ij + 10.0)   # tune 10.0
    M = M * conf

    # normalize to [-1,1]
    if np.max(np.abs(M)) > 0:
        M = M / np.max(np.abs(M))

    # combine with prior
    if prior is not None:
        M = beta * M + (1-beta) * prior

    M = M / (M.sum() + 1e-12)  # normalize
    # k = 2  # sparsify: keep only top-k strong correlations
    # for i in range(M.shape[0]):
    #     idx = np.argsort(M[i])[:-k]  # exclude top-k
    #     M[i, idx] = 0
    return M  # this is R

import torch
import torch.nn.functional as F

# R: numpy LxL in [-1,1] -> convert to torch


def corr_loss_from_probs(p, R):
    # p: [B, L], probs after sigmoid
    R_t = torch.tensor(R, dtype=torch.float32, device=p.device)  # [L,L]
    R_pos = torch.clamp(R_t, min=0.0)
    R_neg = -torch.clamp(R_t, max=0.0)

    diff = p.unsqueeze(2) - p.unsqueeze(1)    # [B, L, L]
    pos_term = (diff**2) * R_pos.unsqueeze(0) # broadcast
    sum_pos = pos_term.sum(dim=(1,2))         # [B]

    sum_p = (p.unsqueeze(2) + p.unsqueeze(1) - 1.0)
    neg_term = (sum_p**2) * R_neg.unsqueeze(0)
    sum_neg = neg_term.sum(dim=(1,2))         # [B]

    # average over batch
    return (sum_pos + sum_neg).mean()

def cal_loss(outputs, targets, R, lambda_corr =0.5):
    # training:
    bce = F.binary_cross_entropy(outputs, outputs)
    corr = corr_loss_from_probs(outputs, R)
    loss = bce + lambda_corr * corr
    loss.backward()

def save_fasta_from_dataset(dataset, fasta_path):
    """
    Export des and rowseq from RNAGraphDataset to a FASTA file.
    """
    with open(fasta_path, "w") as f:
        for data in dataset:
            desc = data.des if hasattr(data, "des") else "unknown"
            seq = data.rowseq if hasattr(data, "rowseq") else ""
            f.write(f">{desc}\n{seq}\n")
    print(f"FASTA saved to {fasta_path}")

def save_fasta_from_df(df, fasta_path):
    """
    Save DataFrame Description and Sequence columns to a FASTA file.
    """
    with open(fasta_path, "w") as f:
        for row in df.itertuples():
            desc = str(row.Description).strip()
            label = str(row.Label).strip()
            seq = str(row.Sequence).strip()
            if len(seq) < 6000:
                f.write(f">{desc}|{label}\n{seq}\n")
    print(f"FASTA saved to {fasta_path}")
