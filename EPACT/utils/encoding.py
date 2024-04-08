import torch
import numpy as np


BLOSUM50_MATRIX = np.array([
    [5, -2, -1, -2, -1, -3, 0, -2, -1, -2, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
    [-2, 7, -1, -2, -4, 1, -1, -3, 0, -4, -3, 3, -2, -2, -3, -1, -1, -3, -1, -3],
    [-1, -1, 7, 2, -2, 0, 0, 0, 1, -3, -4, 0, -2, -4, -2, 1, 0, -4, -2, -3],
    [-2, -2, 2, 8, -4, 0, 2, -1, -1, -4, -4, -1, -4, -5, -1, 0, -1, -5, -3, -4],
    [-1, -4, -2, -4, 11, -3, -4, -3, -4, -5, -5, -3, -5, -4, -3, -1, -1, -4, -4, -1],
    [-3, 1, 0, 0, -3, 7, -1, -2, 1, -2, -1, 2, 0, -3, -2, 0, -1, -1, -1, -2],
    [0, -1, 0, 2, -4, -1, 7, -2, -1, -4, -3, 0, -2, -4, -1, 0, -1, -4, -3, -3],
    [-2, -3, 0, -1, -3, -2, -2, 5, -2, -3, -3, -2, -3, -3, -2, 0, -2, -3, -3, -4],
    [-1, 0, 1, -1, -4, 1, -1, -2, 5, -3, -2, 1, 0, -3, -1, 0, -1, -3, -2, -3],
    [-2, -4, -3, -4, -5, -2, -4, -3, -3, 5, 2, -3, 2, 0, -3, -3, -2, -2, 1, 4],
    [-1, -3, -4, -4, -5, -1, -3, -3, -2, 2, 5, -2, 3, 1, -4, -3, -2, -2, 0, 1],
    [-1, 3, 0, -1, -3, 2, 0, -2, 1, -3, -2, 6, -2, -4, -1, 0, -1, -3, -2, -3],
    [-1, -2, -2, -4, -5, 0, -2, -3, 0, 2, 3, -2, 7, 0, -4, -2, -1, -1, 1, 2],
    [-2, -2, -4, -5, -4, -3, -4, -3, -3, 0, 1, -4, 0, 8, -3, -2, -2, 4, 4, 0],
    [-1, -3, -2, -1, -3, -2, -1, -2, -1, -3, -4, -1, -4, -3, 9, -1, -1, -3, -3, -3],
    [1, -1, 1, 0, -1, 0, 0, 0, 0, -3, -3, 0, -2, -2, -1, 5, 2, -4, -2, -1],
    [0, -1, 0, -1, -1, -1, -1, -2, -1, -2, -2, -1, -1, -2, -1, 2, 5, -3, -2, 0],
    [-3, -3, -4, -5, -4, -1, -4, -3, -3, -2, -2, -3, -1, 4, -3, -4, -3, 15, 2, -1],
    [-2, -1, -2, -3, -4, -1, -3, -3, -2, 1, 0, -2, 1, 4, -3, -2, -2, 2, 8, 0],
    [0, -3, -3, -4, -1, -2, -3, -4, -3, 4, 1, -3, 2, 0, -3, -1, 0, -1, 0, 5]
])


def one_hot_encoding(seq):
    # List of 20 standard amino acids.
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
                   'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    # Mapping from amino acids to indices
    amino_acid_to_index = {aa: i for i, aa in enumerate(amino_acids)}
    # Length of the protein sequence
    seq_len = len(seq)
    # Initialize an empty one-hot encoded matrix
    encoded_seq = np.zeros((seq_len, 20))
    # Encode the protein sequence
    for i, aa in enumerate(seq):
        if aa in amino_acids:
            encoded_seq[i, amino_acid_to_index[aa]] = 1

    return encoded_seq


def bolossum_encoding(seq, matrix):
    # List of 20 standard amino acids.
    amino_acids = "ARNDCQEGHILKMFPSTWYV"
    # Mapping from amino acids to indices
    amino_acid_to_index = {aa: i for i, aa in enumerate(amino_acids)}
    # Length of the protein sequence
    seq_len = len(seq)
    # Initialize an empty one-hot encoded matrix
    encoded_seq = np.full((seq_len, 20), -1)
    # Encode the protein sequence
    for i, aa in enumerate(seq):
        if aa in amino_acids:
            encoded_seq[i] = matrix[amino_acid_to_index[aa]]

    return encoded_seq


def atchley_factor_encoding(seq):
    atchley_factors = {
        'A': [-0.591, -1.302, -0.733, 1.570, -0.146],
        'C': [-1.343, 0.465, -0.862, -1.020, -0.255],
        'D': [1.050, 0.302, -3.656, -0.259, -3.242],
        'E': [1.357, -1.453, 1.477, 0.113, -0.837],
        'F': [-1.006, -0.590, 1.891, -0.397, 0.412],
        'G': [-0.384, 1.652, 1.330, 1.045, 2.064],
        'H': [0.336, -0.417, -1.673, -1.474, -0.078],
        'I': [-1.239, -0.547, 2.131, 0.393, 0.816],
        'K': [1.831, -0.561, 0.533, -0.277, 1.648],
        'L': [-1.019, -0.987, -1.505, 1.266, -0.912],
        'M': [-0.663, -1.524, 2.219, -1.005, 1.212],
        'N': [0.945, 0.828, 1.299, -0.169, 0.933],
        'P': [0.189, 2.081, -1.628, 0.421, -1.392],
        'Q': [0.931, -0.179, -3.005, -0.503, -1.853],
        'R': [1.538, -0.055, 1.502, 0.440, 2.897],
        'S': [-0.228, 1.399, -4.760, 0.670, -2.647],
        'T': [-0.032, 0.326, 2.213, 0.908, 1.313],
        'V': [-1.337, -0.279, -0.544, 1.242, -1.262],
        'W': [-0.595, 0.009, 0.672, -2.128, -0.184],
        'Y': [0.260, 0.830, 3.097, -0.838, 1.512]
    }

    # List of amino acids
    amino_acids = list(atchley_factors.keys())
    # Mapping from amino acids to indices
    amino_acid_to_index = {aa: i for i, aa in enumerate(amino_acids)}
    # Length of the protein sequence
    seq_len = len(seq)
    # Number of Atchley factors
    num_factors = len(atchley_factors[amino_acids[0]])
    # Initialize an empty matrix to store the encoded sequence
    encoded_seq = np.zeros((seq_len, num_factors))
    # Encode the protein sequence
    for i, aa in enumerate(seq):
        if aa in amino_acids:
            encoded_seq[i] = atchley_factors[aa]

    return encoded_seq


def pad_protein_sequence(sequence, max_seq_len, pad_token='X'):
    padded_sequence = sequence.ljust(max_seq_len, pad_token)
    return padded_sequence[:max_seq_len]


def hla_allele_to_seq(allele, hla_library):
    allele_ = allele.replace('HLA-', '')
    for key in hla_library.keys():
        if key.startswith(allele_):
            return hla_library[key]
    return None


def mhc_encoding(mhc_seq, max_seq_len):
    # max_seq_len=366 for MHC(HLA) class I
    seq = pad_protein_sequence(mhc_seq, max_seq_len)

    one_hot_feat = one_hot_encoding(seq)
    blosum50_feat = bolossum_encoding(seq, BLOSUM50_MATRIX)
    atchley_factor_feat = atchley_factor_encoding(seq)

    concat_feat = np.concatenate(
        (one_hot_feat, blosum50_feat, atchley_factor_feat), axis=1)
    return concat_feat


def epitope_batch_encoding(seq_str_list, tokenizer, batch_size, max_epitope_len=None, atchley_factor=False):

    if max_epitope_len:
        seq_str_list = [seq_str[:max_epitope_len] for seq_str in seq_str_list]

    seq_encoded_list = [tokenizer.encode(seq_str) for seq_str in seq_str_list]
    max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
    tokens = torch.empty((batch_size, max_len + int(tokenizer.prepend_bos) + int(tokenizer.append_eos)), dtype=torch.int64)
    tokens.fill_(tokenizer.padding_idx)
    
    if atchley_factor:
        atchley_factor_feat = torch.zeros((batch_size, max_len + int(tokenizer.prepend_bos) + int(tokenizer.append_eos), 5), dtype=torch.float32)

    strs = []
    for i, (seq_str, seq_encoded) in enumerate(zip(seq_str_list, seq_encoded_list)):
        strs.append(seq_str)

        if tokenizer.prepend_bos:
            tokens[i, 0] = tokenizer.cls_idx
            
        seq = torch.tensor(seq_encoded, dtype=torch.int64)
        tokens[i, int(tokenizer.prepend_bos): len(
            seq_encoded) + int(tokenizer.prepend_bos),] = seq

        if tokenizer.append_eos:
            tokens[i, len(seq_encoded) + int(tokenizer.prepend_bos)
                   ] = tokenizer.eos_idx
        
        if atchley_factor:
            atchley_factor_feat[i, int(tokenizer.prepend_bos): len(seq_encoded) + int(tokenizer.prepend_bos), :] = torch.tensor(atchley_factor_encoding(seq_str), dtype=torch.float32)

    if atchley_factor:
        return strs, tokens, atchley_factor_feat
    return strs, tokens


def paired_cdr3_batch_encoding(cdr3_alpha_list, cdr3_beta_list, tokenizer, batch_size, max_cdr3_len=None, atchley_factor=False):
    
    seq_encoded_list = []
    alpha_indices = []
    beta_indices = []
    
    for cdr3_alpha, cdr3_beta in zip(cdr3_alpha_list, cdr3_beta_list):
        if max_cdr3_len:
            cdr3_alpha, cdr3_beta = cdr3_alpha[:max_cdr3_len], cdr3_beta[:max_cdr3_len]
        seq_encoded_list.append(tokenizer.encode(cdr3_alpha) + [tokenizer.sep_idx] + tokenizer.encode(cdr3_beta))
        alpha_indices.append([0, len(cdr3_alpha)])
        beta_indices.append([len(cdr3_alpha) + 1, len(cdr3_alpha) + len(cdr3_beta) + 1])
    
    max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
    tokens = torch.empty((batch_size, max_len + int(tokenizer.prepend_bos) + int(tokenizer.append_eos)), dtype=torch.int64)
    tokens.fill_(tokenizer.padding_idx)
    chain_token_mask = torch.zeros((batch_size, max_len + int(tokenizer.prepend_bos) + int(tokenizer.append_eos)), dtype=torch.int64)
    if atchley_factor:
        atchley_factor_feat = torch.zeros((batch_size, max_len + int(tokenizer.prepend_bos) + int(tokenizer.append_eos), 5), dtype=torch.float32)
        
    for i, (seq_encoded, cdr3_alpha, cdr3_beta) in enumerate(zip(seq_encoded_list, cdr3_alpha_list, cdr3_beta_list)):
        if tokenizer.prepend_bos:
            tokens[i, 0] = tokenizer.cls_idx
        seq = torch.tensor(seq_encoded, dtype=torch.int64)
        tokens[i, int(tokenizer.prepend_bos): len(seq_encoded) + int(tokenizer.prepend_bos),] = seq
        chain_token_mask[i, alpha_indices[i][0] + int(tokenizer.prepend_bos): alpha_indices[i][1] + int(tokenizer.prepend_bos)] = 1
        chain_token_mask[i, beta_indices[i][0] + int(tokenizer.prepend_bos): beta_indices[i][1] + int(tokenizer.prepend_bos)] = 2
        
        if atchley_factor:
            if max_cdr3_len:
                cdr3_alpha, cdr3_beta = cdr3_alpha[:max_cdr3_len], cdr3_beta[:max_cdr3_len]
            atchley_factor_feat[i, alpha_indices[i][0] + int(tokenizer.prepend_bos): alpha_indices[i][1] + int(tokenizer.prepend_bos), :] = torch.tensor(atchley_factor_encoding(cdr3_alpha), dtype=torch.float32)
            atchley_factor_feat[i, beta_indices[i][0] + int(tokenizer.prepend_bos): beta_indices[i][1] + int(tokenizer.prepend_bos), :] = torch.tensor(atchley_factor_encoding(cdr3_beta), dtype=torch.float32)

        if tokenizer.append_eos:
            tokens[i, len(seq_encoded) + int(tokenizer.append_eos)] = tokenizer.eos_idx
            
    if atchley_factor:
        return tokens, chain_token_mask, atchley_factor_feat
    return tokens, chain_token_mask


def paired_cdr123_batch_encoding(batch, tokenizer, batch_size, max_cdr3_len=None, atchley_factor=False):
    if isinstance(batch, dict):
        batch = [{key: values[i] for key, values in batch.items()} for i in range(len(batch['cdr3_beta']))]
        
    alpha_encoded_list, beta_encoded_list = [], []
    cdr1a_indices, cdr2a_indices, cdr3a_indices = [], [], []
    cdr1b_indices, cdr2b_indices, cdr3b_indices = [], [], []
       
    for data in batch: 
        if max_cdr3_len is not None:
            cdr3_alpha, cdr3_beta = data['cdr3_alpha'][:max_cdr3_len], data['cdr3_beta'][:max_cdr3_len]
        else:
            cdr3_alpha, cdr3_beta = data['cdr3_alpha'], data['cdr3_beta']
        cdr1_alpha, cdr2_alpha, cdr1_beta, cdr2_beta = data['cdr1_alpha'], data['cdr2_alpha'], data['cdr1_beta'], data['cdr2_beta']
        
        alpha_encoded_list.append(tokenizer.encode(cdr1_alpha) + tokenizer.encode(cdr2_alpha) + tokenizer.encode(cdr3_alpha))
        beta_encoded_list.append(tokenizer.encode(cdr1_beta) + tokenizer.encode(cdr2_beta) + tokenizer.encode(cdr3_beta))
        
        cdr1a_indices.append([0, len(cdr1_alpha)])
        cdr2a_indices.append([len(cdr1_alpha), len(cdr1_alpha + cdr2_alpha)])
        cdr3a_indices.append([len(cdr1_alpha + cdr2_alpha), len(cdr1_alpha + cdr2_alpha + cdr3_alpha)])

        cdr1b_indices.append([0, len(cdr1_beta)])
        cdr2b_indices.append([len(cdr1_beta), len(cdr1_beta + cdr2_beta)])
        cdr3b_indices.append([len(cdr1_beta + cdr2_beta), len(cdr1_beta + cdr2_beta + cdr3_beta)])

    max_alpha_len = max(len(seq_encoded) for seq_encoded in alpha_encoded_list)
    alpha_tokens = torch.empty((batch_size, max_alpha_len + int(tokenizer.prepend_bos)), dtype=torch.int64)
    alpha_segment_mask = torch.zeros((batch_size, max_alpha_len + int(tokenizer.prepend_bos)), dtype=torch.int64)
    alpha_tokens.fill_(tokenizer.padding_idx)
    
    max_beta_len = max(len(seq_encoded) for seq_encoded in beta_encoded_list)
    beta_tokens = torch.empty((batch_size, max_beta_len + int(tokenizer.append_eos)), dtype=torch.int64)
    beta_segment_mask = torch.zeros((batch_size, max_beta_len + int(tokenizer.append_eos)), dtype=torch.int64)
    beta_tokens.fill_(tokenizer.padding_idx)
 
    for i, seq_encoded in enumerate(alpha_encoded_list):
        if tokenizer.prepend_bos:
            alpha_tokens[i, 0] = tokenizer.cls_idx
        seq = torch.tensor(seq_encoded, dtype=torch.int64)
        alpha_tokens[i, int(tokenizer.prepend_bos): len(seq_encoded) + int(tokenizer.prepend_bos),] = seq
        alpha_segment_mask[i, cdr1a_indices[i][0] + int(tokenizer.prepend_bos): cdr1a_indices[i][1] + int(tokenizer.prepend_bos)] = 1
        alpha_segment_mask[i, cdr2a_indices[i][0] + int(tokenizer.prepend_bos): cdr2a_indices[i][1] + int(tokenizer.prepend_bos)] = 2
        alpha_segment_mask[i, cdr3a_indices[i][0] + int(tokenizer.prepend_bos): cdr3a_indices[i][1] + int(tokenizer.prepend_bos)] = 3
    
    for i, seq_encoded in enumerate(beta_encoded_list):
        seq = torch.tensor(seq_encoded, dtype=torch.int64)
        beta_tokens[i, :len(seq_encoded),] = seq
        beta_segment_mask[i, cdr1b_indices[i][0]: cdr1b_indices[i][1]] = 4
        beta_segment_mask[i, cdr2b_indices[i][0]: cdr2b_indices[i][1]] = 5
        beta_segment_mask[i, cdr3b_indices[i][0]: cdr3b_indices[i][1]] = 6
        if tokenizer.append_eos:
            tokens[i, len(seq_encoded) + int(tokenizer.append_eos)] = tokenizer.eos_idx
    
    tokens = torch.cat((alpha_tokens, beta_tokens), dim=1)
    segment_mask = torch.cat((alpha_segment_mask, beta_segment_mask), dim=1)
    
    if atchley_factor:
        alpha_atchley_factor_feat = torch.zeros((batch_size, max_alpha_len + int(tokenizer.prepend_bos), 5), dtype=torch.float32)
        beta_atchley_factor_feat = torch.zeros((batch_size, max_beta_len + int(tokenizer.append_eos), 5), dtype=torch.float32)
        for i, data in enumerate(batch):
            if max_cdr3_len is not None:
                cdr3_alpha, cdr3_beta = data['cdr3_alpha'][:max_cdr3_len], data['cdr3_beta'][:max_cdr3_len]
            else:
                cdr3_alpha, cdr3_beta = data['cdr3_alpha'], data['cdr3_beta']
            cdr1_alpha, cdr2_alpha, cdr1_beta, cdr2_beta = data['cdr1_alpha'], data['cdr2_alpha'], data['cdr1_beta'], data['cdr2_beta']
            alpha_atchley_factor_feat[i, int(tokenizer.prepend_bos): int(tokenizer.prepend_bos) + len(cdr1_alpha + cdr2_alpha + cdr3_alpha), :] = torch.tensor(
                atchley_factor_encoding(cdr1_alpha + cdr2_alpha + cdr3_alpha), dtype=torch.float32)
            beta_atchley_factor_feat[i, :len(cdr1_beta + cdr2_beta + cdr3_beta), :] = torch.tensor(
                atchley_factor_encoding(cdr1_beta + cdr2_beta + cdr3_beta), dtype=torch.float32)
        atchley_factor_feat = torch.cat((alpha_atchley_factor_feat, beta_atchley_factor_feat), dim=1)
        return tokens, segment_mask, atchley_factor_feat

    return tokens, segment_mask


def encoding_dist(mat_list, max_cdr3_len, max_epi_len):
    '''
        mat_list: distance matrices in a batch,
        max_cdr3_len: 1st dim of the encoded matrices (=tcr_tokens.shape[1]-1),
        max_epi_len: 2nd dim of the encoded matrices (=epitope_tokens.shape[1]-1),
    '''
    dist_encoding = np.zeros([len(mat_list), max_cdr3_len, max_epi_len], dtype='float32')
    contact_encoding = np.zeros([len(mat_list), max_cdr3_len, max_epi_len], dtype='float32')
    masking = np.zeros([len(mat_list), max_cdr3_len, max_epi_len], dtype='bool')
    
    for i, mat in enumerate(mat_list):
        dist_mat = mat['dist']
        contact_mat = mat['contact']
        cdr3_len, epi_len = dist_mat.shape
        dist_encoding[i, :cdr3_len, :epi_len] = dist_mat
        contact_encoding[i, :cdr3_len, :epi_len] = contact_mat
        masking[i, :cdr3_len, :epi_len] = True # mask=1 when the loss at this postion need to be computed.
    
    return dist_encoding, contact_encoding, masking
    

def encoding_paired_dist(alpha_mat_list, beta_mat_list, max_cdr3_len, max_epi_len):
    '''
        mat_list: distance matrices in a batch,
        max_cdr3_len: 1st dim of the encoded matrices (=tcr_tokens.shape[1]-1),
        max_epi_len: 2nd dim of the encoded matrices (=epitope_tokens.shape[1]-1),
    '''
    assert len(alpha_mat_list) == len(beta_mat_list)
    batch_size = len(alpha_mat_list)
    dist_encoding = np.zeros([batch_size, max_cdr3_len, max_epi_len], dtype='float32')
    contact_encoding = np.zeros([batch_size, max_cdr3_len, max_epi_len], dtype='float32')
    alpha_masking = np.zeros([batch_size, max_cdr3_len, max_epi_len], dtype='bool')
    beta_masking = np.zeros([batch_size, max_cdr3_len, max_epi_len], dtype='bool')
    
    for i, (mat_a, mat_b) in enumerate(zip(alpha_mat_list, beta_mat_list)):
        dist_mat_alpha, dist_mat_beta = mat_a['dist'], mat_b['dist']
        contact_mat_alpha, contact_mat_beta = mat_a['contact'], mat_b['contact']
        
        cdr3_alpha_len, epi_len = dist_mat_alpha.shape
        dist_encoding[i, :cdr3_alpha_len, :epi_len] = dist_mat_alpha
        contact_encoding[i, :cdr3_alpha_len, :epi_len] = contact_mat_alpha
        alpha_masking[i, :cdr3_alpha_len, :epi_len] = True
        
        cdr3_beta_len, epi_len = dist_mat_beta.shape
        dist_encoding[i, cdr3_alpha_len+1:cdr3_alpha_len+cdr3_beta_len+1, :epi_len] = dist_mat_beta # skip the <sep> token in cdr3 sequences
        contact_encoding[i, cdr3_alpha_len+1:cdr3_alpha_len+cdr3_beta_len+1, :epi_len] = contact_mat_beta
        beta_masking[i, cdr3_alpha_len+1:cdr3_alpha_len+cdr3_beta_len+1, :epi_len] = True
    
    return dist_encoding, contact_encoding, alpha_masking, beta_masking
