import collections
import pandas as pd
from pathlib import Path
from itertools import chain, combinations
from timeit import default_timer as timer
import csv
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
import nimfa
import scipy.spatial.distance as dist
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy.ma as ma
import re
import numpy as np

def convert_nucleotide_to_AA(three_nucleotides):
    if 'T' in three_nucleotides:
        three_nucleotides = three_nucleotides.replace('T', 'U')
    codon_table = { # Phenylalanine (Phe)
                    'UUU' : 'F', 'UUC' : 'F',
                    # Leucine (Leu)
                    'UUA' : 'L', 'UUG' : 'L', 'CUU' : 'L', 
                    'CUC' : 'L', 'CUA' : 'L', 'CUG' : 'L',
                    # Isoleucine (Ile)
                    'AUU' : 'I', 'AUC' : 'I', 'AUA' : 'I',
                    # Methionine (Met)
                    'AUG' : 'M',
                    # Valine (Val)
                    'GUU' : 'V', 'GUC' : 'V', 
                    'GUA' : 'V', 'GUG' : 'V',
                    # Serine (Ser)
                    'UCU' : 'S', 'UCC' : 'S', 'UCA' : 'S',
                    'UCG' : 'S', 'AGU' : 'S', 'AGC' : 'S',
                    # Proline (Pro)
                    'CCU' : 'P', 'CCC' : 'P', 
                    'CCA' : 'P', 'CCG' : 'P',
                    # Threonine (Thr)
                    'ACU' : 'T', 'ACC' : 'T', 
                    'ACA' : 'T', 'ACG' : 'T',
                    # Alanine (Ala)
                    'GCU' : 'A', 'GCC' : 'A', 
                    'GCA' : 'A', 'GCG' : 'A',
                    # Tyrosine (Tyr)
                    'UAU' : 'Y', 'UAC' : 'Y',
                    # Termination (ochre) (Ter)
                    'UAA' : 'X',
                    # Termination (amber) (Ter)
                    'UAG' : 'X',
                    # Histidine (His)
                    'CAU' : 'H', 'CAC' : 'H',
                    # Glutamine (Gln)
                    'CAA' : 'Q', 'CAG' : 'Q',
                    # Asparagine (Asn)
                    'AAU' : 'N', 'AAC' : 'N',
                    # Lysine (Lys)
                    'AAA' : 'K', 'AAG' : 'K',
                    # Aspartate (Asp)
                    'GAU' : 'D', 'GAC' : 'D',
                    # Glutamate (Glu)
                    'GAA' : 'E', 'GAG' : 'E',
                    # Cysteine (Cys)
                    'UGU' : 'C', 'UGC' : 'C',
                    # Termination (opal or umber) (Ter)
                    'UGA' : 'X',
                    # Tryptophan (Trp)
                    'UGG' : 'W',
                    # Arginine (Arg)
                    'CGU' : 'R', 'CGC' : 'R', 'CGA' : 'R',
                    'CGG' : 'R', 'AGA' : 'R', 'AGG' : 'R',
                    # Glycine (Gly)
                    'GGU' : 'G', 'GGC' : 'G',
                    'GGA' : 'G', 'GGG' : 'G'}
    return codon_table.get(three_nucleotides, 'X')

def convert_entire_seq_to_AA(seq):
    extra_nucleotides = len(seq) % 3
    if not extra_nucleotides == 0:
        seq = seq[0:len(seq)-extra_nucleotides]
    seq_chunks = [seq[i:i+3] for i in range(0, len(seq), 3)]
    AA_seq = []
    for sequence in seq_chunks:
        AA_seq.append(convert_nucleotide_to_AA(sequence))
    return ''.join(AA_seq)

def convert_entire_seq_list_to_AA(seq_list):
    AA_list = []
    for seq in seq_list:
        AA_list.append(convert_entire_seq_to_AA(seq))
    return AA_list

def reorder_sequences(index_list, seq_list):
    reordered_list = []
    for index in index_list:
        reordered_list.append(seq_list[index])
    return reordered_list
        
def filter_W_non_mutations(W, seqs):
    if len(W) != len(seqs):
        print("ERROR: row count in W and sequence labels do not match")
        return None
    W = W.tolist()
    labeled_pairs = [(W[i], seqs[i]) for i in range(0, len(seqs))]
    res = []
    for (w, seq) in labeled_pairs:
        if sum(w) > 0:
            # mutation occured
            res.append((w, seq))       
    W, seqs = (np.array([ w for w,s in res ]), [ s for w,s in res ])
    return(convert_V(W), seqs)

def hamming_distance(seq1, seq2):
    if len(seq1) != len(seq2):
        print("ERROR: hamming distance cannot be computed on sequences of different lengths!")
        return None
    count = 0
    for i in range(len(seq1)):
        if seq1[i] != seq2[i]:
            count += 1
    return count

def get_consensus(sequences):
    if len(sequences) < 1:
        print("ERROR: sequence list is too short!")
        return None
    seq_length = len(sequences[0])
    consensus_string = []
    for index in range(seq_length):
        votes = {'A' : 0, 'T' : 0, 'G' : 0, 'C' : 0}
        for seq in sequences:
            if seq[index] == 'A':
                votes['A'] += 1
            elif seq[index] == 'T':
                votes['T'] += 1
            elif seq[index] == 'G':
                votes['G'] += 1 
            elif seq[index] == 'C':
                votes['C'] += 1
        winner = sorted(votes.items(), key=lambda x: -x[1])[0][0] # most frequent key is in the first index
        consensus_string.append(winner) 
    return ''.join(consensus_string)

def get_binary_seq_diff(seq, consensus):
    if len(seq) != len(consensus):
        print("ERROR: sequence is a different length than consensus sequence!")
        return None
    seq_length = len(seq)
    diff = []
    for index in range(seq_length):
        if seq[index] == consensus[index]:
            diff.append(0.0) # match
        else:
            diff.append(1.0) # non-match
    if len(diff) != len(consensus):
        print("ERROR: sequence difference is a different length than consensus sequence!")
        return None
    return diff

def get_cost_seq_diff(seq, consensus):
    if len(seq) != len(consensus):
        print("ERROR: sequence is a different length than consensus sequence!")
        return None
    extra_nucleotides = len(seq) % 3
    if not extra_nucleotides == 0:
        seq = seq[0:len(seq)-extra_nucleotides]
        consensus = consensus[0:len(consensus)-extra_nucleotides]
    seq_chunks = [seq[i:i+3] for i in range(0, len(seq), 3)]
    target_chunks = [consensus[i:i+3] for i in range(0, len(consensus), 3)]
    diff = []
    for i in range(len(seq_chunks)):
        seq_AA = convert_nucleotide_to_AA(seq_chunks[i])
        target_AA = convert_nucleotide_to_AA(target_chunks[i])
        if seq_AA != target_AA and seq_AA != 'X' and target_AA != 'X':
            val = hamming_distance(seq_chunks[i], target_chunks[i]) / 3 # > non-match
            if val > 3:
                print("ERROR: hamming distance between two length 3 sequences cannot be greater than 3. \
                       Recommend checking the sizes of the sequences being compared.")
                return None
        else:
            val = 0 # match
        diff.append(val)
    if len(diff) != len(consensus)//3:
        print("ERROR: sequence difference is a different length than 1/3 consensus sequence!")
        return None
    return diff

def construct_V(sequences, consensus):
# constructs V
# Each row is a sequence and each column is an index. 
# Each cell contains 1/0 denoting a value match/mismatch of that sequence relative to consensus at that index
    V = []
    for seq in sequences:
        V.append(get_binary_seq_diff(seq, consensus))
    if len(V) != len(sequences):
        print("ERROR: V's row count does not match number of sequences!")
        return None
    if len(V[0]) != len(consensus):
        print("ERROR: V's column count does not match consensus nucleotide sequence length!")
        return None
    return V

def construct_cost_V(sequences, consensus):
    V = []
    for seq in sequences:
        V.append(get_cost_seq_diff(seq, consensus))
    if len(V) != len(sequences):
        print("ERROR: V's row count does not match number of sequences!")
        return None
    if len(V[0]) != len(consensus)//3:
        print("ERROR: V's column count does not match consensus amino acid sequence length!")
        return None
    return V
        
def print_V(V):
    for row in V:
        lis = []
        for elem in row:
            lis.append(str(elem))
        print(''.join(lis))
        
def convert_V(V):
    return np.asmatrix(V, dtype=np.float32)

def get_power_set(seq):
    ret_set = set()
    comutation_list = seq.split(',')
    powerset = chain.from_iterable(combinations(comutation_list, r) for r in range(len(comutation_list)+1))
    for pseq in powerset:
        string_seq = ','.join(pseq)
        if string_seq:
            ret_set.add(string_seq)    
    return sorted(list(ret_set))

def get_comutation_pos(comutation, offset):
    mutated_AA = comutation.split(',')
    pos_list = []
    for elem in mutated_AA:
        pos = int(elem[1:])-offset
        pos_list.append(pos)
    return pos_list

def is_match(pos_list, bit_string):
    for pos in pos_list:
        if bit_string[pos] != '1':
            return False
    return True

def count_mutations_across_data_set(binary_data_seqs, comutation, offset):
    count = 0
    for key, freq in binary_data_seqs.items():
        pos_list = get_comutation_pos(comutation, offset)
        if is_match(pos_list, key):
            count += freq
    return count
    
def get_bit_diff(seq, ref_seq):
    # 1 at indexes of mutation, 0 otherwise
    bit_array = ""
    for i in range(len(ref_seq)):
        if seq[i] != ref_seq[i]:
            bit_array += '1'
        else:
            bit_array += '0'
    return bit_array

def get_bit_diff_dataset(ref_seq, data_set):
    # create dictionary: key => diff bitarray, value => frequency accross dataset
    bit_dict = dict()
    for seq in data_set:
        bits_diff = get_bit_diff(seq, ref_seq)
        if bits_diff  in bit_dict.keys():
            bit_dict[bits_diff] += 1
        else:
            bit_dict[bits_diff] = 1
    return bit_dict

def compute_consensus_matrix(lis):
    consensus = np.zeros(lis[0].shape, dtype=np.float32)
    for matrix in lis:
        consensus = consensus + matrix
    consensus = consensus / len(lis)
    if consensus.shape != lis[0].shape:
        print("ERROR: the mean matrix has an invalid shape!")
    return consensus

################## PARSE DATA ############################
path = str(Path.home() / f"Desktop/PLOS") # Change path for saved files here
# define ancestor sequences
covid_mother_seq_DNA = "AGAGTCCAACCAACAGAATCTATTGTTAGATTTCCTAATATTACAAACTTGTGCCCTTTTGGTGAAGTTTTTAACGCCACCAGATTTGCATCTGTTTATGCTTGGAACAGGAAGAGAATCAGCAACTGTGTTGCTGATTATTCTGTCCTATATAATTCCGCATCATTTTCCACTTTTAAGTGTTATGGAGTGTCTCCTACTAAATTAAATGATCTCTGCTTTACTAATGTCTATGCAGATTCATTTGTAATTAGAGGTGATGAAGTCAGACAAATCGCTCCAGGGCAAACTGGAAAGATTGCTGATTATAATTATAAATTACCAGATGATTTTACAGGCTGCGTTATAGCTTGGAATTCTAACAATCTTGATTCTAAGGTTGGTGGTAATTATAATTACCTGTATAGATTGTTTAGGAAGTCTAATCTCAAACCTTTTGAGAGAGATATTTCAACTGAAATCTATCAGGCCGGTAGCACACCTTGTAATGGTGTTGAAGGTTTTAATTGTTACTTTCCTTTACAATCATATGGTTTCCAACCCACTAATGGTGTTGGTTACCAACCATACAGAGTAGTAGTACTTTCTTTTGAACTTCTACATGCACCAGCAACTGTTTGTGGACCTAAAAAGTCTACTAATTTGGTTAAAAACAAATGTGTCAATTTC"
covid_mother_seq_AA = "RVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF"
# parse data from csv files
covid_dna_seqs = []
covid_aa_seqs = []
pango = []
year = []

for i in range(1,6):
    data = pd.read_csv(f"spike_protein_w_pango_part_{i}.csv")
    for dna_seq, aa_seq, pango_elem, year_elem in zip(data['rbd_dna'], data['rbd_AA'], data['pango'], data['year']):
        covid_dna_seqs.append(dna_seq)
        covid_aa_seqs.append(aa_seq)
        pango.append(pango_elem)
        year.append(year_elem)   


# Create H matrices
def save_matrix_graphic(aa, r, path, filename, offset, threshold):
    df_h = pd.read_csv(f"{path}/{filename}", header = None)
    adj_inx = []
    for a in aa:
        adj_inx.append(str(a)+str(offset))
        offset += 1

    df_htemp_filt = df_h.loc[:, (df_h > 0.005).any(axis=0)]
    df_htemp = df_h.loc[:, (df_h != 0).any(axis=0)]

    df_htemp_filt_copy = df_h.copy()
    df_htemp_filt_copy = df_htemp_filt_copy.set_axis(adj_inx, axis='columns')
    df_htemp_filt_copy = df_htemp_filt_copy.loc[:, (df_htemp_filt_copy > .005).any(axis=0)]

    sns.set(font_scale=1.0)

    mask = df_htemp_filt_copy.T >= threshold
    sns.set_style("dark")
    c_map = sns.clustermap(df_htemp_filt_copy.T, figsize = (8,10),  linewidth=.5, annot=True, annot_kws={"size":8},fmt=".2f" , cmap = 'bone', mask=~mask)

    for text in c_map.ax_heatmap.texts:
        if text.get_text() == '':
            text.set_color('black',weight='bold')

    plt.ylabel('Amino Acid Position')
    filt_labels = c_map.ax_heatmap.yaxis.get_majorticklabels()
    rows = c_map.dendrogram_row.reordered_ind
    cols = c_map.dendrogram_col.reordered_ind
    c_map.ax_row_dendrogram.set_visible(False)
    c_map.ax_col_dendrogram.set_visible(False)
    c_map.ax_cbar.set_position((.1, .2, .02, .6))
    plt.grid(False)

    plt.savefig(f"{path}/Factor_cluster_{r}.png", format='png')

threshold = .1
offset = 319 # offset is 319
V = convert_V(construct_cost_V(covid_dna_seqs, covid_mother_seq_DNA))

for r in range(7,18):
    lsnmf = nimfa.Lsnmf(V, seed="random_vcol", rank=r, max_iter=100, beta=0.1, n_run=1)
    lsnmf_fit = lsnmf()

    H = lsnmf_fit.coef() # matrix of Factors x Positions 
    H_df = pd.DataFrame(H)
    
    H_df.to_csv(f"{path}/H_Matrices/H_rank_{r}.csv", header=False, index=False)
    save_matrix_graphic(covid_mother_seq_AA, r, f"{path}/H_Matrices", f"H_rank_{r}.csv", offset, threshold)

### Extract Union of Mutations from H matrices
amino_acid_lis = list(covid_mother_seq_AA)
start = 7
end = 17
union_mutations = list()

def build_mutation_str(pos_aa):
    mutation = ""
    for pos, aa in pos_aa:
        mutation += f",{aa}{pos}"
    return mutation[1:] # remove the leading comma
    
for r in range(start,end+1):
    H_df = pd.read_csv(f"{path}/H_Matrices/H_rank_{r}.csv")
    for index, row in H_df.iterrows():
        pos_aa_tuple_set = set()
        for i in range(0, len(row)):
            if row[i] >= threshold:
                pos_aa_tuple = (i+offset, amino_acid_lis[i])
                pos_aa_tuple_set.add(pos_aa_tuple)

        if pos_aa_tuple_set:
            sorted_tuple_lis = sorted(list(pos_aa_tuple_set), key=lambda x: x[0]) # sort by position
            union_mutations.append(sorted_tuple_lis)
        
str_mutations = map(build_mutation_str, union_mutations)
str_mutations = sorted(list(set(str_mutations)), key=len)

# Calculate frequency
zero_lis = [0] * len(str_mutations)
mutation_dict = dict(zip(str_mutations, zero_lis))

condensed_dataset_dict = get_bit_diff_dataset(covid_mother_seq_AA, covid_aa_seqs)
for key in mutation_dict.keys():
    freq = count_mutations_across_data_set(condensed_dataset_dict, key, offset)
    mutation_dict[key] = freq

save_df = pd.DataFrame({"Mutations" : mutation_dict.keys(), "Frequency" : mutation_dict.values()})
save_df.to_csv(f"{path}/Union_Extracted_Mutations.csv")

import pandas as pd
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
from pathlib import Path

def is_subset(str1, str2):
    set1 = set(str1.split(','))
    set2 = set(str2.split(','))
    return set1.issubset(set2)

# change so that asymmetric relationship between subsets and supersets
# any SINGLE failure to meet threshold, then drop subset mutation

path = str(Path.home() / f"Desktop/PLOS")
df = pd.read_csv(f"{path}/Union_Extracted_Mutations.csv")

mut_freq_tuples = [(mut, freq) for mut, freq in zip(list(df["Mutations"]), list(df["Frequency"]))]
final_list = mut_freq_tuples.copy()

for i in range(0, len(mut_freq_tuples)):
    for j in range(0, len(mut_freq_tuples)):
        if i != j:
            i_elem = mut_freq_tuples[i]
            j_elem = mut_freq_tuples[j]
            i_mut_str = i_elem[0]
            j_mut_str = j_elem[0]
            if is_subset(i_mut_str, j_mut_str):
                subset_freq = i_elem[1]
                superset_freq = j_elem[1]
                freq_delta = abs(subset_freq - superset_freq)
                
                larger_number = 0
                if subset_freq > superset_freq:
                    larger_number = subset_freq
                else:
                    larger_number = superset_freq
                
                result = freq_delta / larger_number
                
                if result < threshold:
                    if i_elem in final_list:
                        # keep only superset by dropping subset
                        final_list.remove(i_elem)
                # else
                    # keep both
                    # list is initialized with all elements, so updating list is not neccessary
                        
final_list.sort(key=lambda t: len(t[0]), reverse=False)
final_mut_list, final_freq_lis = zip(*final_list)
df_temp = pd.DataFrame()
df_temp['Mutations'] = final_mut_list
df_temp['Frequency'] = final_freq_lis
df_temp.to_csv(f"{path}/Extracted_Mutations_Without_Any_Statistically_Irrelvent_Subsets.csv")

### Caculate DSIM for r = [1,50]
def factor_sim(f):
    import numpy as np
    fxft = np.dot(f, f.transpose())
    return (float(np.linalg.det(fxft)))

r_lis = []
H_SIM_lis = []
CCC_lis = []

for r in range(1,51):
    r_lis.append(r)
    start = timer()
    lsnmf = nimfa.Lsnmf(V, seed="random_vcol", rank=r, max_iter=100, beta=0.1, n_run=1)
    lsnmf_fit = lsnmf()

    # matrix of Factors x Positions
    H = lsnmf_fit.coef()
    H_sim = factor_sim(H)
    H_SIM_lis.append(H_sim)

    # compute CCC for rank
    C_prime = lsnmf_fit.fit.connectivity()
    I = np.identity(C_prime.shape[0])
    HC_linkage = dist.squareform(dist.pdist(C_prime))
    corr = np.corrcoef((I - C_prime).flat, HC_linkage.flat)[0][1]
    CCC_lis.append(corr)

stats_df = pd.DataFrame({"Rank" : r_lis, "H Similarity" : H_SIM_lis, "CCC" : CCC_lis})
stats_df.to_csv(f"{path}/H_Matrices/H_rank_stats.csv", index=False)

######### BRUTE FORCE FREQUENCY ANALYSIS ###########################
def write_exhaustive_frequency_csvs(path, origin_seq, bitarray_data):
    powerset = get_power_set(origin_seq)
    metric_dict = {"Mutation" : [], "Frequency" : []}
    data_frame_size = 0
    file_count = 1
    for current_mutation in powerset:
        freq = count_mutations_across_data_set(bitarray_data, current_mutation, offset)
        # If none of the data sequences contain the current mutation, then it is dropped to reduce space
        if freq > 0:
            metric_dict["Mutation"].append(current_mutation)
            metric_dict["Frequency"].append(freq)
            data_frame_size += 1
            
        if data_frame_size > 300000:
            # Caps files at 300,000 rows to prevent them from growing too large 
            filename = f"Mutation_Frequency_Batch_{file_count}"
            matrix_data = pd.DataFrame(metric_dict)
            matrix_data.to_csv(f"{path}/{filename}.csv", index=False)
            file_count += 1
            data_frame_size = 0
            metric_dict = {"Mutation" : [], "Frequency" : []}
            
    # Write any remaining values        
    filename = f"Mutation_Frequency_Batch_{file_count}"
    matrix_data = pd.DataFrame(metric_dict)
    matrix_data.to_csv(f"{path}/{filename}.csv", index=False)

save_path = f"{path}/FREQ_CSVS"

# these are all 21 mutation sites
covid_mutation_sites =  f"G{339-offset},R{346-offset},S{371-offset},S{373-offset},S{375-offset},T{376-offset},D{405-offset},R{408-offset},K{417-offset},N{440-offset},G{446-offset},L{452-offset},S{477-offset},T{478-offset},E{484-offset},F{486-offset},Q{493-offset},G{496-offset},Q{498-offset},N{501-offset},Y{505-offset}"

bitarray_data = get_bit_diff_dataset(covid_mother_seq_AA, covid_aa_seqs)
write_exhaustive_frequency_csvs(save_path, covid_mutation_sites, bitarray_data)

