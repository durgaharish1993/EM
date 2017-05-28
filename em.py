from __future__ import division
import math
from collections import defaultdict
import sys


def align(rem_short, rem_long, limit):
    if rem_short == 1:
        return [rem_long]

    min_map_cnt = max([0, int(math.ceil(rem_long/(rem_short-1)) - limit)])
    max_map_cnt = min([limit, rem_long])

    out = []
    for m in xrange(min_map_cnt, max_map_cnt+1):
        a = align(rem_short-1, rem_long-m, limit)
        if isinstance(a[0], list):
            out += [[m] + x for x in a]
        else:
            out += [[m] + [x] for x in a]
    return out



def gen_legal_alignments(seq1, seq2, max_sub_size = 3):
    if (len(seq2) - len(seq1)) / len(seq1) > max_sub_size-1:
        return None    
    ex =  align(len(seq1), len(seq2) - len(seq1), max_sub_size-1)            
    aligned_indices = [[x+1 for x in (ex[i])] for i in xrange(len(ex))]        
    mappings = defaultdict(lambda: defaultdict(list))
    all_alignments = []
    for e in aligned_indices:
        idx = 0
        alignment = []
        for i in xrange(len(seq1)):
            sub_seq2 = ' '.join(seq2[idx:idx+e[i]])
            mappings[seq1[i]][sub_seq2]+=[len(all_alignments)]        
            alignment += [sub_seq2]
            idx += e[i]
        all_alignments += [alignment]
    
    return all_alignments, mappings
        
def em(english, katakana, max_iter = 100):
    # if len(english) != len(katakana):
    #     return -1
    
    #initialization
    alignments, mappings = gen_legal_alignments(english, katakana)
    prob_table = defaultdict(lambda: defaultdict(float))
    for ep in mappings:
        for jp in mappings[ep]:
            prob_table[ep][jp] = 1/len(mappings[ep])    
    
    for iteration in xrange(max_iter):
        #E-step
        p_x_z = defaultdict(float)
        p_x = 0.0
        for al_idx,alg in enumerate(alignments):
            p_x_z[al_idx] = 1.0        
            for ix, jp in enumerate(alg):                                
                p_x_z[al_idx] *= prob_table[english[ix]][jp]
            p_x += p_x_z[al_idx]
        
        #renormalized to get fraccount
        for al_idx,alg in enumerate(alignments):                        
            p_x_z[al_idx] = p_x_z[al_idx]/p_x

        #M-step
        for ep in prob_table:
            for jp in prob_table[ep]:
                new_prob = 0.0
                for al_idx in mappings[ep][jp]:
                    new_prob += p_x_z[al_idx]
                prob_table[ep][jp] = new_prob

        print 'iteration {0} ----- corpus prob = {1}'.format(iteration,p_x)
        non_zeros = 0
        for ep in prob_table:
            m = ep+'|->\t'
            for jp in prob_table[ep]:
                if prob_table[ep][jp] > 0.000001:
                    non_zeros += 1
                    m += '{0}: {1:.2f}\t'.format(jp, prob_table[ep][jp])
            print m
        print 'nonzeros = {0}'.format(non_zeros)
        if non_zeros == len(prob_table):
            break


if __name__ == '__main__':
    arg = sys.argv
    print arg
    if len(arg) > 1:
        max_iter = int(arg[1])



    em(['W', 'AY', 'N'],['W', 'A', 'I', 'N'], max_iter)
