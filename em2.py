from __future__ import print_function
from collections import defaultdict
import numpy as np
from math import log,exp
from decimal import Decimal
import sys

'''
Expectation
compute forward and backward probabilities
then return fractional counts of each alignment

wordPairs:  (string, string) of English, Japanese
prior:      dict mapping e->j->p where p is current prob. est.
maxE2J:     maximum k in k-to-1 mapping
'''
def Expectation(wordPairs, prior, maxE2J):
    fracCount = {}
    for i in range(len(wordPairs)):
        eword = wordPairs[i][0]
        jword = wordPairs[i][1]
        e, j = eword.split(), jword.split()
        # compute forward and backward prob for each word
        alpha = Forward(e, j, prior, maxE2J)
        beta = Backward(e, j, prior, maxE2J)

        # sum the fractional counts
        fracCount[i] = FindFracCounts(e, j, alpha, beta, prior, maxE2J)

    return fracCount


'''
FindFracCounts
eprons:     list of English sounds
jprons:     list of Japanese sounds
alpha:      array of forward probabilities by [epron][jpron]
beta:       array of backward probabilities
prior:      dict mapping e->j->p where p is current prob. est.
maxE2J:     maximum k in k-to-1 mapping
'''


def FindFracCounts(eprons, jprons, alpha, beta, prior, maxE2J):
    numJ, numE = len(jprons), len(eprons)
    # counts = defaultdict(lambda : defaultdict(float))



    eplison = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    for i in range(1,numE+1):
        prev_alpha = np.array(alpha[i])
        cur_alpha = np.array(alpha[i - 1])

        cur_indexes = list(np.where(prev_alpha > 0)[0])
        pre_indexes = list(np.where(cur_alpha > 0)[0])
        ### I will be doing more work but its fine.  I will be visiting more edges which zero probability.
        for i_1 in pre_indexes:
            for i_2 in cur_indexes:
                ep = eprons[i-1]
                jp = tuple(jprons[i_1:i_2])
                prob = (alpha[i-1][i_1] * prior[ep][jp] * beta[i][i_2]) /alpha[numE][numJ]
                if prob > 0:
                    eplison[i][ep][jp] = prob


    return eplison






    #
    #
    # for a in EnumAligns(eprons, jprons, []):
    #     e, j = 0, 0
    #     p_x_z = 1.
    #     for (ep, js) in a:
    #         e += 1
    #         j += len(js)
    #         #p(x | z) : whole word alignment prob
    #         p_x_z *= alpha[e][j] * beta[e][j] * prior[ep][js]
    #
    #         counts[ep][js] += 1
    #
    # # now we need to do something with p_x_z
    #
    #
    # # normalize...?
    # for e, d in counts.items():
    #     for j in d.keys():
    #         counts[e][j] /= beta[1][1]





def Maximization(wordPairs, fracCount, prior):
    updated_prior = defaultdict(lambda :defaultdict(float))
    sum_prob = defaultdict(float)

    for i in range(len(wordPairs)):
        for t in fracCount[i]:
            for key in fracCount[i][t]:
                for key1 in fracCount[i][t][key]:
                    updated_prior[key][key1] += fracCount[i][t][key][key1]
                    sum_prob[key]+=fracCount[i][t][key][key1]

    for key in updated_prior:
        for key1 in updated_prior[key]:
            updated_prior[key][key1] =  updated_prior[key][key1]/sum_prob[key]


    return updated_prior







'''
forward
eprons:     list of English sounds
jprons:     list of Japanese sounds
prior:      dict mapping e->j->p where p is current prob. est.
maxE2J:     maximum k in k-to-1 mapping
'''
def Forward(eprons, jprons, prior, maxE2J):
    # alpha: prob. of getting to this state by some path from start
    numJ, numE = len(jprons), len(eprons)
    alpha = [[0. for i in range(numJ + 1)] for j in range(numE + 1)]
    alpha[0][0] = 1.
    # find alpha for each alignment
    for i in range(numE):
        for j in range(numJ):
            for k in range(1, min(numJ - j, maxE2J) + 1):
                ep, js = eprons[i], tuple(jprons[j : j + k])
                alpha[i + 1][j + k] += alpha[i][j] * prior[ep][js]

    return alpha
    #return [row[1 :] for row in alpha[1 :]]



'''
backward
eprons:     list of English sounds
jprons:     list of Japanese sounds
prior:      dict mapping e->j->p where p is current prob. est.
maxE2J:     maximum k in k-to-1 mappings
'''
def Backward(eprons, jprons, prior, maxE2J):
    # beta: prob. of getting to final state from this state
    numJ, numE = len(jprons), len(eprons)
    beta = [[0. for i in range(numJ + 1)] for j in range(numE + 1)]
    beta[numE][numJ] = 1.
    for i in range(numE - 1, -1, -1):
        for j in range(numJ - 1, -1, -1):
            for k in range(min(j, maxE2J) + 1):
                ep, js = eprons[i], tuple(jprons[j - k : j + 1])
                beta[i][j - k] += beta[i + 1][j + 1] * prior[ep][js]

    # for row in beta:
    #     row.insert(0, 0)
    # beta.insert(0, [0 for x in beta[-1]])
    return beta
    #return [row[: numJ] for row in beta[: numE]]






'''
ReadEpronJpron
get word pairs from file
'''
def ReadEpronJpron(filename):
    wordPairs = []
    with open(filename) as fp:
        for i, line in enumerate(fp.readlines()):
            if i % 3 == 0:
                eword = line.strip('\n')
            elif i % 3 == 1:
                jword = line.strip('\n')
                wordPairs.append((eword, jword))

    return wordPairs

def ReadEpronJpron_arguments(lines):
    wordPairs = []

    for i, line in enumerate(lines):
        if i % 3 == 0:
            eword = line.strip('\n')
        elif i % 3 == 1:
            jword = line.strip('\n')
            wordPairs.append((eword, jword))

    return wordPairs

'''
EnumAligns
enumerate all possible alignments
ephon:  list of English phonemes
jphon:  list of Japanese phonemes
'''
def EnumAligns(ephon, jphon, pre = []):
    aligns = []
    ep = ephon[0]
    for i in range(1, min(3, len(jphon)) + 1):
        js = tuple(jphon[: i])
        if len(ephon) == 1 and len(jphon) - i == 0:
            aligns.append(pre + [(ep, js)])
        elif len(ephon) == 1 or len(jphon) - i == 0:
            continue
        else:
            s = [(ep, js)]
            post = EnumAligns(ephon[1 :], jphon[i :], s)
            for p in post:
                aligns.append(pre + p)

    return aligns



'''
InitProb
initialize probabilities of possible alignments
'''
def InitProb(pairs):
    counts = defaultdict(lambda: defaultdict(int))
    for ew, jw in pairs:
        ew, jw = ew.split(), jw.split()
        aligns = EnumAligns(ew, jw, [])
        for alt in aligns:
            for (ep, js) in alt:
                counts[ep][js] += 1

    # initialize probabilities from "observed" counts
    probs = defaultdict(lambda : defaultdict(float))
    for ep, js_co in counts.items():
        n = sum(js_co.values())
        for js, co in js_co.items():
            probs[ep][js] = float(co) / n

    return probs

###############################################################
def get_corpus_prob(align_prob,all_z,train_data):
    all_prob = 0
    for t_index in all_z:

        e_pron = train_data[t_index][0].split()
        j_pron = train_data[t_index][1].split()
        total_prob = 0
        for key,z in all_z[t_index].items():
            z = np.array(z)
            prob=1
            for i,esym in enumerate(e_pron):
                indexes = np.where(z==i)[0]
                jsym=tuple([j_pron[j] for j in indexes])
                prob*=align_prob[esym][jsym]
            total_prob+=prob
        all_prob+=log(total_prob)

    return Decimal(exp(1)) ** (Decimal(all_prob))




def Uniform_prob(train_data,alignments_dict):
    align_prob =  defaultdict(lambda : defaultdict(float))
    for i in range(len(train_data)):

        epron = np.array(train_data[i][0].split())
        jpron = np.array(train_data[i][1].split())

        for k in alignments_dict[i]:


            align_array=np.array(alignments_dict[i][k])
            for j in range(len(epron)):
                jp=tuple(list(jpron[list(np.where(align_array==j)[0])]))

                align_prob[epron[j]][jp] = 1

    for key in align_prob:
        normalize=sum(align_prob[key].values())
        for key1 in align_prob[key]:
            align_prob[key][key1] = align_prob[key][key1]/normalize

    return align_prob




def alignments(epron,jpron,k=3):
    alings = []

    def all_alignments(alings,cur_e=0,start_j=0):
        if cur_e >= len(epron) and start_j >= len(jpron):
            return [[]]

        if cur_e>=len(epron) and start_j<len(jpron):
            return [['None']]
        if cur_e<len(epron) and start_j>=len(jpron):
            return [['None']]


        temp_align=[]
        for j in range(start_j,start_j+k):
            if j< len(jpron):

                cur_align = [cur_e]* (j-start_j+1)

                result_align = all_alignments(alings,cur_e+1,j+1)
                for temp in result_align:
                    if temp!=['None']:
                        temp_align+=[cur_align+temp]
        return temp_align

    final_data = all_alignments(alings, 0, 0)

    final_dict={i:final_data[i] for i in range(len(final_data))}
    return final_dict







if __name__ == '__main__':
    fname = 'epron-jpron.data'


    lines=sys.stdin.readlines()
    arugments = sys.argv

    if len(arugments)>1:
         iterations=int(arugments[1])
    else:
        iterations = 5

    pairs = ReadEpronJpron_arguments(lines)

    #pairs = [('W AY N','W A I N')]

    all_z = {}
    for i in range(len(pairs)):
        all_z[i] = alignments(pairs[i][0].split(), pairs[i][1].split())

    uniform_prior =Uniform_prob(pairs, all_z)

    corpus_prob = get_corpus_prob(uniform_prior, all_z, pairs)
    print('iteration ', 0, ' ----- corpus prob=', corpus_prob,file=sys.stderr)





    prior = InitProb(pairs)


    iters = 5

    Uprior = prior
    for i in range(1,iters+1):

        prior = Uprior
        #########
        count = 0
        for key in prior:
            print(key, '|->', end=' ',file=sys.stderr)
            for key1 in prior[key]:
                if prior[key][key1] > 0.001:
                    count += 1
                    print(' '.join(key1), ':', round(prior[key][key1], 3), end=' ',file=sys.stderr)
            print('\n',file=sys.stderr)
        print('nonzeros =', count,file=sys.stderr)
        ##########
        corpus_prob = get_corpus_prob(prior, all_z, pairs)
        if i !=iters:
            print('iteration ', i, ' ----- corpus prob=', corpus_prob,file=sys.stderr)


        counts = Expectation(pairs, prior, 3)
        Uprior = Maximization(pairs, counts, prior)



    ######Writing probabilities.
    for key in prior:
        keys1=sorted(prior[key], key=prior[key].get, reverse=True)
        for key1 in keys1:
            prob = prior[key][key1]
            if prob>0.001:
                print(key,':',' '.join(key1),'#',round(prob,3))

