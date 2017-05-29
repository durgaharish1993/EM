from collections import defaultdict
import numpy as np
from math import log,exp
from decimal import Decimal





def alignments(epron,jpron):
    alings = []

    def all_alignments(alings,cur_e=0,start_j=0):
        if cur_e >= len(epron) and start_j >= len(jpron):
            return [[]]

        if cur_e>=len(epron) and start_j<len(jpron):
            return [['None']]
        if cur_e<len(epron) and start_j>=len(jpron):
            return [['None']]


        temp_align=[]
        for j in range(start_j,start_j+5):
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






def e_step(train_data,all_z,fractional_counts,iter,soft=True):
    align_prob = defaultdict(lambda: defaultdict(float))

    for index in all_z:
        e_pron = train_data[index][0].split()
        j_pron = train_data[index][1].split()
        for key,z in all_z[index].items():
            z = np.array(z)
            for i,esym in enumerate(e_pron):
                indexes = np.where(z==i)[0]
                jsym=' '.join([j_pron[j] for j in indexes])
                align_prob[esym][jsym]+=fractional_counts[index][key]


    for esym in align_prob:
        sum_val = sum(align_prob[esym].values())
        for k,v in align_prob[esym].items():
            updated_value  = v/sum_val
            #if updated_value>=0.001:
            align_prob[esym][k]= updated_value#.update(k,updated_value)
            # else:
            #     align_prob[esym][k]=0#.update(k,0)

        #align_prob[esym].update((k,v/sum_val) for k,v in align_prob[esym].items())

    return align_prob





def m_step(align_prob,all_z,train_data):
    joint_prob = defaultdict(lambda : defaultdict(float))
    for index in all_z:
        e_pron = train_data[index][0].split()
        j_pron = train_data[index][1].split()
        for key,z in all_z[index].items():
            z = np.array(z)
            prob=1
            for i,esym in enumerate(e_pron):
                indexes = np.where(z==i)[0]
                jsym=' '.join([j_pron[j] for j in indexes])
                prob*=align_prob[esym][jsym]
            joint_prob[index][key]= prob
    return joint_prob




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
                jsym=' '.join([j_pron[j] for j in indexes])
                prob*=align_prob[esym][jsym]
            total_prob+=prob
        all_prob+=log(total_prob)

    return Decimal(exp(1)) ** (Decimal(all_prob))








def fractional_count_update(joint_prob,fractional_counts,train_data,align_prob):
    for t_index in joint_prob:
        sum_val = sum(joint_prob[t_index].values())
        for key in joint_prob[t_index]:
            fractional_counts[t_index][key]=joint_prob[t_index][key]/sum_val
    return fractional_counts







def read_data(file_name):
    train_data=[]
    fp = open(file_name)
    count=0
    ep_bin=True
    for line in fp.readlines():
        count+=1
        if count%3==0:
            train_data+=[(ep,jp)]
            ep_bin=True
            continue

        if ep_bin==True:
            ep = line[:-1]
            ep_bin=False
        else:
            jp=line[:-1]
            ep_bin=True

    return train_data






















if __name__=='__main__':
    file_name = 'epron-jpron.data'
    iterations =15


    #train_data = [('T EH S T','T E S U T O')]
    #train_data = [('W AY N', 'W A I N')]


    train_data=read_data(file_name)
    #train_data =train_data
    #train_data = [('B IY', 'B I I')]
    ###initalization

    align_prob = defaultdict(lambda: defaultdict(lambda :1))

    fractional_counts = defaultdict(lambda : defaultdict(float))
    all_z = {}
    for i in range(len(train_data)):
        all_z[i] = alignments(train_data[i][0].split(), train_data[i][1].split())
        for key in all_z[i]:
            fractional_counts[i][key]=1





    for i in range(iterations):

        if i!=0:
            corpus_prob = get_corpus_prob(align_prob, all_z, train_data)
            print('iteration ', i, ' ----- corpus prob=', corpus_prob)

        align_prob=e_step(train_data,all_z,fractional_counts,i,soft=True)

        #########

        count=0
        for key in align_prob:
            print(key, '|->', end=' ')
            for key1 in align_prob[key]:
                if align_prob[key][key1]>0.001:
                    count+=1
                    print(key1, ':', round(align_prob[key][key1],3), end=' ')
            print('\n')
        print('nonzeros =',count)
        ##########
        joint_prob = m_step(align_prob, all_z, train_data)
        fractional_counts=fractional_count_update(joint_prob,fractional_counts,train_data,align_prob)



    ######Writing probabilities.
    for key in align_prob:
        keys1=sorted(align_prob[key], key=align_prob[key].get, reverse=True)
        for key1 in keys1:
            prob = align_prob[key][key1]
            if prob>0.001:
                print(key,':',key1,'#',round(prob,3))





