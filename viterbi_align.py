from collections import defaultdict
import numpy as np
import sys



def alignments(epron,jpron):
    #epron = ['W']
    #jpron = ['W','A','D']
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



def read_data(file_name):
    train_data=[]
    fp = open(file_name,'r')
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


def read_prob(file_name):
    align_prob = defaultdict(lambda :defaultdict(float))
    fp = open(file_name,'r')
    #EY : E E # 0.582
    for line in fp.readlines():
        data = line[:-1]
        e_p = data.split(':')[0].strip()
        j_p_str =data.split(':')[1]
        j_p=j_p_str.split('#')[0].strip()
        prob = float(j_p_str.split('#')[1].strip())

        align_prob[e_p][j_p] = prob

    return align_prob



def choose_best(all_z,align_prob,train_data):
    best_z = defaultdict(tuple)
    for t_index in all_z:

        e_pron = train_data[t_index][0].split()
        j_pron = train_data[t_index][1].split()

        best_prob = 0
        best_align = None
        for key,z in all_z[t_index].items():
            z = np.array(z)
            prob=1
            for i,esym in enumerate(e_pron):
                indexes = np.where(z==i)[0]
                jsym=' '.join([j_pron[j] for j in indexes])
                prob*=align_prob[esym][jsym]

            if prob > best_prob:
                best_prob = prob
                best_align  = (key,z,prob)
        best_z[t_index] = best_align

    return best_z

def read_data_arguments(lines):
    train_data = []
    count=0
    ep_bin=True
    for line in lines:
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

    # file_name ='epron-jpron.data'
    # file_name1 = 'epron-jpron.probs'
    arguments=sys.argv
    if len(arguments)>1:
        file_name1=arguments[1]


    lines = sys.stdin.readlines()

    train_data = read_data_arguments(lines)
    align_prob = read_prob(file_name1)


    all_z = {}
    for i in range(len(train_data)):
        all_z[i] = alignments(train_data[i][0].split(), train_data[i][1].split())



    best_z=choose_best(all_z,align_prob,train_data)


    ###printing_to a file in the format.
    for t_index in range(len(train_data)):
        e_pron = train_data[t_index][0]
        j_pron = train_data[t_index][1]
        best_align=' '.join(map(str, list(best_z[t_index][1] + 1)))
        print(e_pron)
        print(j_pron)
        print(best_align)







