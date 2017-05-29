from collections import defaultdict
import itertools
import sys
from math import log
from decimal import Decimal


def bi_gram_generator(train_data,vocab,gram=1,weight=(1,0,0)):
    window = (gram-1)
    start_sequence = '*' * window
    end_sequence = '&'
    count_dict=defaultdict(lambda : defaultdict(lambda : defaultdict(float)))
    final_dict = defaultdict(lambda :defaultdict(float))

    for g in range(1,gram+1):
        for tup in itertools.product(vocab,repeat=g):
            len_tup = g
            input = ''.join(tup[:len_tup-1])
            output = tup[len_tup-1]
            count_dict[g][input][output] = 1.0


    ####
    ##for unigram
    count_dict[1]['']['*']=1.0
    ##other wise
    for g in range(2, gram + 1):
        input = '*' * (g-1)
        for letter in vocab:
            count_dict[g][input][letter]=1.0


    ###########################################
    for line in train_data:
        temp_line = start_sequence+line+end_sequence
        for i in range(len(start_sequence),len(temp_line)):
            for j in range(window+1):
                input = temp_line[i-j:i]
                output = temp_line[i]
                count_dict[j+1][input][output]+=1


    ###########################################
    for gram in count_dict:
        for seq in count_dict[gram]:
            sum_data = sum(count_dict[gram][seq].values())
            count_dict[gram][seq].update( (k,v/(sum_data)) for k,v in count_dict[gram][seq].items())
    ###########################################

    return count_dict[gram]

def transition_probability():
    #######################################################
    train_file = 'train.txt'
    gram = 2
    weight = (1, 0)
    fp = open(train_file, 'r')
    train_data = []
    for line in fp.readlines():
        train_data += [line[:-1].replace(' ', '_')]


    vocab = 'a b c d e f g h i j k l m n o p q r s t u v w x y z'.split(' ')
    vocab +=['_','&']


    transition_prob =bi_gram_generator(train_data,vocab,gram,weight=weight)
    #####################

    return transition_prob





def forward_prop(transition_prob,emission_prob,observed_data,hidden_states):

    start_symbol = '*'
    end_symbol = '&'

    observed_data = start_symbol+observed_data+end_symbol
    alpha = defaultdict(lambda : defaultdict(float))
    alpha[0][start_symbol] = 1
    ##initialization the first layer of alpha calculations.
    for cur_h in vocab:
        alpha[1][cur_h] = transition_prob[start_symbol][cur_h] * emission_prob[cur_h][observed_data[1]]


    for i in range(2,len(observed_data)-1):

        ###This loop for i hidden states.
        for cur_h in vocab:
            ###This loop for i-1 hidden states.
            sum_prob = 0
            for prev_h in alpha[i-1].keys():
                sum_prob += alpha[i - 1][prev_h] * transition_prob[prev_h][cur_h]*emission_prob[cur_h][observed_data[i]]
            if sum_prob!=0:
                alpha[i][cur_h]=sum_prob


    ##This is a special case where we need to add all transitions at the final state
    i = len(observed_data)-1
    sum_prob =0
    for prev_h in alpha[i-1].keys():
        sum_prob+=alpha[i - 1][prev_h] * transition_prob[prev_h][end_symbol]
    alpha[i][end_symbol]= sum_prob


    return alpha



def backward_prop(transition_prob,emission_prob,observed_data,hidden_states):
    start_symbol = '*'
    end_symbol = '&'
    observed_data = start_symbol + observed_data + end_symbol
    beta = defaultdict(lambda: defaultdict(float))

    ##Initialization length-2 index
    beta[len(observed_data)-1][end_symbol]=1
    for prev_h in vocab:
        beta[len(observed_data)-2][prev_h] = transition_prob[prev_h][end_symbol]


    iter_list = list(range(1,len(observed_data)-2))[::-1]

    for i in iter_list:
        ##This is a special case where we need to add all transitions at the final state

        for prev_h in vocab:
            sum_prob = 0
            for cur_h in beta[i+1].keys():
                sum_prob += beta[i + 1][cur_h] * transition_prob[prev_h][cur_h] * emission_prob[cur_h][observed_data[i+1]]

            if sum_prob != 0:
                beta[i][prev_h] = sum_prob


    i =0
    sum_prob=0
    for cur_h in beta[i+1].keys():
        sum_prob+= transition_prob[start_symbol][cur_h]*emission_prob[cur_h][observed_data[i+1]]*beta[i+1][cur_h]
    beta[i][start_symbol]=sum_prob

    return beta




def estimating_state_transtion_count(alpha,beta,vocab,transition_prob,emission_prob,observed_data):
    '''
    FORMULA : not-quite-ξt(i, j) = αt(i)(aij bj(ot+1))βt+1(j)
    :param alpha: 
    :param beta: 
    :param vocab: 
    :return: 
    
    '''
    start_symbol = '*'
    end_symbol = '&'

    eplison = defaultdict(lambda : defaultdict(lambda :defaultdict(float)))
    observed_data = start_symbol + observed_data + end_symbol
    normalize = alpha[len(observed_data)-1][end_symbol]
    for t in range(0,len(observed_data)-1):
        prev_states = alpha[t].keys()
        cur_states = alpha[t+1].keys()
        for prev_h in prev_states:
            for cur_h in cur_states:
                eplison[t][prev_h][cur_h] = ((alpha[t][prev_h]) * (transition_prob[prev_h][cur_h]*emission_prob[cur_h][observed_data[t+1]]) * (beta[t+1][cur_h]))/normalize


    return eplison










def estimating_occupancy_count(alpha,beta,vocab,transition_prob,emission_prob,observed_data):
    '''
    
    :param alpha: forward estimates
    :param beta: backward estimates
    :param vocab: 
    :return: updated observation probability
    
    
    '''
    start_symbol = '*'
    end_symbol = '&'

    observed_data = start_symbol+observed_data+end_symbol
    normalize = beta[0][start_symbol]

    gamma = defaultdict(lambda : defaultdict(float))

    for t in range(1,len(observed_data)-1):
        for cur_h in beta[t].keys():
            gamma[t][cur_h] = (alpha[t][cur_h]*beta[t][cur_h])/normalize

    return gamma






def EM_step(transition_prob,emission_prob,observed_data,vocab):
    alpha = forward_prop(transition_prob,emission_prob,observed_data,vocab)
    beta = backward_prop(transition_prob,emission_prob,observed_data,vocab)



    transition_count = estimating_state_transtion_count(alpha,beta,vocab,transition_prob,emission_prob,observed_data)
    occupancy_count = estimating_occupancy_count(alpha,beta,vocab,transition_prob,emission_prob,observed_data)



    return transition_count, occupancy_count







def Max_step(transition_count,occupancy_count,observed_data,vocab):
    '''
    returns up
    :param transition_count: 
    :param occupancy_count: 
    :param observed_data: 
    :param vocab: 
    :return: 
    '''

    start_symbol = '*'
    end_symbol = '&'
    updated_transition_prob = defaultdict(lambda: defaultdict(float))
    updated_emission_prob = defaultdict(lambda : defaultdict(float))
    observed_data = start_symbol+observed_data+end_symbol


    #####################Transition Probability
    for t in transition_count:

        for prev_h in transition_count[t]:
            for cur_h in transition_count[t][prev_h]:
                updated_transition_prob[prev_h][cur_h]+=transition_count[t][prev_h][cur_h]

    ##normalizing the values.
    for  prev_h in updated_transition_prob:
        sum_prob=sum(updated_transition_prob[prev_h].values())
        for cur_h in updated_transition_prob[prev_h]:
            updated_transition_prob[prev_h][cur_h] =  updated_transition_prob[prev_h][cur_h]/sum_prob



    ########### Emission Probability
    sum_prob = defaultdict(float)
    for t in range(1,len(observed_data)-1):
        for cur_h in occupancy_count[t]:
            updated_emission_prob[cur_h][observed_data[t]] +=occupancy_count[t][cur_h]
            sum_prob[cur_h] +=occupancy_count[t][cur_h]

    for cur_h in updated_emission_prob:
        for o in updated_emission_prob[cur_h]:
            updated_emission_prob[cur_h][o] = updated_emission_prob[cur_h][o]/sum_prob[cur_h]













    return updated_transition_prob,updated_emission_prob












if __name__=='__main__':


    vocab = 'a b c d e f g h i j k l m n o p q r s t u v w x y z _'.split(' ')
    start_symbol = '*'
    end_symbol = '&'
    #observed_data = 'gjkgcbkycnjpkovryrbgkcrvsczbkyhkahqvykgjvjkcpekrbkjxdjayrpmkyhkmhkyhkyvrcukrpkbjfjvcukihpygboqykcykujcbykhpjkejihavcyrakvjmrijkjxrbybkrpkjfjvzkiclhvkarfrurtcyrhpkhvkaquyqvjkrpkygjkshvueauqobkrpkvjajpykzjcvbkgcfjkwhqpekohvcbkbhkerwwraquykyhkpjmhyrcyjksrygkygcykicpzkbgqpkgrbkaurjpybgjkbcrekygjkoqvjcqkiqbykgcfjkygjkerbdqyjkvjbhufjekozkwjovqcvzkrpkhvejvkyhkdjvwhvikygjkajpbqboqykygjkdqouraryzkbqvvhqperpmkygjkejocaujkajvycrpuzkbjvfjekyhkwhaqbkckbdhyurmgykhpkyghbjkjwwhvyb'
    observed_data = 'durga'

    ## This is a bigram Model.
    transition_prob=transition_probability()
    emission_prob = defaultdict(lambda : defaultdict(float))



    ###initialization for emission Probability to Uniform distribution.
    for tup in itertools.product(vocab, repeat=2):
        input = tup[0]
        output = tup[1]
        emission_prob[input][output] = 1/27.


    ##################Testing values


    '''
           |	p(…|C) |	p(…|H)  | p(…|*)
    -------------------------------
    p(1|…) |	0.7	   |    0.1	    |
    p(2|…) |	0.2	   |    0.2	    |
    p(3|…) |	0.1	   |    0.7     |
    -------------------------------
    p(C|…) |	0.8	   |     0.1	| 0.5
    p(H|…) |	0.1	   |     0.8	| 0.5
    p(&|…) |	0.1	   | 	 0.1    |  0


    '''
    #
    transition_prob['C'] = {'C':0.8,'H':0.1,'&':0.1}
    transition_prob['H'] = {'C':0.1,'H':0.8,'&':0.1}
    transition_prob['*'] = {'C':0.5,'H':0.5,'&':0}

    emission_prob['C'] = {'1':0.7,'2':0.2,'3':0.1}
    emission_prob['H'] = {'1':0.1,'2': 0.2, '3': 0.7}

    vocab = ['C','H']

    observed_data = '233232322313311121113121112332322'
    iter=2
    for i in range(iter):
        transition_count,occupancy_count = EM_step(transition_prob,emission_prob,observed_data,vocab)
        Utransition_prob,Uemission_prob = Max_step(transition_count,occupancy_count,observed_data,vocab)
        transition_prob = Utransition_prob
        emission_prob = Uemission_prob
    -























