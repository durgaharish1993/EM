from collections import defaultdict
import itertools
import sys
from math import log
import random
from decimal import Decimal


def bi_gram_generator(train_data, vocab, gram=1, weight=(1, 0, 0)):
    window = (gram - 1)
    start_sequence = '*' * window
    end_sequence = '&'
    count_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    final_dict = defaultdict(lambda: defaultdict(float))

    for g in range(1, gram + 1):
        for tup in itertools.product(vocab, repeat=g):
            len_tup = g
            input = ''.join(tup[:len_tup - 1])
            output = tup[len_tup - 1]
            count_dict[g][input][output] = 1.0

    ####
    ##for unigram
    count_dict[1]['']['*'] = 1.0
    ##other wise
    for g in range(2, gram + 1):
        input = '*' * (g - 1)
        for letter in vocab:
            count_dict[g][input][letter] = 1.0

    ###########################################
    for line in train_data:
        temp_line = start_sequence + line + end_sequence
        for i in range(len(start_sequence), len(temp_line)):
            for j in range(window + 1):
                input = temp_line[i - j:i]
                output = temp_line[i]
                count_dict[j + 1][input][output] += 1

    ###########################################
    for gram in count_dict:
        for seq in count_dict[gram]:
            sum_data = sum(count_dict[gram][seq].values())
            count_dict[gram][seq].update((k, v / (sum_data)) for k, v in count_dict[gram][seq].items())
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
    vocab += ['_', '&']

    transition_prob = bi_gram_generator(train_data, vocab, gram, weight=weight)
    #####################

    return transition_prob


def forward_prop(transition_prob, emission_prob, observed_data, hidden_states):
    start_symbol = '*'
    end_symbol = '&'

    observed_data = start_symbol + observed_data + end_symbol
    alpha = defaultdict(lambda: defaultdict(float))
    alpha[0][start_symbol] = 1
    ##initialization the first layer of alpha calculations.
    for cur_h in vocab:
        alpha[1][cur_h] = transition_prob[start_symbol][cur_h] * emission_prob[cur_h][observed_data[1]]

    for i in range(2, len(observed_data) - 1):

        ###This loop for i hidden states.
        for cur_h in vocab:
            ###This loop for i-1 hidden states.
            sum_prob = 0
            for prev_h in alpha[i - 1].keys():
                sum_prob += alpha[i - 1][prev_h] * transition_prob[prev_h][cur_h] * emission_prob[cur_h][
                    observed_data[i]]
            if sum_prob != 0:
                alpha[i][cur_h] = sum_prob

    ##This is a special case where we need to add all transitions at the final state
    i = len(observed_data) - 1
    sum_prob = 0
    for prev_h in alpha[i - 1].keys():
        sum_prob += alpha[i - 1][prev_h] * transition_prob[prev_h][end_symbol]
    alpha[i][end_symbol] = sum_prob

    return alpha


def backward_prop(transition_prob, emission_prob, observed_data, hidden_states):
    start_symbol = '*'
    end_symbol = '&'
    observed_data = start_symbol + observed_data + end_symbol
    beta = defaultdict(lambda: defaultdict(float))

    ##Initialization length-2 index
    beta[len(observed_data) - 1][end_symbol] = 1
    for prev_h in vocab:
        beta[len(observed_data) - 2][prev_h] = transition_prob[prev_h][end_symbol]

    iter_list = list(range(1, len(observed_data) - 2))[::-1]

    for i in iter_list:
        ##This is a special case where we need to add all transitions at the final state

        for prev_h in vocab:
            sum_prob = 0
            for cur_h in beta[i + 1].keys():
                sum_prob += beta[i + 1][cur_h] * transition_prob[prev_h][cur_h] * emission_prob[cur_h][
                    observed_data[i + 1]]

            if sum_prob != 0:
                beta[i][prev_h] = sum_prob

    i = 0
    sum_prob = 0
    for cur_h in beta[i + 1].keys():
        sum_prob += transition_prob[start_symbol][cur_h] * emission_prob[cur_h][observed_data[i + 1]] * beta[i + 1][
            cur_h]
    beta[i][start_symbol] = sum_prob

    return beta


def estimating_state_transtion_count(alpha, beta, vocab, transition_prob, emission_prob, observed_data):
    '''
    FORMULA : not-quite-ξt(i, j) = αt(i)(aij bj(ot+1))βt+1(j)
    :param alpha: 
    :param beta: 
    :param vocab: 
    :return: 

    '''
    start_symbol = '*'
    end_symbol = '&'

    eplison = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    observed_data = start_symbol + observed_data + end_symbol
    normalize = alpha[len(observed_data) - 1][end_symbol]
    for t in range(0, len(observed_data) - 1):
        prev_states = alpha[t].keys()
        cur_states = alpha[t + 1].keys()
        for prev_h in prev_states:
            for cur_h in cur_states:
                if observed_data[t + 1] != end_symbol:
                    eplison[t][prev_h][cur_h] = ((alpha[t][prev_h]) * (
                    transition_prob[prev_h][cur_h] * emission_prob[cur_h][observed_data[t + 1]]) * (
                                                 beta[t + 1][cur_h])) / normalize
                else:
                    eplison[t][prev_h][cur_h] = ((alpha[t][prev_h]) * (transition_prob[prev_h][cur_h]) * (
                    beta[t + 1][cur_h])) / normalize

    # print('dkfjdkfjdk')

    return eplison


def estimating_occupancy_count(alpha, beta, vocab,observed_data):
    '''

    :param alpha: forward estimates
    :param beta: backward estimates
    :param vocab: 
    :return: updated observation probability


    '''
    start_symbol = '*'
    end_symbol = '&'

    observed_data = start_symbol + observed_data + end_symbol
    normalize = beta[0][start_symbol]

    gamma = defaultdict(lambda: defaultdict(float))

    for t in range(1, len(observed_data) - 1):
        for cur_h in beta[t].keys():
            gamma[t][cur_h] = (alpha[t][cur_h] * beta[t][cur_h]) / normalize

    return gamma


def EM_step(transition_prob, emission_prob, observed_data_list, vocab):
    alpha_dict ={}
    beta_dict = {}
    for ex_index in range(len(observed_data_list)):
        alpha_dict[ex_index] = forward_prop(transition_prob, emission_prob, observed_data_list[ex_index], vocab)
        beta_dict[ex_index] = backward_prop(transition_prob, emission_prob, observed_data_list[ex_index], vocab)

    #transition_count = estimating_state_transtion_count(alpha, beta, vocab, transition_prob, emission_prob,observed_data)


    return alpha_dict, beta_dict


def Max_step(alpha_dict, beta_dict, observed_data_list, vocab):
    # updated_transition_prob = defaultdict(lambda: defaultdict(float))
    # #####################Transition Probability
    # for t in transition_count:
    #
    #     for prev_h in transition_count[t]:
    #         for cur_h in transition_count[t][prev_h]:
    #             updated_transition_prob[prev_h][cur_h] += transition_count[t][prev_h][cur_h]
    #
    # ##normalizing the values.
    # for prev_h in updated_transition_prob:
    #     sum_prob = sum(updated_transition_prob[prev_h].values())
    #     for cur_h in updated_transition_prob[prev_h]:
    #         updated_transition_prob[prev_h][cur_h] = updated_transition_prob[prev_h][cur_h] / sum_prob



    ########### Emission Probability

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
    occupancy_count ={}
    for key in alpha_dict:
        occupancy_count[key] = estimating_occupancy_count(alpha_dict[key], beta_dict[key], vocab, observed_data_list[key])

    updated_emission_prob = defaultdict(lambda: defaultdict(float))
    sum_prob = defaultdict(float)
    for i in range(len(observed_data_list)):
        observed_data = start_symbol + observed_data_list[i] + end_symbol
        for t in range(1, len(observed_data) - 1):
            for cur_h in occupancy_count[i][t]:
                updated_emission_prob[cur_h][observed_data[t]] += occupancy_count[i][t][cur_h]
                sum_prob[cur_h] += occupancy_count[i][t][cur_h]

    for cur_h in updated_emission_prob:
        for o in updated_emission_prob[cur_h]:
            updated_emission_prob[cur_h][o] = updated_emission_prob[cur_h][o] / sum_prob[cur_h]

# print('dfdkfjkdf')


    return updated_emission_prob






def Viterbi_best(transition_prob, emission_prob, observed_data):
    start_symbol = '*'
    end_symbol = '&'

    observed_data = start_symbol + observed_data + end_symbol
    best_val = defaultdict(lambda: defaultdict(float))
    best_path = defaultdict(lambda : defaultdict(str))

    #######################################################
    best_val[0][start_symbol] = 1

    ##initialization the first layer of best_val calculations.
    for cur_h in vocab:
        best_val[1][cur_h] = log(transition_prob[start_symbol][cur_h]) + log(emission_prob[cur_h][observed_data[1]])
        best_path[1][cur_h] = start_symbol

    #######################################################
    for i in range(2, len(observed_data) - 1):

        ###This loop for i hidden states.
        for cur_h in vocab:
            ###This loop for i-1 hidden states.
            best_prob = float('-inf')
            best_state = None
            for prev_h in best_val[i - 1].keys():
                prob = best_val[i - 1][prev_h] +log(transition_prob[prev_h][cur_h])+ log(emission_prob[cur_h][observed_data[i]])
                if prob>best_prob:
                    best_prob = prob
                    best_state = prev_h

            best_val[i][cur_h] = best_prob
            best_path[i][cur_h] = best_state


    ##This is a special case where we need to add all transitions at the final state
    #######################################################
    i = len(observed_data) - 1
    best_prob = float('-inf')
    best_state = None
    for prev_h in best_val[i - 1].keys():
        prob= best_val[i - 1][prev_h] + log(transition_prob[prev_h][end_symbol])
        if prob > best_prob:
            best_prob = prob
            best_state = prev_h


    best_val[i][end_symbol] = best_prob
    best_path[i][end_symbol] = best_state

    #######################################################
    ###backtracking algorithm
    i = len(observed_data)-1
    state = end_symbol
    output_str = []
    while True:
        next_state = best_path[i][state]
        if next_state==start_symbol:
            break
        output_str+=[next_state]
        state = next_state
        i-=1


    output_str = output_str[::-1]

    str_out = ''.join(output_str)


    return str_out


def calculate_entropy(final_dict,data,gram=2):

    window = (gram-1)
    start_sequence = '*' * window
    end_sequence = '&'
    #print('I am here')
    total_log_prob = 0
    total_characters = 0
    total_lines =0
    for line in data:
        temp_line = start_sequence+line+end_sequence
        total_lines+=1
        prod =1
        total_characters+=2
        for i in range(len(start_sequence),len(temp_line)):
            input = temp_line[i-window:i]
            output = temp_line[i]
            prod*=final_dict[input][output]
            total_characters+=1

        total_log_prob+=log(prod,2)

        #print(temp_line,prod)

    #print('Entropy Per Character:',total_characters,-total_log_prob/total_characters)
    #print('Entropy Per Line :',total_lines,-total_log_prob/total_lines)

    return -total_log_prob/total_characters, total_log_prob







if __name__ == '__main__':

    vocab = 'a b c d e f g h i j k l m n o p q r s t u v w x y z _'.split(' ')
    start_symbol = '*'
    end_symbol = '&'
    observed_data_list = ['gjkgcbkycnjpkovryrbgkcrvsczbkyhkahqvykgjvjkcpekrbkjxdjayrpmkyhkmhkyhkyvrcukrpkbjfjvcukihpygb',
                          'oqykcykujcbykhpjkejihavcyrakvjmrijkjxrbybkrpkjfjvzkiclhvkarfrurtcyrhpkhvkaquyqvjkrpkygjkshvue',
                          'auqobkrpkvjajpykzjcvbkgcfjkwhqpekohvcbkbhkerwwraquykyhkpjmhyrcyjksrygkygcykicpzkbgqpkgrbkaurjpyb',
                          'gjkbcrekygjkoqvjcqkiqbykgcfjkygjkerbdqyjkvjbhufjekozkwjovqcvzkrpkhvejvkyhkdjvwhvikygjkajpbqb',
                          'oqykygjkdqouraryzkbqvvhqperpmkygjkejocaujkajvycrpuzkbjvfjekyhkwhaqbkckbdhyurmgykhpkyghbjkjwwhvyb']


    ## This is a bigram Model.
    transition_prob = transition_probability()
    emission_prob = defaultdict(lambda: defaultdict(float))

    ###initialization for emission Probability to Uniform distribution.
    for tup in itertools.product(vocab, repeat=2):
        input = tup[0]
        output = tup[1]
        emission_prob[input][output] =  1/27.

    ##################Testing values

    iter = 50
    Uemission_prob = emission_prob
    for i in range(1,iter+1):
        output_list = []
        emission_prob = Uemission_prob
        ####################################################
        # print('dfdfdfd')
        for j in range(len(observed_data_list)):
            out_str=Viterbi_best(transition_prob, emission_prob, observed_data_list[j])
            output_list +=[out_str]

        entropy,logp_corpus = calculate_entropy(transition_prob, output_list, 2)
        ####################################################
        non_zeros=1
        for cur_h in emission_prob:
            for o in emission_prob[cur_h]:
                if emission_prob[cur_h][o]>0.001:
                    non_zeros +=1

        ####################################################
        print('epoch', i, 'logp(corpus)=',logp_corpus,'entropy=', round(entropy,3),'nonzeros=',non_zeros)

        for j in range(len(observed_data_list)):
            print(observed_data_list[j])
            print(output_list[j].replace('_',' '))
            print('\n')


        ####################################################
        alpha_dict, beta_dict = EM_step(transition_prob, emission_prob, observed_data_list, vocab)
        Uemission_prob=Max_step(alpha_dict, beta_dict, observed_data_list, vocab)


























