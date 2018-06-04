import numpy as np
import scipy.sparse as sp
import re
import json
import time
import math
import pickle
from scipy.sparse import linalg as spl
from collections import defaultdict
from IPython.display import Markdown, display
import matplotlib.pyplot as plt
from IPython.display import Image

data = np.load("task03_data.npy")
reviews_1star = data.item()["reviews_1star"]
reviews_5star = data.item()["reviews_5star"]

class HMM_Params:
    
    def __init__(self,n_states,n_symbols):
        """ Makes three randomly initialized stochastic matrices `self.A`, `self.B`, `self.pi`.
        
        Parameters
        ----------
        n_states: int
                  number of possible values for Z_t.
        n_symbols: int
                  number of possible values for X_t.
                  
        Returns
        -------
        None
        
        """
        
        self.A  = self.rnd_stochastic_mat(n_states,n_states)
        self.B  = self.rnd_stochastic_mat(n_states,n_symbols)
        self.pi = self.rnd_stochastic_mat(1,n_states).transpose()
        
    def rnd_stochastic_mat(self,I,J):
        """ Retruns a randomly initialized stochastic matrix with shape (I,J).
        
        Parameters
        ----------
        I: int
           shape[0] of desired matrix.
        J: int
           shape[1] of disired matrix.
                  
        Returns
        -------
        x: np.ndarray
           a rondom stochastic matrix with shape (I,J)
        
        """
        
        x = np.full((I,J),(1/J))
        x = x + (np.random.randn(I,J)*(1.0/(J*J)))
        x = x/np.reshape(np.sum(x,axis=1),newshape=(I,1))
        
        return x
    


class HMM_TxtGenerator:
    
    def __init__(self,corpus,K):
        """Given the set of sentences `corpus` and number of states `K`, builds an HMM.
           Firstly it makes the volcabulary `self.word_list` based on all present words in 
           `corpus`. The variable `self.word_list` is a list of words. Then index of the word
           `self.word_list[v]` is v. Moreover, this function constructs `self.model_params`
           which is an instance of randomly initialized `HMM_Params`.
    
        Parameters
        ----------
        corpus : A list of sentences. Each sentence is a list of words.  
            We will learn model_params using sentences in `corpus`.
        K: int
           Number of possible states, i.e. Z_t \in {0,...,K-1}
        
    
        Returns
        -------
        None :
        """
        
        self.corpus = corpus.copy()
        self.K = K
        
        #collect all words ---
        word_dic = {}
        
        for sent in self.corpus:
            for w in sent:
                if(w in word_dic):
                    
                    word_dic[w] = word_dic[w] + 1
                    
                else:
                    
                    word_dic[w] = 1
                    
        self.word_list = [u for u in word_dic.keys()]
        self.word_dic  = word_dic
        self.V = len(self.word_list)
        #init params
        self.model_params = HMM_Params(K,len(self.word_list))
    
    def forwards_backwards(self,sentence_in):
        """Does the forwards-backwards algorithm for an observed list of words
           (i.e. and observed sentence).
    
        Parameters
        ----------
        sentence_in : a list of T words. Each word is a string.
                      You can convert `sentence_in` to a sequence of word-indices
                      as `x = self.sentence_to_X(sentence_in)`. 
            
        Returns
        -------
        alpha : np.ndarray, shape=(T,K)
                alpha(t,k) = Pr(Z_t=k,x[1:t])
                
        beta  : np.ndarray, shape=(T,K)
                beta(t,k)  = Pr(X_{t+1:T}|Z_t=k)
                
        log_likelihood  : scalar
                log probability of evidence, Pr(X_{1:T}=sentence_in) 
        """
        
        ### YOUR CODE HERE ### ===> DONE
        
        K = self.K
        A = self.model_params.A
        B = self.model_params.B
        
        #number of words in sentence corresponds to number of time steps
        T = len(sentence_in)
        
        #sequence of the corresponding word indices to the words in sentence_in
        seq_words_idx = self.sentence_to_X(sentence_in)
        
        #init alpha and beta
        alpha = np.empty((T, K))
        beta = np.empty((T, K))
        
        #init alpha vector for first time step
        alpha[0] = np.ravel(self.model_params.pi) * B[:, seq_words_idx[0]]
        #init beta vector for last time step
        beta[T-1] = np.ones(K)
        
        #iteration over all remaining time steps
        for t in range(T-1):
            #forward step
            alpha[t+1] = B[:, seq_words_idx[t+1]] * np.dot(A.transpose(), alpha[t])
            #backward step
            beta[T-t-2] = np.dot(A, B[:, seq_words_idx[T-t-1]] * beta[T-t-1])
    
        log_likelihood = np.log(np.sum(alpha[T-1]))
        
        return alpha, beta, log_likelihood
    
    def E_step(self,sentence_in):
        """Given the observed `sentence_in`, computes E[N(i,j)], E[M(i,j)], and E[L(k)].
           The notation here is compatible with notations of lecture slides, slides 44-46.
           Hint: You can begin by computing alpha and beta as
                    `forwards_backwards(self,sentence_in)`
        
        Parameters
        ----------
        sentence_in : a list of T words. Each word is a string.
                      You can convert sentence_in to a sequence of word-indices
                      as `x = self.sentence_to_X(sentence_in)`. 
            
        Returns
        -------
        EN : np.ndarray, shape=(K,K)
             Contains values for E[N(i,j)], where N(i,j) is the expected number of 
             transitions [i=>j] in the sequence [Z_1,..,Z_T], given the observed `sentence_in`.
        EM : np.ndarray, shape=(K,V)
             Contains values for E[M(i,v)], where M(i,v) is the expected number of 
             transitions [i=>v] in the sequence [Z_t,X_t], given the observed `sentence_in`.
        EL : np.ndarray, shape=(K,1)
             Contains values for E[L(k)]  where L(k)=[Z_1==k] given the observed `sentence_in`.
        """
        
        ### YOUR CODE HERE ### ===> DONE
        
        K = self.K
        V = self.V
        A = self.model_params.A
        B = self.model_params.B
        
        #number of words in sentence corresponds to number of time steps
        T = len(sentence_in)
        
        #sequence of the corresponding word indices to the words in sentence_in
        seq_words_idx = self.sentence_to_X(sentence_in)
        
        #compute alpha and beta
        alpha, beta, _ = self.forwards_backwards(sentence_in)
        
        #compute gamma 
        #gamma[t,i] = Pr(Z_t = i | X_{1:T})
        gamma = (alpha * beta) / np.reshape(np.sum(alpha*beta, axis=1), newshape=(T,1))
        
        #compute xi
        #xi[t,i,j] = Pr(Z_t = i, Z_{t+1} = j, X_{1:T})
        xi = np.empty((T-1,K,K))
        for t in range(T-1):
            xi_t = np.reshape(alpha[t], newshape=(K,1)) * A * \
                    np.reshape(beta[t+1] * B[:, seq_words_idx[t+1]], newshape=(1,K))
            #normalize
            xi[t] = xi_t / np.sum(xi_t)
        
        #compute the expected values for N
        EN = np.sum(xi, axis=0)
        #compute the expected values for M (first init)
        EM = np.zeros((K,V))
        EM[:, seq_words_idx] += gamma.transpose()
        #compute the expected values for L
        EL = np.reshape(gamma[0], newshape=(K,1))
        
        return EN, EM, EL
        
    
    def generate_sentence(self,sentence_length):
        """ Given the model parameter,generates an observed
            sequence of length `sentence_length`.
            Hint: after generating a list of word-indices like `x`, you can convert it to
                  an actual sentence as `self.X_to_sentence(x)`
            
        Parameters
        ----------
        sentence_length : int,
                        length of the generated sentence.
            
        Returns
        -------
        sent : a list of words, like ['the' , 'food' , 'was' , 'good'] 
               a sentence generated from the model.
        """
        
        ### YOUR CODE HERE ### ==> DONE
        
        sent = []
        Z = np.zeros_like(self.model_params.pi) #hidden state probabilites
        X = np.zeros((1, len(self.word_list))) #evidence probabilites
        #compute next timestep by applying
        #Z_t+1 = Z_t * A
        #X_t = Z_t * B
        for i in range(sentence_length):
            #for init timestep
            if (i == 0):
                Z = self.model_params.pi.T
            #for all other timesteps
            else:
                Z = Z.dot(self.model_params.A)
            X = np.reshape(Z.dot(self.model_params.B), len(self.word_list))
            #pick a random word w.r.t. the probabilities of the single words
            sent.append(np.random.choice(np.arange(0, len(self.word_list), 1), p=X))
        #convert list of integers to actual word list
        sent = self.X_to_sentence(sent)    
        return sent
        
    
    def X_to_sentence(self,input_x):
        """Convert a list of word-indices to an actual sentence (i.e. a list of words).
           To convert a word-index to an actual word, it looks at `self.word_list`.
           
    
        Parameters
        ----------
        input_x : a list of integer
                  list of word-indices, like [0,6,1,3,2,...,1]
        
    
        Returns
        -------
        sent : a list of words like ['the', 'food', 'was', 'good']
        """
        
        sent = []
        V = len(self.word_list)
        
        for u in input_x:
            if(u<V):
                
                sent.append(self.word_list[u])
                
            else:
                
                raise Exception("values of input_x have to be in " +\
                                str([0,V-1])  + ", but got the value " + str(u) + ".")
                
        return sent
    
    def sentence_to_X(self,input_sentence):
        """Convert a sentence (i.e. a list of words) to a list of word-indices.
           Index of the word `w` is `self.word_list.index(w)`.
           
    
        Parameters
        ----------
        input_sentence : list
                         a list of words like ['the', 'food', 'was', 'good']
        
        Returns
        -------
        X : list
            a list of word-indices like [50,4,3,20]
        """
        
        X = []
        
        for w in input_sentence:
            
            X.append(self.word_list.index(w))
            
        return X
    
    def is_in_vocab(self,sentence_in):
        """Checks if all words in sentence_in are in vocabulary.
           If `sentence_in` contains a word like `w` which is not in `self.word_list`,
           it means that we've not seen word `w` in training set (i.e. `curpus`).
           
    
        Parameters
        ----------
        sentence_in : list
                      a list of words like ['the', 'food', 'was', 'good']
        
        Returns
        -------
        to_ret : boolean
            [We've seen all words in `sentence_in` when training model-params.]
        """
        
        to_return = True
        
        for w in sentence_in:
            if(w not in self.word_list):
                
                to_return = False
                
        return to_return
    
    def update_params(self):
        """ One update procedure of the EM algorithm.
            - E-step: For each sentence like `sent` in corpus, it firstly computes expected
                     number of transitions in posterior distribution by calling 
                    `sent_EN,sent_EM,sent_EL = self.E_step(sent)`. Then it sums them up to get
                     expected number of total transitions in posterior distribution.
            - M-step: makes accumulated EN,EM,EL row-normalized, and assigned the row-normalized
                      values to A,B,pi.
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        
        #E-step
        K = self.K
        V = self.V
        sum_EN , sum_EM , sum_EL = np.zeros([K,K]),np.zeros([K,V]),np.zeros([K,1])
        for sent in self.corpus:
            
            en,em,el = self.E_step(sent)
            sum_EN = sum_EN + en
            sum_EM = sum_EM + em
            sum_EL = sum_EL + el
            
        #M-step
        A_new  = sum_EN / np.reshape(np.sum(sum_EN,axis=1),newshape=(K,1))
        B_new  = sum_EM / np.reshape(np.sum(sum_EM,axis=1),newshape=(K,1))
        pi_new = sum_EL / np.sum(sum_EL)
        self.model_params.A = A_new
        self.model_params.B = B_new
        self.model_params.pi = pi_new
    
    def learn_params(self,num_iter):
        """ Runs update procedures of the EM-algorithm for `num_iter` iterations.
        
        Parameters
        ----------
        num_iter: int
                  number of iterations.
        
        Returns
        -------
        history_loglik: list of floats
                `history_loglik[t]` is log-probability of training data in iteration `t`.
        """
        
        history_loglik = []
        
        for counter in range(num_iter):
            
            print("iteration " + str(counter) +\
                  " of " + str(num_iter) , end="\n")
            history_loglik.append(self.loglik_corpus())
            self.update_params()
            
        return history_loglik
    
    def loglik_corpus(self):
        """ Computes log-likelihood of the corpus based on current parameters.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        loglik: float
                log-likelihood of the corpus based on current parameters.
        
        """
        
        loglik = 0
        
        for sent in self.corpus:
            
            _,_,loglik_of_sent = self.forwards_backwards(sent)
            loglik += loglik_of_sent
            
        return loglik
    
    def loglik_sentence(self,sentence_in):
        """ Computes log-likelihood of `sentence_in` based on current parameters.
        
        Parameters
        ----------
        sentence_in: a list of words
        
        Returns
        -------
        loglik_of_sent: float
                        log-likelihood of `sentence_in` based on current parameters.
        """
        
        #check if all words are in corpus.
        for w in sentence_in:
            if(w not in self.word_list):
                
                return -np.Inf
            
        _,_,loglik_of_sent = self.forwards_backwards(sentence_in)
        
        return loglik_of_sent



train_percentage = 80

def split_to_traintest(in_list,percentage):
    
    n_train = math.floor(len(in_list)*percentage/100)
    n_test  = len(in_list) - n_train
    
    return in_list[0:n_train],in_list[n_train:]


reviews_1star_train , reviews_1star_test = split_to_traintest(reviews_1star,train_percentage)
reviews_5star_train , reviews_5star_test = split_to_traintest(reviews_5star,train_percentage)
reviews_test = reviews_1star_test + reviews_5star_test

y_test  = [1 for i in range(len(reviews_1star_test))] + \
          [5 for i in range(len(reviews_5star_test))]

"""
K = 8
hmm_1 = HMM_TxtGenerator(reviews_1star_train,K)
hmm_5 = HMM_TxtGenerator(reviews_5star_train,K)

n_iter = 50
history_loglik_1 = hmm_1.learn_params(n_iter)
plt.figure()
plt.plot(range(len(history_loglik_1)) , history_loglik_1)
plt.xlabel("iteration",fontsize=16)
plt.ylabel("log-likelihood",fontsize=16)
plt.show()

history_loglik_5 = hmm_5.learn_params(n_iter)
plt.figure()
plt.plot(range(len(history_loglik_5)) , history_loglik_5)
plt.xlabel("iteration",fontsize=16)
plt.ylabel("log-likelihood",fontsize=16)
plt.show()

pickle.dump(hmm_1, open("hmm1.p", "wb"))
pickle.dump(hmm_5, open("hmm5.p", "wb"))

"""
#load data from file
hmm_1 = pickle.load(open("hmm1.p", "rb"))
hmm_5 = pickle.load(open("hmm5.p", "rb"))


temp_reviews = []
temp_y = []
for counter in range(len(reviews_test)):
    current_review = reviews_test[counter]
    current_y   = y_test[counter]
    if(hmm_1.is_in_vocab(current_review) | hmm_5.is_in_vocab(current_review)):
        temp_reviews.append(current_review)
        temp_y.append(current_y)
reviews_test_filtered = temp_reviews
y_test_filtered = temp_y


def classify_review(hmm_1,hmm_5,p,sentence_in):
    """Given the trained models `hmm_1` and `hmm_2` and frequency of
       1-star reviews, classifies `sentence_in` 
    
    Parameters
    ----------
    hmm_1 : HMM_TxtGenerator
        The trained model on 1-star reviews.
    hmm_5 : HMM_TxtGenerator
        The trained model on 5-star reviews.
    p: a scalar in [0,1]
        frequency of 1-star reviews, (#1star)/(#1star + #5star)
    
    Returns
    -------
    c : int in {1,5}
        c=1 means sentence_in is classified as 1. 
        similarly c=5 means sentence_in is classified as 5.
        
    """
    
    ### YOUR CODE HERE ###
    
    
    
    
    
    
    
p = len(reviews_1star_train)/(len(reviews_1star_train)+len(reviews_5star_train))
y_pred = []
for sent in reviews_test_filtered:
    y_pred.append(classify_review(hmm_1,hmm_5,p,sent))
accuracy = np.sum(np.array(y_pred)==np.array(y_test_filtered))/len(y_test_filtered)
print("classification accuracy for " + str(len(y_test_filtered)) +\
      " test instances: " + str(accuracy))

sample_1star = hmm_1.generate_sentence(15)
sample_5star = hmm_5.generate_sentence(15)
print("generated 1star review: ")
print(sample_1star)
print("\n")
print("generated 5star review: ")
print(sample_5star)