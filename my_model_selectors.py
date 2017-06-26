import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states, param_X=None, param_lengths=None):
        if param_X is None:
            param_X = self.X
        if param_lengths is None:
            param_lengths = self.lengths
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(param_X, param_lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        max_BIC = float("-inf")
        max_BIC_n = self.max_n_components
        
        log_n = math.log(len(self.X))
        for n in range(self.min_n_components, self.max_n_components):
            try:
                p = n
                model = self.base_model(n)
                log_l = model.score(self.X, self.lengths)
                BIC = -2 * log_l + p * log_n
                if BIC > max_BIC:
                    max_BIC = BIC
                    max_BIC_n = n
            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, n))

        # TODO implement model selection based on BIC scores
        return self.base_model(max_BIC_n)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        max_DIC = float("-inf")
        max_DIC_n = self.max_n_components
        
        log_l_sum = 0
        log_l_dict = {}
        
        for n in range(self.min_n_components, self.max_n_components):
            try:
                p = n
                model = self.base_model(n)
                log_l = model.score(self.X, self.lengths)
                log_l_sum = log_l_sum + log_l
                log_l_dict[n] = log_l
            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, n))
            
        M = self.max_n_components - self.min_n_components
        for n, log_l in log_l_dict.items():
            DIC = log_l - (1 / (M - 1)) * (log_l_sum - log_l)
            if DIC > max_DIC:
                max_DIC = DIC
                max_DIC_n = n
        
        return self.base_model(max_DIC_n)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        
        split_cnt = 3
        kf = KFold(n_splits=split_cnt)
        
        max_log_l = float("-inf")
        max_log_l_n = self.min_n_components
        
        for n in range(self.min_n_components, self.max_n_components):
            try:
                avg_logL = 0
                for train_index, test_index in kf.split(self.sequences):
                    cv_train_param_X, cv_train_param_lengths = combine_sequences(train_index, self.sequences)
                    model = self.base_model(n, param_X=cv_train_param_X, param_lengths=cv_train_param_lengths)
                    cv_test_param_X, cv_test_param_lengths = combine_sequences(test_index, self.sequences)
                    logL = model.score(cv_test_param_X, cv_test_param_lengths)
                    avg_logL = avg_logL + logL/split_cnt
                if avg_logL > max_log_l:
                    max_log_l = avg_logL
                    max_log_l_n = n
            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, n))
                
        return self.base_model(max_log_l_n)
