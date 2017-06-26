import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    
    for X, lengths in test_set.get_all_Xlengths().values():
        score_dict = {}
        max_log_l = float("-inf")
        guess = None
        
        for word, model in models.items():
            try:
                log_l = model.score(X, lengths)
            except:
                None
            score_dict[word] = log_l
            
            if log_l > max_log_l:
                max_log_l = log_l
                guess = word
        
        probabilities.append(score_dict)
        guesses.append(guess)
    
    return probabilities, guesses