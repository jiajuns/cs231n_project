from __future__ import print_function
import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu



def BLUE_score(prediction, ground_truth, smooth=False, weights = None):
    """
    For a single prediction, given a list of reference ground true captions
    by directly calling this function, it will give you a blue score for that prediction
    """
    cc = SmoothingFunction()
    if smooth:
        return nltk.translate.bleu_score.sentence_bleu(ground_truth, prediction, smoothing_function=cc.method4)
    elif weights:
        return nltk.translate.bleu_score.sentence_bleu(ground_truth, prediction, weights)
    else:
        return nltk.translate.bleu_score.sentence_bleu(ground_truth, prediction)
    
if __name__ == '__main__':
    prediction = ['It', 'is', 'a', 'cat', 'at', 'room']
    ground_truth = [['It', 'is', 'a', 'cat', 'inside', 'the', 'room']]
    print(BLUE_score(prediction, ground_truth))