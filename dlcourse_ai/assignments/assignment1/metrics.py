import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    gr = ground_truth.astype(int) 
    a = np.append(np.bincount(gr[prediction == True]), [0, 0])
    b = np.append(np.bincount(gr[prediction == False]), [0, 0])
    fp, tp = a[0], a[1]
    tn, fn = b[0], b[1]
    precision = tp/(tp + fp) if(tp + fp) != 0 else 1
    recall = tp/(tp + fn) if(tp + fn) != 0 else 1
    accuracy = (tp + tn)/(tp + fp + tn + fn)
    f1 = 2 * (precision * recall)/(precision + recall) if(precision + recall) != 0 else 0
    
    return precision, recall, f1, accuracy



def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    N = prediction.size
    accuracy = np.sum(prediction == ground_truth)/N
    
    return accuracy
