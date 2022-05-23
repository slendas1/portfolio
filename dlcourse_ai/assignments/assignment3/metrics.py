def binary_classification_metrics(prediction, ground_truth):
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    gr = ground_truth.astype(int) 
    a = np.append(np.bincount(gr[prediction == True]), [0, 0])
    b = np.append(np.bincount(gr[prediction == False]), [0, 0])
    fp, tp = a[0], a[1]
    tn, fn = b[0], b[1]
    precision = tp/(tp + fp) if(tp + fp) != 0 else 1
    recall = tp/(tp + fn) if(tp + fn) != 0 else 1
    accuracy = (tp + tn)/(tp + fp + tn + fn)
    f1 = 2 * (precision * recall)/(precision + recall) if(precision + recall) != 0 else 0
    
    return accuracy, precision, recall, f1


def multiclass_accuracy(prediction, ground_truth):
    return 0
