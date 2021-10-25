import numpy as np



training_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",")
print("Shape of the spam training data set:", training_spam.shape)
print(training_spam)

def estimate_log_class_priors(data):
    """
    Given a data set with binary response variable (0s and 1s) in the
    left-most column, calculate the logarithm of the empirical class priors,
    that is, the logarithm of the proportions of 0s and 1s:
    log(P(C=0)) and log(P(C=1))

    :param data: a two-dimensional numpy-array with shape = [n_samples, 1 + n_features]
                 the first column contains the binary response (coded as 0s and 1s).

    :return log_class_priors: a numpy array of length two
    """
    ### YOUR CODE HERE...
    zeros = np.count_nonzero(data[:,0]==0)
    ones = np.count_nonzero(data[:,0]==1)
    
    probability_zeros = zeros/1000
    probability_ones = ones/1000
    
    log_zeros = np.log(probability_zeros)
    log_ones = np.log(probability_ones)
    
    log_class_priors = np.array([log_zeros,log_ones])
    return log_class_priors

estimate_log_class_priors(training_spam)



def estimate_log_class_conditional_likelihoods(data, alpha=1.0):
    """
    Given a data set with binary response variable (0s and 1s) in the
    left-most column and binary features (words), calculate the empirical
    class-conditional likelihoods, that is,
    log(P(w_i | c)) for all features w_i and both classes (c in {0, 1}).

    Assume a multinomial feature distribution and use Laplace smoothing
    if alpha > 0.

    :param data: a two-dimensional numpy-array with shape = [n_samples, 1 + n_features]

    :return theta:
        a numpy array of shape = [2, n_features]. theta[j, i] corresponds to the
        logarithm of the probability of feature i appearing in a sample belonging 
        to class j.
    """
    ### YOUR CODE HERE...
    
    splitt = np.hsplit(data,[1])
    spam_or_ham = splitt[0]
    features = splitt[1]
    
    new_features = np.transpose(features)
    new_spam = np.transpose(spam_or_ham)
    
    number_of_emails = new_spam[0].size
    number_of_features = features[0].size
    
    results = np.zeros((2,54))
    results[1][0] = 1
    
    spam_count = 0 #1
    ham_count = 0 #0
    
    for i in range(number_of_features):
            for j in range(number_of_emails):
                if(new_features[i][j]==1):
                    if(new_spam[0][j]==0):
                        ham_count=ham_count+1
                    else:
                        spam_count=spam_count+1
            results[0][i] = ham_count
            results[1][i] = spam_count
            
            spam_count = 0 #1
            ham_count = 0 #0
            
            
            
    #laplace smoothing
    for i in range(2):
            for j in range(number_of_features):
                if(results[i][j]==0):
                    results[i][j] = alpha
                                      
    probabilities = np.divide(results,number_of_emails)
    
    
    for i in range(2):
            for j in range(number_of_features):
                    probabilities[i][j] = np.log(probabilities[i][j])
    
    return probabilities




def predict(new_data, log_class_priors, log_class_conditional_likelihoods):
    """
    Given a new data set with binary features, predict the corresponding
    response for each instance (row) of the new_data set.

    :param new_data: a two-dimensional numpy-array with shape = [n_test_samples, n_features].
    :param log_class_priors: a numpy array of length 2.
    :param log_class_conditional_likelihoods: a numpy array of shape = [2, n_features].
        theta[j, i] corresponds to the logarithm of the probability of feature i appearing
        in a sample belonging to class j.
    :return class_predictions: a numpy array containing the class predictions for each row
        of new_data.
    """
    ### YOUR CODE HERE...
    number_of_emails = np.transpose(new_data)[0].size
    number_of_features = new_data[0].size
    maxfinder = np.zeros((2, number_of_emails))
    ######
    for i in range(number_of_emails):
        maxfinder[0][i] =  maxfinder[0][i] + log_class_priors[0]
        maxfinder[1][i] =  maxfinder[1][i] + log_class_priors[1]
    
    classpredictions = np.zeros(number_of_emails)
       
    for i in range(number_of_emails):
        for j in range(number_of_features):
            if(new_data[i][j]==1):
                maxfinder[0][i] =  maxfinder[0][i] + log_class_conditional_likelihoods[0][j]
                maxfinder[1][i] =  maxfinder[1][i] + log_class_conditional_likelihoods[1][j]
    
    for i in range(number_of_emails):
        if(maxfinder[0][i]<maxfinder[1][i]):
            classpredictions[i] = 1

    return classpredictions



def accuracy(y_predictions, y_true):
    """
    Calculate the accuracy.
    
    :param y_predictions: a one-dimensional numpy array of predicted classes (0s and 1s).
    :param y_true: a one-dimensional numpy array of true classes (0s and 1s).
    
    :return acc: a float between 0 and 1 
    """
    ### YOUR CODE HERE...
    number_of_emails = y_predictions.size
    correct_counter = 0
    
    for i in range(number_of_emails):
        if(y_predictions[i] == y_true[i]):
            correct_counter = correct_counter + 1
    
    acc = correct_counter/number_of_emails
    print(acc)
    return acc
    



class_priors = estimate_log_class_priors(testing_spam)
cond_likelihoods = estimate_log_class_conditional_likelihoods(testing_spam,alpha = 1.0)
results = testing_spam[:, 0]
class_predictions = predict(testing_spam[:, 1:], class_priors, cond_likelihoods)

testing_set_accuracy = accuracy(class_predictions,results)
print(testing_set_accuracy)






