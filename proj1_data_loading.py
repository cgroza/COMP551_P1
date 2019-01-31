import json  # we need to use the JSON package to load the data, since the data is stored in JSON format
import numpy
#from google.cloud import language
#from google.cloud.language import enums
#from google.cloud.language import types
import math
import time

with open("proj1_data.json") as fp:
    data = json.load(fp)


# Split the full data set into a training set, validating set, and testing set.
# Training set
training = data[:10000]
# Validating set
validating = data[10000:11000]
# Testing set
testing = data[11000:12000]

# This is the default learning rate constant that should be used for gradient descent.
DEFAULT_ALPHA = 0.000000005

# Number of word text features to be included in the model
NUM_TEXT = 0

# Helper function to Convert the comment text to lower case.
# comments: list of strings
def preprocess_words(comments):
    # Strings are immutable in Python
    return [comment.lower() for comment in comments]


# Returns an ordered list of the 160 most common words
# comments: list of strings to be considered
# txt_num: number of top ranked words to compute
def get_common_words(comments, txt_num):
    if txt_num == 0:
        return []
    word_counts = {}
    for comment in comments:
        for word in comment.split():
            if word in word_counts:
                word_counts[word] = word_counts[word] + 1
            else:
                word_counts[word] = 1
    # Return word list in decreasing occurrence order
    return [w[0] for w in reversed(sorted(word_counts.items(), key= lambda kv: kv[1]))][:txt_num]


# Counts the occurrence of word features in a comment.
# Returns a list of counts in the same order as words.
# featured_words: ordered word list to be counted
# comment: string to count words in.
def count_word_features(featured_words, comment):
    feature_counts = {}
    for word in featured_words:
        feature_counts[word] = 0
    for word in comment.split():
        if word in feature_counts:
            feature_counts[word] = feature_counts[word] + 1
    return [feature_counts[w] for w in featured_words]


# Function that queries the Google API for sentiment analysis.
# We did not use this due to number of allowed API calls limitation.
def sentiment_analysis(comment):
    client = language.LanguageServiceClient()
    document = types.Document(
        content=comment,
        type=enums.Document.Type.PLAIN_TEXT)

    sentiment = client.analyze_sentiment(document=document).document_sentiment
    return sentiment.score * sentiment.magnitude


# This function counts the length (number of words) of comment.
def count_word_length(comment):
    count = len(comment.split())
    return count


# Gathers the target feature from the data and outputs it as vector.
def build_target_vector(data):
    return numpy.array([d["popularity_score"] for d in data])


# Gathers features from the data and puts them in a matrix.
# Features are columns, examples are rows.
# data: list of comment dictionaries to be considered.
# txt_num: number of text word features to include (taken from the ranking of words).
def build_feature_matrix(data, txt_num):
    # Gather comment texts
    comments = preprocess_words([d["text"] for d in data])
    # Build list of common words to be included as features
    common_words = get_common_words(comments, txt_num)
    # A list of lists. Every sublist is a row.
    matrix = []
    for comment in data:
        # Initial features are controversiality and number of children features
        features = [comment["controversiality"], comment["children"]]
        # is_root features
        if comment["is_root"]:
            features.append(1)
        else:
            features.append(0)
        # Get counts for the common words
        word_counts = count_word_features(common_words, comment["text"])
        features = features + word_counts
        # Add word_length feature
        features.append(count_word_length(comment["text"]))
        # Add word_length*children interaction feature
        features.append(comment["children"] * count_word_length(comment["text"]))

        # Bias column
        features.append(1)
        # Add the row we just built to the matrix
        matrix.append(features)
    # Convert to efficient numpy array
    return numpy.array(matrix)

# Gradient descent implementation.
#x: features matrix
#y: target feature vector
#w: initial weight vector
#alpha: learning rate
def gradient_descent(x, y, w, alpha):
    # Precompute reusable values
    x_t = numpy.transpose(x)
    epsilon = 0.00001
    count = 2
    a = numpy.matmul(x_t, x)
    b = numpy.matmul(x_t, y)
    while True:
        gradient = 2 * numpy.subtract(numpy.matmul(a, w), b)
        w_0 = w
        # Compute weight deltas
        w = numpy.subtract(w_0, alpha * gradient)
        # Update weights
        theta = numpy.subtract(w, w_0)
        # Stop if we reached minimum precision
        if numpy.linalg.norm(theta) < epsilon:
            break
        count = count + 1
    return w


# Linearly applies the weights vector w to the new features X and
# returns a vector of predicted values
def apply_regression(w, X):
    return numpy.matmul(X, w)


# Error metrics. The R^2 represents the proportion of the variance explained by
# this model.
def r_squared(observed, predicted):
     obs_mean = observed.mean()
     total_ss = sum((observed - obs_mean)**2)
     explained_ss = sum((predicted - obs_mean)**2)
     return 1 - (explained_ss/total_ss)

# MSE error metric
def mean_squared_error(observed, predicted):
     error_ss = sum((predicted - observed)**2)
     return error_ss/len(observed)

# Closed form solution implementation.
#x: features matrix
#y: target feature vector
def least_squares_method(x, y):
    x_t = numpy.transpose(x)
    x_tx_inv = numpy.linalg.inv(numpy.matmul(x_t, x))
    x_tx_inv_x_t = numpy.matmul(x_tx_inv, x_t)
    w = numpy.matmul(x_tx_inv_x_t, y)
    return w

# This function runs function iterations number of times and measures the
# execution time. It writes a csv file to the file out. We will use it to
# gather data points and plot it with R.
def time_function(name, func, iterations, out):
    with open(out, 'w') as f:
        f.write(name+"\n")
        for i in range(0, iterations):
            t1 = time.time()
            func()
            t2 = time.time()
            f.writelines(str(t2 - t1)+"\n")

# Helper function for time benchmarks.
def time_least_squares():
    time_function("LEAST_SQUARES", lambda: least_squares_method(build_feature_matrix(training, 0), build_target_vector(training)), 20, "least_squares_time.csv")

def time_gradient_descent(alpha, alpha_str):
    print("Timing descentt with alpha = " + str(alpha))
    time_function("GRADIENT_DESCENT_"+alpha_str, lambda: gradient_descent(build_feature_matrix(training, 0), build_target_vector(training), numpy.random.rand(6), alpha), 20, alpha_str+"gradient_descent.csv")

# Helper function that evaluates the performance of the model in terms of MSE and R^2.
#weights: the coefficients of the model
#data: the data to apply linear regression to.
#txt_num: number of text features in the model.
def evaluate_model(weights, data, txt_num):
    # Run model on the data data.
    predicted = apply_regression(weights, build_feature_matrix(data, txt_num))
    # Report R^2 of the model.
    return {"R^2" : r_squared(build_target_vector(data), predicted),
            "MSE" : mean_squared_error(build_target_vector(data), predicted)}

# Example of function timing
#time_least_squares()
#time_gradient_descent()

# Train the model on the training data.
train_feature_matrix = build_feature_matrix(training, NUM_TEXT)
# Get the target feature vector.
train_target_matrix = build_target_vector(training)

# get the number of features in this model.
num_features = train_feature_matrix.shape[1]
print("Running on " + str(num_features) + " features")

# Pick a starting point for gradient descent
random_vector = numpy.random.rand(num_features)
# Train with least squares
weights = least_squares_method(train_feature_matrix, train_target_matrix)
# Train with gradient descent
weights_gd = gradient_descent(train_feature_matrix, train_target_matrix, random_vector, DEFAULT_ALPHA)
# Run model on the validating data
# Report least squares solution on validating dataset
print("Closed Form Solution:")
print(evaluate_model(weights, validating, NUM_TEXT))

# Report gradient descent solution on validating dataset
print("Gradient Descent Solution")
print(evaluate_model(weights_gd, validating, NUM_TEXT))

# Report least squares solution on testing dataset
print("Least squares on testing data set")
print(evaluate_model(weights, testing, NUM_TEXT))
