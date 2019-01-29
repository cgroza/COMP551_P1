import json  # we need to use the JSON package to load the data, since the data is stored in JSON format
import numpy
#from google.cloud import language
#from google.cloud.language import enums
#from google.cloud.language import types
import math
import time

with open("proj1_data.json") as fp:
    data = json.load(fp)

# Now the data is loaded.
# It a list of data points, where each datapoint is a dictionary with the following attributes:
# popularity_score : a popularity score for this comment (based on the number of upvotes) (type: float)
# children : the number of replies to this comment (type: int)
# text : the text of this comment (type: string)
# controversiality : a score for how "controversial" this comment is (automatically computed by Reddit)
# is_root : if True, then this comment is a direct reply to a post; if False, this is a direct reply to another comment

# Example:
data_point = data[0] # select the first data point in the dataset

# Now we print all the information about this datapoint
# for info_name, info_value in data_point.items():
#    print(info_name + " : " + str(info_value))

# split data set
# training set
training = data[:10000]
# validating set
validating = data[10000:11000]
# testing set
testing = data[11000:12000]


def preprocess_words(comments):
    # Strings are immutable in Python
    return [comment.lower() for comment in comments]


# Returns an ordered list of the 160 most common words
def get_common_words(comments):
    word_counts = {}
    for comment in comments:
        for word in comment.split():
            if word in word_counts:
                word_counts[word] = word_counts[word] + 1
            else:
                word_counts[word] = 1
    return [w[0] for w in reversed(sorted(word_counts.items(), key= lambda kv: kv[1]))][:160]


# Counts the occurrence of word features in a comment.
# Returns a list of counts in the same order as words.
def count_word_features(featured_words, comment):
    feature_counts = {}
    for word in featured_words:
        feature_counts[word] = 0
    for word in comment.split():
        if word in feature_counts:
            feature_counts[word] = feature_counts[word] + 1
    return [feature_counts[w] for w in featured_words]


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
def build_feature_matrix(data):
    # gather comment texts
    comments = preprocess_words([d["text"] for d in data])
    # build list of common words to be included as features
    common_words = get_common_words(comments)
    # print(common_words)
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
        word_value = numpy.linalg.norm(word_counts)
        # print(common_words)
        # print(word_counts)
        # features.append(word_value)
        # features.append(comment["children"]**2)
        # word_value = 0
        # TODO: norm or linear equation with exponential decay as wights
        # for word_count in word_counts:
        #     word_value = numpy.linalg.norm(word_count)
        # Add them to the row
        # features.append(word_value)
        # Comment length
        # NOTE: I think we should transform this feature
        # somehow. Both log and sqrt transforms do a tiny bit better. Maybe a
        # Z-score?
        features.append(count_word_length(comment["text"]))
        features.append(comment["children"] * count_word_length(comment["text"]))
        # if("!" in comment["text"]):
        #     features.append(1)
        # else:
        #     features.append(0)
        # bias column
        features.append(1)
        # add the row we just built to the matrix
        matrix.append(features)
    # Convert to efficient numpy array
    return numpy.array(matrix)

DEFAULT_ALPHA = 0.0000000005

def gradient_descent(x, y, alpha):
    print(alpha)
    num_features = x.shape[1]
    w = numpy.random.rand(num_features)
    x_t = numpy.transpose(x)
    epsilon = 0.000001
    count = 2
    a = numpy.matmul(x_t, x)
    b = numpy.matmul(x_t, y)
    while True:
        gradient = 2 * numpy.subtract(numpy.matmul(a, w), b)
        w_0 = w
        # alpha = alpha * math.log2(count)
        w = numpy.subtract(w_0, alpha * gradient)
        theta = numpy.subtract(w, w_0)
        # print("theta", theta)
        '''print("theta_shape", theta.shape)
        print("theta_norm", numpy.linalg.norm(theta))'''
        if numpy.linalg.norm(theta) < epsilon:
            break
        else:
            pass
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


def mean_squared_error(observed, predicted):
     error_ss = sum((predicted - observed)**2)
     return error_ss/len(observed)


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

def time_least_squares():
    time_function("LEAST_SQUARES", lambda: least_squares_method(build_feature_matrix(training), build_target_vector(training)), 20, "least_squares_time.csv")

def time_gradient_descent(alpha, alpha_str):
    print("Timing descentt with alpha = " + str(alpha))
    time_function("GRADIENT_DESCENT_"+alpha_str, lambda: gradient_descent(build_feature_matrix(training), build_target_vector(training), alpha), 20, alpha_str+"gradient_descent.csv")

def evaluate_model(weights, data):
    # Run model on the data data
    predicted = apply_regression(weights, build_feature_matrix(data))
    # Report R^2 of the model
    return {"R^2" : r_squared(build_target_vector(data), predicted),
            "MSE" : mean_squared_error(build_target_vector(data), predicted)}

# Example of function timing
#time_least_squares()
#time_gradient_descent(DEFAULT_ALPHA, "DEF")
#time_gradient_descent(DEFAULT_ALPHA/4, "QUART")
#time_gradient_descent(DEFAULT_ALPHA/8, "EIGTH")
#time_gradient_descent(DEFAULT_ALPHA/16, "SIXTEENTH")

# Here is an example run with the least squared method
# Train the model on the training data
train_feature_matrix = build_feature_matrix(training)
train_target_matrix = build_target_vector(training)
# print("w: ", random_vector)
weights = least_squares_method(train_feature_matrix, train_target_matrix)
# print(train_target_matrix.shape)
weights_gd = gradient_descent(train_feature_matrix, train_target_matrix, DEFAULT_ALPHA)
# Run model on the validating data
print("closed form solution")
print(evaluate_model(weights, validating))

print("gradient descent solution")
# Report R^2 of the model
# Report of GD algorithm
print(evaluate_model(weights_gd, validating))
