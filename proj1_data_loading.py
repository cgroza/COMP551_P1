import json # we need to use the JSON package to load the data, since the data is stored in JSON format
import numpy

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
for info_name, info_value in data_point.items():
    print(info_name + " : " + str(info_value))

# split data set
# training set
training = data[:10000]
# validating set
validating = data[10000:11000]
# testing set
testing = data[11000:12000]

# Returns an ordered list of the 160 most common words
def get_common_words(comments):
    word_counts = {}
    for comment in comments:
        for word in comment.split():
            if word in word_counts:
                word_counts[word] = word_counts[word] + 1
            else:
                word_counts[word] = 1
    # return the first 160 words from large frequency to small frequency
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

def build_feature_matrix(data):
    # gather comment texts
    comments = [d["text"] for d in data]
    # build list of common words to be included as features
    common_words = get_common_words(comments)
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
        # Add them to the row
        features = features + word_counts
        # bias column
        features.append(1)
        # add the row we just built to the matrix
        matrix.append(features)
    # Convert to efficient numpy array
    return numpy.array(matrix)

def least_squares_method(X, Y):
    X_t = numpy.transpose(X)
    X_tX_inv = numpy.linalg.inv(numpy.matmul(X_t, X))
    X_tX_inv_X_t = numpy.matmul(X_tX_inv, X_t)
    w = numpy.matmul(X_tX_inv_X_t, Y)
    return w
