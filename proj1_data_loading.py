import json # we need to use the JSON package to load the data, since the data is stored in JSON format

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
