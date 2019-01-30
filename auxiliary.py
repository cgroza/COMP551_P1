def run_gd():
    random_vector = numpy.random.rand(num_features)
    weights_gd = gradient_descent(train_feature_matrix, train_target_matrix, random_vector, DEFAULT_ALPHA)
    print(evaluate_model(weights_gd, validating, 0))
    print(weights_gd)
