import sqrt_distance_classifier
import abs_distance_classifier
import percent_distance_classifier
import stddev_distance_classifier
import knn
import classnet


def run(df):

    # get input features and output feature:
    input_features = df.get_input_feature_labels()
    feature_labels = df.get_feature_labels()
    output = ''
    for feature in feature_labels:
        if feature not in input_features:
            output = feature
            break
    
    # relative significance for each classifier
    classifier_weights = {}
    classifier_weights['sqrt'] = 1
    classifier_weights['abs'] = 1
    classifier_weights['percent'] = 1
    classifier_weights['stddev'] = 1
    classifier_weights['knn'] = 2
    classifier_weights['classnet'] = 0.5
    division_factor = 0
    for key in classifier_weights:
        division_factor += classifier_weights[key]

    # calculate feature accuracies:
    feature_accs = []
    for feature in input_features:
        
        acc = 0
        sqrt_model = sqrt_distance_classifier.train(df, [feature])
        abs_model = abs_distance_classifier.train(df, [feature])
        percent_model = percent_distance_classifier.train(df, [feature])
        stddev_model = stddev_distance_classifier.train(df, [feature])
        knn_model = knn.train(df, [feature], output, 1)
        classnet_model = classnet.train(df, [feature])

        acc += sqrt_distance_classifier.eval(sqrt_model, df, output) * classifier_weights['sqrt']
        acc += abs_distance_classifier.eval(abs_model, df, output) * classifier_weights['abs']
        acc += percent_distance_classifier.eval(percent_model, df, output) * classifier_weights['percent']
        acc += stddev_distance_classifier.eval(stddev_model, df, output) * classifier_weights['stddev']
        acc += knn.eval(knn_model, df, output) * classifier_weights['knn']
        acc += classnet.eval(classnet_model, df, output) * classifier_weights['classnet']
        acc /= division_factor
        feature_accs.append([feature, acc])

        # TODO: display sorted key column:
    
    print(feature_accs)