class classnet_model:

    def __init__(self, df, input_features):

        # find distinct outputs:
        feature_map = df.get_feature_map()
        output_col = df.get_col(len(feature_map) - 1)
        outputs = []
        for element in output_col:
            if element not in outputs:
                outputs.append(element)

        feature_avg = 0
        n = 0
        for i in range(len(feature_map) - 1):
            col = df.get_col(i)
            for element in col:
                feature_avg += float(element)
                n += 1
        feature_avg /= n
        curr_label = feature_avg * 10
        label_step = feature_avg * 10

        output_labels = {}
        for output in outputs:
            output_labels[output] = curr_label
            curr_label += label_step

        # transform output col to numerical values:
        for i in range(len(output_col)):
            output_col[i] = output_labels[output_col[i]]

        # calculate init weights for each input feature:
        feature_weights = {}
        for feature in input_features:
            col = df.get_col(feature_map[feature])
            weight = 0
            n = 0
            for i in range(len(col)):
                if float(col[i]) != 0:
                    weight += output_col[i] / float(col[i])
                    n += 1
            weight /= n
            feature_weights[feature] = weight
        
        # calculate baseline percent error for each input feature:
        percent_error_map = {}
        for feature in feature_weights:
            error = 0
            col = df.get_col(feature_map[feature])
            for i in range(len(col)):
                real = output_col[i]
                predicted = float(col[i]) * feature_weights[feature]
                error += ((predicted - real) / real)
            error /= len(col)
            percent_error_map[feature] = error

        # proportionally adjust weights until error min is achieved:
        for feature in feature_weights:
            
            col = df.get_col(feature_map[feature])
            while True:

                weight = feature_weights[feature] - (percent_error_map[feature] * feature_weights[feature] * 0.1)
                error = 0
                for i in range(len(col)):
                    real = output_col[i]
                    predicted = float(col[i]) * weight
                    error += ((predicted - real) / real)
                error /= len(col)

                if abs(error) < abs(percent_error_map[feature]):
                    feature_weights[feature] = weight
                    percent_error_map[feature] = error
                    break
                else:
                    feature_weights[feature] = weight
                    percent_error_map[feature] = error
        
        self.feature_weights = feature_weights
        self.output_labels = output_labels


# getter for classnet model:
def train(df, input_features):
    model = classnet_model(df, input_features)
    return model


# prediction function:
def predict(point_map, model):
    
    prediction_raw = 0
    feature_weights = model.feature_weights
    output_labels = model.output_labels

    for feature in feature_weights:
        prediction_raw += float(point_map[feature]) * float(feature_weights[feature])
    prediction_raw /= len(feature_weights)

    min_dist = float('inf')
    min_output = ''
    for output in output_labels:
        diff = abs(prediction_raw - output_labels[output])
        if diff < min_dist:
            min_dist = diff
            min_output = output
    
    return min_output


# evaluate model accuracy
def eval(model, df, output):

    test_points = df.get_test_points()
    correct = 0

    for point in test_points:
        real = point[output]
        predicted = predict(point, model)
        if predicted == real:
            correct += 1
    
    return correct / len(test_points)