class classnet_model:

    def mean(self, arr):
        return sum(arr) / len(arr)

    # constructor:
    def __init__(self, df, input_features):

        # get unique outputs:
        outputs = set()
        point_arr = df.get_point_arr()
        output_row = df.get_col(len(point_arr[0]) - 1)
        for y in output_row:
            if y not in outputs:
                outputs.add(y)

        # calculate the mean numerical feature value:
        feature_map = df.get_feature_map()
        feature_mean = 0
        seen_features = 0

        for feature in input_features:
            col = df.get_col(feature_map[feature])
            for n in col:
                feature_mean += float(n)
                seen_features += 1
        
        feature_mean /= seen_features

        # determine output labels:
        step = feature_mean * 10
        label = step
        output_labels = {}
        for output in outputs:
            output_labels[output] = label
            label += step
        
        # calculate init weights per input feature -- TODO:
        feature_weights = {}
        feature_map = df.get_feature_map()
        output_col = df.get_col(len(feature_map) - 1)
        for i in range(len(output_col)):
            output_col[i] = output_labels[output_col[i]]
        
        print(self.mean(output_col)) # BUG: WHY DOES THIS KEEP RETURNING DIFFERENT RESULTS, THE ARRAY IS THE SAME EACH TIME?!?!!?!?

        # calculate baseline error per input feature -- TODO:

        # proportionally adjust weights depth first until error eaches min -- TODO:

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