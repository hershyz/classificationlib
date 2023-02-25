import sqrt_distance_classifier

class stddev_distance_classifier_model:

    def __init__(self, df, input_features): 
        
        # get mean map
        sqrt_distance_classifier_model = sqrt_distance_classifier.train(df, input_features)
        mean_map = sqrt_distance_classifier_model.mean_map
        self.mean_map = mean_map

        # calculate standard deviation per input feature
        feature_map = df.get_feature_map()
        stddev_map = {}
        for feature in input_features:
            
            feature_index = feature_map[feature]
            col_arr = df.get_col(feature_index)

            # calculate mean of col arr
            mean = 0
            for n in col_arr:
                mean += float(n)
            mean /= len(col_arr)

            # calculate average diff from mean
            avg_diff = 0
            for n in col_arr:
                avg_diff += abs(float(n) - mean)
            avg_diff /= len(col_arr)

            stddev_map[feature] = avg_diff
        
        self.stddev_map = stddev_map

    # display
    def display(self):
        print('mean map:')
        print(self.mean_map)
        print('stddev map:')
        print(self.stddev_map)
    

# getter for stddev distance classifier model
def train(df, input_features):
    model = stddev_distance_classifier_model(df, input_features)
    return model


# prediction function
def predict(point_map, model):

    min_dist = float('inf')
    min_cat = ''

    for cat in model.mean_map:
        
        avg_point = model.mean_map[cat]
        dist = 0

        for feature in avg_point:
            x1 = float(avg_point[feature])
            x2 = float(point_map[feature])
            stddev = float(model.stddev_map[feature])
            dist += abs((x1 - x2) / stddev)
        
        if dist < min_dist:
            min_dist = dist
            min_cat = cat
    
    return min_cat


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