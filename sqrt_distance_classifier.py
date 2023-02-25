import math

class sqrt_distance_classifier_model:
    
    # constructor
    def __init__(self, df, input_features) -> None:
        
        # get unique outputs:
        outputs = set()
        point_arr = df.get_point_arr()
        output_row = df.get_col(len(point_arr[0]) - 1)
        for y in output_row:
            if y not in outputs:
                outputs.add(y)
        
        # calculate the average of input features, parameterized by output
        feature_map = df.get_feature_map()
        mean_map = {}
        for output in outputs:
            
            curr_averages = {}
            for feature in input_features:
                curr_averages[feature] = 0
            
            freq = 0
            for row in point_arr:
                if row[len(point_arr[0]) - 1] == output:
                    freq += 1
                    for feature in input_features:
                        curr_averages[feature] += float(row[feature_map[feature]])
            
            for feature in curr_averages:
                curr_averages[feature] /= freq
            
            mean_map[output] = curr_averages
        
        self.mean_map = mean_map
    
    # display
    def display(self):
        print('mean map:')
        for output in self.mean_map:
            print(output + ': ' + str(self.mean_map[output]))


# getter for sqrt distance classifier model
def train(df, input_features):
    model = sqrt_distance_classifier_model(df, input_features)
    return model


# prediction function
def predict(point_map, model):
    
    min_dist = float('inf')
    min_cat = ''

    for cat in model.mean_map:
        
        avg_point = model.mean_map[cat]
        dist = 0
        
        for feature in avg_point:
            x1 = avg_point[feature]
            x2 = point_map[feature]
            dist += ((float(x1) - float(x2)) ** 2)
        dist = math.sqrt(dist)

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