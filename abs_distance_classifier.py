import sqrt_distance_classifier

class abs_distance_classifier_model:

    # this uses the same information as the same as the sqrt distance classifier model, we just need the mean map
    def __init__(self, df, input_features):
        sqrt_distance_classifier_model = sqrt_distance_classifier.train(df, input_features)
        mean_map = sqrt_distance_classifier_model.mean_map
        self.mean_map = mean_map

    # display
    def display(self):
        print('mean map:')
        for output in self.mean_map:
            print(output + ': ' + str(self.mean_map[output]))


# getter for abs distance classifier model
def train(df, input_features):
    model = abs_distance_classifier_model(df, input_features)
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
            dist += abs(float(x1) - float(x2))

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