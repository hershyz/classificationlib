import sqrt_distance_classifier

class percent_distance_classifier_model:

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


# getter for percent distance classifier model
def train(df, input_features):
    model = percent_distance_classifier_model(df, input_features)
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

            if x1 == x2:
                dist += 0
            elif x1 == 0 or x2 == 0:
                dist += abs(2*(x1 - x2) / (x1 + x2))
            else:
                dist += abs((x1 - x2) / x1)
        
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