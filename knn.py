import random
import math

class knn_model:

    def __init__(self, df, input_features, output, ratio):

        self.input_features = input_features
        training_points = df.get_test_points()
        sampled_training_points = []

        indices = set()
        while (len(indices) / len(training_points)) < ratio:
            n = random.randrange(len(training_points))
            if n not in indices:
                indices.add(n)
        
        for index in indices:
            point_raw = training_points[index]
            point = {}
            for feature in input_features:
                point[feature] = point_raw[feature]
            point[output] = point_raw[output]
            sampled_training_points.append(point)
        
        self.sampled_training_points = sampled_training_points

    def display(self):
        print('sampled training points:')
        for point in self.sampled_training_points:
            print(point)


# getter for knn model
def train(df, input_features, output, ratio):
    model = knn_model(df, input_features, output, ratio)
    return model


# prediction function
def predict(point_map, output, model):

    points = model.sampled_training_points
    dists = []

    for sample_point in points:
        y = sample_point[output]
        dist = 0
        for feature in sample_point:
            if feature != output:
                x1 = float(sample_point[feature])
                x2 = float(point_map[feature])
                dist += abs((x1 - x2) ** 2)
        dist = math.sqrt(dist)
        dists.append([y, dist])
    
    dists = sorted(dists, key=lambda x: x[1])
    freqs = {}

    for i in range(5): # k = 5, 5 closest elements
        y = dists[i][0]
        if y in freqs:
            freqs[y] = freqs[y] + 1
        if y not in freqs:
            freqs[y] = 1
    
    max = 0
    max_cat = ''
    for cat in freqs:
        if freqs[cat] > max:
            max = freqs[cat]
            max_cat = cat
    
    return max_cat


# evaluate model accuracy
def eval(model, df, output):

    test_points = df.get_test_points()
    correct = 0

    for point in test_points:
        real = point[output]
        predicted = predict(point, output, model)
        if predicted == real:
            correct += 1
    
    return correct / len(test_points)