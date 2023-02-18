import dataframe
import sqrt_distance_classifier

df = dataframe.Dataframe('test-data/glass.csv')
df.make_numerical()
model = sqrt_distance_classifier.train(df, df.get_input_feature_labels())

test_points = df.get_test_points()
correct = 0
for point in test_points:
    real = point['Type']
    predicted = sqrt_distance_classifier.predict(point, model)
    if predicted == real:
        correct += 1
print('acc: ' + str(correct / len(test_points)))