import dataframe
import sqrt_distance_classifier
import abs_distance_classifier
import percent_distance_classifier
import stddev_distance_classifier
import knn
import classnet
import correlation_module

df = dataframe.Dataframe('test-data/drug200.csv')
# (dataframe, input features, training dataset ratio)
model = knn.train(df, ['Na_to_K', 'Cholesterol', 'BP'], 'Drug', 0.5)
# (model, dataframe, output feature)
print(knn.eval(model, df, 'Drug'))

point = {
    'Age': 23,
    'Sex': 'F',
    'BP': 'HIGH',
    'Cholesterol': 'HIGH',
    'Na_to_K': 25.355
}
point_numerical = df.convert_point(point)
# (converted numerical point, output feature, model)
print(knn.predict(point_numerical, 'Drug', model))