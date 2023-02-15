import dataframe
import sqrt_distance_classifier

df = dataframe.Dataframe('test-data/stars.csv')
df.make_numerical()
model = sqrt_distance_classifier.get_model(df, df.get_input_feature_labels())
model.display()