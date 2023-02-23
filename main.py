import dataframe
import sqrt_distance_classifier

df = dataframe.Dataframe('test-data/drug200.csv')
model = sqrt_distance_classifier.train(df, ['Cholesterol', 'Na_to_K', 'BP'])
model.display()
print(sqrt_distance_classifier.eval(model, df, 'Drug'))