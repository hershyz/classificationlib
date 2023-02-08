import dataframe

df = dataframe.Dataframe('test-data/gender.csv')
df.make_numerical()
df.display()