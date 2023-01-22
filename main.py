import dataframe

df = dataframe.Dataframe('test-data/Stars.csv')
df.make_numerical()
print(df.get_int_map())
df.display()