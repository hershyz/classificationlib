import line

def correlate(dataframe):

    # get outputs:
    feature_map = dataframe.get_feature_map()
    inv_feature_map = {}
    for key in feature_map:
        inv_feature_map[feature_map[key]] = key
    y = dataframe.get_col(len(feature_map) - 1)

    # find % error from regression of each input
    for i in range(len(feature_map) - 1):
        x = dataframe.get_col(i)
        L = line.Line(x, y)
        print(inv_feature_map[i] + ': ' + str(L.eval()))