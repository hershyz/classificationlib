class Dataframe:
    
    # constructor
    def __init__(self, path) -> None:
        
        # read raw file
        f = open(path)
        raw = f.readlines()

        lines = []
        # take newlines out of initial array
        for i in range(len(raw)):
            lines.append(raw[i].replace('\n', ''))
        
        # extract features, map to respective index
        features = lines[0].split(',')
        feature_map = {}
        for i in range(len(features)):
            feature_map[features[i]] = i
        self.feature_map = feature_map

        # extract dataframe into point array
        point_arr = []
        for i in range(1, len(lines)):
            point_arr.append(lines[i].split(','))
        self.point_arr = point_arr
    
    # display vanilla datastructures
    def display(self):
        print(self.feature_map)
        print(self.point_arr)