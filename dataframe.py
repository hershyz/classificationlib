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
        print('features: ' + str(self.feature_map))
        print(self.point_arr)
    
    # make categorical input features numerical
    def make_numerical(self):
        
        # generate int map
        n = 0
        int_map = {}
        for col in range(len(self.feature_map)):
            for row in range(len(self.point_arr)):
                feature = self.point_arr[row][col]
                try:
                    float(feature)
                except: # if float conversion fails, the feature is a string
                    if feature not in int_map:
                        int_map[feature] = n
                        n += 1
        self.int_map = int_map

        # convert features to ints based on int map
        for row in range(len(self.point_arr)):
            for col in range(len(self.point_arr[row])):
                if self.point_arr[row][col] in self.int_map:
                    self.point_arr[row][col] = self.int_map[self.point_arr[row][col]]
    
    # get a specific column
    def get_col(self, n):
        res = []
        for i in range(len(self.point_arr)):
            res.append(float(self.point_arr[i][n]))
        return res

    # getter for int map, must be called after make_numerical()
    def get_int_map(self):
        return self.int_map

    # getter for feature map
    def get_feature_map(self):
        return self.feature_map

    # getter for point arr    
    def get_point_arr(self):
        return self.point_arr
    
    # return raw data points with only the selected features
    def get_condensed_point_arr(self, features):
        res = []
        for i in range(len(self.point_arr)):
            row = []
            for feature in features:
                row.append(self.point_arr[i][self.feature_map[feature]])
            res.append(row)
        return res