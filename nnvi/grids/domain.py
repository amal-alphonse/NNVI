class Domain:
    def __init__(self, pdim, left_coord, right_coord):
        self.pdim = pdim
        self.left_coord = left_coord
        self.right_coord = right_coord
    def get_pdim(self):
        return self.pdim
    def get_left_coord(self):
        return self.left_coord
    def get_right_coord(self):
        return self.right_coord    
    def get_volume(self):
        return (self.right_coord-self.left_coord)**self.pdim
