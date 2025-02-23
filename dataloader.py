class DataLoader:
    def __init__(self, data, curr=0):
        self.data = data
        self.current = curr
        
    def next_batch(self):
        x, y, path = self.data[self.current]
            
        self.current += 1
        if self.current > len(self.data):
            self.current = 0
        return x, y, path