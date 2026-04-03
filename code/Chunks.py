from Global_stuff import *

class Sentence : 
    def __init__(self, index : int, content : str) : 
        self.index = index
        self.content = content
        self.vectorized = None

    def __len__(self) : 
        return len(self.content)

    def __str__(self):
        return self.content

    def __repr__(self):
        return "s__" + self.__str__()

    def select(self) : 
        pass

class Word : 
    def __init__(self, start : int, end : int, content : str) : 
        self.start = start
        self.end = end
        self.content = content
 
    def __len__(self) : 
        return len(self.content)

    def __str__(self):
        return self.content

    def __repr__(self):
        return "w__" + self.__str__() +"_" + str(self.start) 