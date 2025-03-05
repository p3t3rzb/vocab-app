import pandas as pd

class VocabularyLoader:
    def __init__(self,name):
        self.name = name
    
    def load(self):
        package = pd.read_excel("wordlists/" + self.name + ".xlsx",header=None).dropna(how='all')
        return list(zip(package.iloc[:, 0], package.iloc[:, 1]))
