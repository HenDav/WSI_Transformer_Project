from enum import Enum, auto

class Hospital(Enum):
    Carmel = auto()
    Haemek = auto()
    Sheba = auto()
    #add more here

class Staining(Enum):
    H_and_E = auto()
    IHC = auto()
    Frozen = auto()
    #add more here

class Dateset(Enum):
    general = auto() #for dataset such as carmel 1-11 
    OncoType = auto()
    Her2 = auto()
    benign = auto()
    #add more here
