import os
from Kirby.Enums import Hospital, Staining 
import Slide
import validationChecks as vChecks
import Enums




class Kirby: 

    def __init__(self, dir_path: str, hospital : Hospital, staining : Staining, dateset : Dateset , batch : int):

        try: 

            vChecks.check_arguments(dir_path , hospital, staining, dateset, batch)

        except:
            print("An exception occurred")
            



        

        self.dir_path = dir_path
        self.hospital = hospital
        self.staining = staining
        self.dataset = dateset 
        self.batch = batch
    
    def readSlides(self):









