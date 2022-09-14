import sys 
import subprocess #need to run "pip install subprocess.run"
import os 
import pandas as pd #need to run "pip install pandas" and "pip install openpyxl"
    
# path = sys.argv[1]
path = "Data Status.xlsx"

print(path)
if (os.path.exists(path) == False):
    print("no such path.")
    exit()

df = pd.read_excel(path)    
dirs = df['path']
dirs_num = len(dirs)
sizes = []

for i, dir in enumerate(dirs):

    process = subprocess.Popen(["du", "-s", dir], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    size_in_kb = int((process.communicate()[0]).split('/')[0]) 
    size_in_tb = size_in_kb / (1024**3)
    sizes.append(round(size_in_tb, 3))
    print('finished', i+1 , 'out of', dirs_num)

df['size in TB'] = sizes
df.to_excel("Data Status with sizes.xlsx", index= False)
print('finished.') 




