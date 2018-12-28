# unzip the data set 
import zipfile
import sys, os 

file = input('Please enter the input file name as file_name.zip',)

with zipfile.ZipFile(file, 'r') as zip_ref:
    zip_ref.extractall(os.path.splitext(file)[0])

print('Processing finished !!!')