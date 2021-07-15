'''
    
    Download HBN data from Amazon AWS, where the database is stored
    It takes on average 5 ~ 6 minutes to download 1 file.
    On average, a file is of size 150MB
    
'''



import pandas as pd
import os
import wget
import csv


#good_subject refers to the ones that are successfully found on AWS
#bad_subject refers to the ones that are not found on AWS
good_subjects=[]
bad_subjects=[]

with open('MDD_indi.csv','rt')as f:
  data = csv.reader(f)
  for each_subject in data:
      try:
          print('Beginning download of subject:',each_subject[0])
          url_aws='https://fcp-indi.s3.amazonaws.com/data/Projects/HBN/EEG/'
        
          folder_to_save='HBN_down/' 
          #files are stored in a folder named HBN_down

#        url_1=url_aws+each_subject[0]+"/EEG/preprocessed/csv_format/RestingState_chanlocs.csv"
#        wget.download(url_1, folder_to_save + each_subject[0] + "RestingState_chanlocs.csv")
#            
          url_2=url_aws+each_subject[0]+"/EEG/preprocessed/csv_format/RestingState_data.csv"
          print("url_2",url_2)
          wget.download(url_2, folder_to_save + each_subject[0] +"RestingState_data.csv")
#
#        url_3=url_aws+each_subject+"/EEG/preprocessed/csv_format/RestingState_event.csv"
#        wget.download(url_3, folder_to_save+"/RestingState_event.csv")

#Channel locations and the Event file are only downloaded once, since it is the same for all files

          good_subjects.append(each_subject[0])
          print('Finished correctly download of subject:',each_subject[0])
      except:
          bad_subjects.append(each_subject)
        

