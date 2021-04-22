from torch.utils.data import Dataset, DataLoader
import os
import random
from make_vocab import *

class DataGenerator(Dataset):
	
	def __init__(self, path):
		self.files = self.get_files(path)
        

	def __len__(self):
		return len(self.files)
        

	def __getitem__(self,idx):

		length = self.__len__()
		while(1):
		
			idx = random.randint(0, length-1)
			f_de, f_en = self.files[idx]
			#print("Single German Sentences : ",f_de)
			#print("Single English Sentences : ",f_en,"\n")
			return f_de, f_en
			
	def get_files(self,path):

		data_de = [l.strip() for l in open(path + ".de", 'r', encoding='utf-8')]
		data_en = [l.strip() for l in open(path + ".en", 'r', encoding='utf-8')]
								
		#List to contain all the sentence pairs in tuples.
		files = []
		for i in range(len(data_de)-1):
			files.append((data_de[i],data_en[i]))
		
		return files		
		
def load_data(data_path, batch_size=128, num_workers=10, shuffle=True):
    
    dataset = DataGenerator(data_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
	
    return data_loader

#data_path = "./data/multi30k/uncompressed_data"
#load_data(data_path)
