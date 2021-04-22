#Import Headers
import os
import wget
import shutil
import tarfile

def download_data_and_unzip(data_directory):
	
	dataset_name = "multi30k"
	data_path = os.path.join(data_directory,dataset_name)
	
	train_url = "http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz"
	test_url = "http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz"
	validation_url = "http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/mmt16_task1_test.tar.gz"
	
	raw_data_path = os.path.join(data_path,"raw_data")
	uncompressed_data_path = os.path.join(data_path,"uncompressed_data")
		
	#shutil.rmtree(data_path, ignore_errors=True)
	if not os.path.exists(data_path):
			
		os.mkdir(data_path)
		os.mkdir(raw_data_path)
		os.mkdir(uncompressed_data_path)
		
		print("\nDownloading Training Data")
		wget.download(train_url,raw_data_path)
		print("\nDownloading Testing Data")
		wget.download(test_url,raw_data_path)
		print("\nDownloading Validation Data")
		wget.download(validation_url,raw_data_path)
		
		print("\nUncompressing the Data")
		file_names = os.listdir(raw_data_path)
		for f in file_names:
			my_tar = tarfile.open(os.path.join(raw_data_path,f))
			my_tar.extractall(uncompressed_data_path)
		print("Done..!!\n")
	
	return uncompressed_data_path
	
	
directory = "./data"
uncompressed_data_path = download_data_and_unzip(directory)

def read_data_return_lists(directory_path):
	
	train_de = [l.strip() for l in open(os.path.join(directory_path,"train.de"), 'r', encoding='utf-8')]
	train_en = [l.strip() for l in open(os.path.join(directory_path,"train.en"), 'r', encoding='utf-8')]
	val_de = [l.strip() for l in open(os.path.join(directory_path,"val.de"), 'r', encoding='utf-8')]
	val_en = [l.strip() for l in open(os.path.join(directory_path,"val.en"), 'r', encoding='utf-8')]
	test_de = [l.strip() for l in open(os.path.join(directory_path,"test.de"), 'r', encoding='utf-8')]
	test_en = [l.strip() for l in open(os.path.join(directory_path,"test.en"), 'r', encoding='utf-8')]
	
	train_de = train_de[:-1]
	train_en = train_en[:-1]
	val_de = val_de[:-1]
	val_en = val_en[:-1]
	
	return train_de, train_en, val_de, val_en, test_de, test_en
	
train_de, train_en, val_de, val_en, test_de, test_en = read_data_return_lists(uncompressed_data_path)

print(len(train_de), len(train_en), len(val_de), len(val_en), len(test_de), len(test_en))

