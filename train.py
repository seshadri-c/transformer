from data_loader import *
from tqdm import tqdm
import numpy as np
import spacy
from make_transformer_model import *
from optimizer import *
from label_smoothing import *
from training_setup import *
from torchtext import data
from make_vocab import *
from bleu import list_bleu
from nltk.tokenize.treebank import TreebankWordDetokenizer
import random
from utils import *
	

def read_sentences(n, src, tgt, out, generator):
	
	source_n = []
	target_n = []
	predicted_n = []
	
	batch_size = src.shape[0]
	for b in range(batch_size):
		source = list(np.array(src[b]))
		source_n.append(source)
		target = list(np.array(tgt[b]))
		target_n.append(target)
		output = out[b]
		predicted = []
		for i in range(output.size(0)):
			out_column = Variable(output[i].data, requires_grad=True)
			gen = generator(out_column)
			predicted.append(int(torch.argmax(gen)))
		predicted_n.append(predicted)
		
		if(b==n-1):
			break
			
	return (source_n, target_n, predicted_n)

		
def train_epoch(epoch, train_loader, model, criterion, model_opt):
	
	model.train()
	progress_bar = tqdm(enumerate(train_loader))
	total_loss = 0.0
	for step, (src, tgt, src_mask, tgt_mask) in progress_bar:
		#print("The shapes are : ",src.shape, tgt[:, :-1].shape, src_mask.shape, tgt_mask[:, :-1, :-1].shape)
		out = model.forward(src.cuda(), tgt[:, :-1].cuda(), src_mask.cuda(), tgt_mask[:, :-1, :-1].cuda())
		#print("Output Shape : ",out.shape)
		ntokens = np.array(tgt[:,:-1]).shape[1]
		loss = loss_backprop(model.generator, criterion, out, tgt[:, 1:].cuda(), ntokens)
		total_loss +=loss
		model_opt.step()
		model_opt.optimizer.zero_grad()
		progress_bar.set_description("Epoch : {} \t Training Loss : {}".format(epoch+1, int(total_loss / (step + 1)))) 
		progress_bar.refresh()
		
	return total_loss/(step+1), model, model_opt			

def valid_epoch(epoch, valid_loader, model, criterion):
	
	model.eval()
	progress_bar = tqdm(enumerate(valid_loader))
	total_loss = 0.0
	total_tokens = 0
	for step, (src, tgt, src_mask, tgt_mask) in progress_bar:
		#print("The shapes are : ",src.shape, tgt[:, :-1].shape, src_mask.shape, tgt_mask[:, :-1, :-1].shape)
		out = model.forward(src.cuda(), tgt[:, :-1].cuda(), src_mask.cuda(), tgt_mask[:, :-1, :-1].cuda())
		#print("Output Shape : ",out.shape)
		ntokens = np.array(tgt[:,:-1]).shape[1]
		loss = loss_backprop(model.generator, criterion, out, tgt[:, 1:].cuda(), ntokens, bp=False)
		total_loss +=loss
		progress_bar.set_description("Epoch : {} \t Validation Loss : {}".format(epoch+1, int(total_loss / (step + 1)))) 
		progress_bar.refresh()
		
		source_n, target_n, predicted_n = read_sentences(3, src, tgt, out,model.generator)
		
		
	return total_loss/(step+1), source_n, target_n, predicted_n
		
def convert_sentences(list_n, dict_int_word):
	
	sentence_list = []
	for s in list_n:
		if(s is not None):
			temp = []
			for t in s:
				if(t==1):
					break
				if(t!=0 and t!=2):
					temp.append(dict_int_word[t])
			x = ' '.join(word for word in temp)
			sentence_list.append(x)
		
	return sentence_list
	
def	training_testing(train_loader, val_loader, model, criterion, model_opt, num_epochs, dict_train_de_word_int, dict_train_de_int_word, dict_train_en_word_int, dict_train_en_int_word):
	
	for epoch in range(num_epochs):
		
		print("\n\nTraining : ")
		print("Starting Epoch No : {}".format(epoch+1))
		total_train_loss, model, model_opt = train_epoch(epoch,train_loader, model, criterion, model_opt)
		print("Epoch No {} completed. Total Training Loss : {}".format(epoch+1,total_train_loss))		
		total_val_loss, source_n, target_n, predicted_n = valid_epoch(epoch,val_loader, model, criterion)
		print("Epoch No {} completed. Total Validation Loss : {}".format(epoch+1,total_val_loss))
		source_sent = convert_sentences(source_n, dict_train_de_int_word)
		target_sent = convert_sentences(target_n, dict_train_en_int_word)
		predicted_sent = convert_sentences(predicted_n, dict_train_en_int_word)
		for i in range(len(source_sent)):
			print("\n\n Pair : ",i+1)
			print("Source Sentence : ",source_sent[i])
			print("Target Sentence : ",target_sent[i])
			print("Predicted Sentence : ",predicted_sent[i])
		print("The BLEU Score is : ",list_bleu(target_sent, predicted_sent))	
		
		if(total_val_loss>total_train_loss):
			print("Training Complete")
			break	
		
def main():
	
	data_path = "./data/multi30k/uncompressed_data"
	train_de, train_en, val_de, val_en, test_de, test_en = read_data_return_lists(data_path)
	
	dict_train_de_word_int, dict_train_de_int_word = make_vocab(train_de, language="de")
	dict_train_en_word_int, dict_train_en_int_word = make_vocab(train_en, language="en")
	
	#Model made only with Train Vocab data	
	model = make_model(len(dict_train_de_word_int.keys()),len(dict_train_en_word_int.keys()), N=6)
	model_opt = get_std_opt(model)
	model.cuda()
	
	#Input is the Target Vocab Size
	criterion = LabelSmoothing(size=len(dict_train_en_word_int.keys()), padding_idx=2, smoothing=0.1)
	criterion.cuda()
	
	train_loader = load_data(data_path+"/train", batch_size=128, num_workers=10, shuffle=True)
	val_loader = load_data(data_path+"/val", batch_size=128, num_workers=10, shuffle=True)
	
	num_epochs = 10
	training_testing(train_loader, val_loader, model, criterion, model_opt, num_epochs, dict_train_de_word_int, dict_train_de_int_word, dict_train_en_word_int, dict_train_en_int_word)	
main()
