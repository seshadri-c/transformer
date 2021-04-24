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

	
def calculate_bleu_scores(src, tgt, pred, dict_src_int_word, dict_tgt_int_word):
	
	#SOURCE
	src_sent_batch = []
	for i in range(src.shape[0]):
		src_int_s = src[i]
		src_word_s = []
		for num in src_int_s:
			if not (num == 0 or num ==1 or num ==2):
				src_word_s.append(dict_src_int_word[num])
		src_sent_batch.append(TreebankWordDetokenizer().detokenize(src_word_s))	
	
	#TARGET	
	tgt_sent_batch = []
	for i in range(tgt.shape[0]):
		tgt_int_s = tgt[i]
		tgt_word_s = []
		for num in tgt_int_s:
			if not (num == 0 or num ==1 or num ==2):
				tgt_word_s.append(dict_tgt_int_word[num])
		tgt_sent_batch.append(TreebankWordDetokenizer().detokenize(tgt_word_s))	
		
	#PREDICTED
	pred_sent_batch = []
	for i in range(pred.shape[0]):
		pred_int_s = pred[i]
		pred_word_s = []
		for num in pred_int_s:
			try:
				if not (num == 0 or num ==1 or num ==2):
					pred_word_s.append(dict_tgt_int_word[num])
			except:
				continue
		pred_sent_batch.append(TreebankWordDetokenizer().detokenize(pred_word_s))
		
	#for i in range(len(src_sent_batch)):
		#print("\n\n\nSource Sentence : {}\nTarget Sentence : {}\nPredicted Sentence : {}".format(src_sent_batch[i],tgt_sent_batch[i],pred_sent_batch[i]))
	
	bleu_score = list_bleu(tgt_sent_batch, pred_sent_batch)
	return bleu_score, src_sent_batch, tgt_sent_batch, pred_sent_batch
	
def print_random_n_triplets(src_sent_batch, tgt_sent_batch, pred_sent_batch,n):
	list_random_numbers = random.sample(range(0, len(src_sent_batch)), n)
	print("\n\nThe Random {} triplets are : ".format(n))
	i=1
	for r in list_random_numbers :
		print("\n{}. Source : {}\nTarget : {}\nPredicted : {}\n\n".format(i, src_sent_batch[r], tgt_sent_batch[r], pred_sent_batch[r]))
		i+=1
		
def train_epoch(epoch, train_loader, model, criterion, model_opt, dict_train_de_word_int, dict_train_de_int_word, dict_train_en_word_int, dict_train_en_int_word):
	
	model.train()
	progress_bar = tqdm(enumerate(train_loader))
	total_loss = 0.0
	
	for step, (src, tgt, src_mask, tgt_mask) in progress_bar:
		out = model.forward(src.cuda(), tgt[:, :-1].cuda(), src_mask.cuda(), tgt_mask[:, :-1, :-1].cuda())
		ntokens = np.array(tgt[:,:-1]).shape[1]
		loss = loss_backprop(model.generator, criterion, out, tgt[:, 1:].cuda(), ntokens, step)
		total_loss +=loss
		model_opt.step()
		model_opt.optimizer.zero_grad()
		progress_bar.set_description("Epoch : {} \t Training Loss : {}".format(epoch+1, int(total_loss / (step + 1)))) 
		progress_bar.refresh()
	return total_loss/(step+1), model, model_opt			
    	
def	training_testing(train_loader, model, criterion, model_opt, num_epochs, dict_train_de_word_int, dict_train_de_int_word, dict_train_en_word_int, dict_train_en_int_word):
	
	data_path = "./data/multi30k/uncompressed_data"
	train_de, train_en, val_de, val_en, test_de, test_en = read_data_return_lists(data_path)
	

	for epoch in range(num_epochs):
		
		print("\n\nTraining : ")
		print("Starting Epoch No : {}".format(epoch+1))
		total_train_loss, model, model_opt = train_epoch(epoch,train_loader, model, criterion, model_opt, dict_train_de_word_int, dict_train_de_int_word, dict_train_en_word_int, dict_train_en_int_word)
		print("Epoch No {} completed. Total Training Loss : {}".format(epoch+1,total_train_loss))

		
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
	
	train_loader = load_data(data_path+"/train", batch_size=5, num_workers=2, shuffle=True)
	
	num_epochs = 10
	training_testing(train_loader, model, criterion, model_opt, num_epochs, dict_train_de_word_int, dict_train_de_int_word, dict_train_en_word_int, dict_train_en_int_word)	
main()
