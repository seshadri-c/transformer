
#Import Headers
from make_data_ready import *
import spacy

train_de, train_en, val_de, val_en, test_de, test_en = read_data_return_lists(uncompressed_data_path)

#Loading German 
spacy_de = spacy.load("de_core_news_sm")
#Loading English
spacy_en = spacy.load("en_core_web_sm")

#Tokenizer for German
def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]
#Tokenizer for English
def tokenize_en(text):
	return [tok.text for tok in spacy_en.tokenizer(text)]
	
def make_vocab(sentences_list,language):
	
	#Beginning of Sentence
	BOS_WORD = '<s>'
	#End of Sentence
	EOS_WORD = '</s>'
	#Padding
	BLANK_WORD = "<blank>"
	
	dict_word_int = {}
	#Creating a Dictionary in the format {Token : Integer}
	i=0
	dict_word_int.update({BOS_WORD:i})
	i+=1
	dict_word_int.update({EOS_WORD:i})
	i+=1
	dict_word_int.update({BLANK_WORD:i})
	i+=1
	
	for s in sentences_list:
		if(language=="de"):
			token_list = tokenize_de(s)
		if(language=="en"):
			token_list = tokenize_en(s)
		for token in token_list:
			if token not in dict_word_int.keys():
				dict_word_int.update({token:i})
				i+=1			
	
	#Reversing the Dictionary in the format {Integer : Token}
	dict_int_word = {value : key for (key, value) in dict_word_int.items()}

	return dict_word_int, dict_int_word
	
dict_train_de_word_int, dict_de_int_word = make_vocab(train_de, language="de")
dict_train_en_word_int, dict_en_int_word = make_vocab(train_en, language="en")

#print(dict_train_de_word_int, dict_de_int_word,dict_train_en_word_int, dict_en_int_word)
