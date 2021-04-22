from torch.autograd import Variable
import torch

import numpy as np

def loss_backprop(generator, criterion, out, targets, normalize, bp=True):
    """
    Memory optmization. Compute each timestep separately and sum grads.
    """
    assert out.size(1) == targets.size(1)
    total = 0.0
    out_grad = []
    
    list_for_all_words_in_sentences = []
    for i in range(out.size(1)):
        out_column = Variable(out[:, i].data, requires_grad=True)
        gen = generator(out_column)
        list_for_all_words_in_sentences.append(word_index_list_time_step_for_batch(gen))
        loss = criterion(gen, targets[:, i]) / normalize
        total += loss.data
        loss.backward()
        out_grad.append(out_column.grad.data.clone())
    if bp:
        out_grad = torch.stack(out_grad, dim=1)
        out.backward(gradient=out_grad)

  
    predicted_sentences = np.transpose(np.array(list_for_all_words_in_sentences))
    target_sentences = np.array(targets.cpu())

    return total, predicted_sentences
    
def word_index_list_time_step_for_batch(predicted_sentence):
	
	list_time_step_for_batch = []
	for line in predicted_sentence:
		list_time_step_for_batch.append(int(torch.argmax(line)))
	return list_time_step_for_batch
