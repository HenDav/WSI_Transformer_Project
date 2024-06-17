import torch

def R_set(x):
	'''Create an indicator matrix of risk sets, where T_j >= T_i.
	Note that the input data have been sorted in descending order.
	Input:
		x: a PyTorch tensor that the number of rows is equal to the number of samples.
	Output:
		indicator_matrix: an indicator matrix (which is a lower traiangular portions of matrix).
	'''
	n_sample = x.size(0)
	matrix_ones = torch.ones(n_sample, n_sample)
	indicator_matrix = torch.tril(matrix_ones)

	return(indicator_matrix)

def c_index(pred, y):
	'''Calculate concordance index to evaluate models.
	From https://github.com/tomcat123a/survival_loss_criteria/
	Input:
		pred: linear predictors from trained model.
		ytime: true survival time.
		yevent: true censoring status. Here 1 is noncensored.
	Output:
		concordance_index: c-index (between 0 and 1).
	'''
	ytime, yevent, _ = y.T
	n_sample = len(ytime)
	ytime_indicator = R_set(ytime)
	ytime_matrix = ytime_indicator - torch.diag(torch.diag(ytime_indicator))
	###T_i is uncensored
	censor_idx = yevent.nonzero()
	zeros = torch.zeros(n_sample)
	ytime_matrix[censor_idx, :] = zeros
	###1 if pred_i < pred_j; 0.5 if pred_i = pred_j
	pred_matrix = torch.zeros_like(ytime_matrix)
	for j in range(n_sample):
		for i in range(n_sample):
			if pred[i] < pred[j]:
				pred_matrix[j, i]  = 1
			elif pred[i] == pred[j]: 
				pred_matrix[j, i] = 0.5
	
	concord_matrix = pred_matrix.mul(ytime_matrix)
	###numerator
	concord = torch.sum(concord_matrix)
	###denominator
	epsilon = torch.sum(ytime_matrix)
	###c-index = numerator/denominator
	concordance_index = torch.div(concord, epsilon)
	###if gpu is being used
	if torch.cuda.is_available():
		concordance_index = concordance_index.cuda()
	###
	return(concordance_index)