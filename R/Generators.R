#AUTHOR: Alexandre Galvão Patriota
#IME-USP

Encoder = function(File = File0, Vocabulary = Voc){
  File = unlist(strsplit(File, ""))
  FileX = numeric(length(File))
  for(i in 1:length(Vocabulary)){
    FileX[File == Vocabulary[i]] <- i 
  }
  return(FileX)  
}

Decoder = function(File = File1, Vocabulary = Voc){
  FileX = File
  for(i in 1:length(Vocabulary)){
    FileX[File == i] <- Vocabulary[i]
  }
  return(FileX)  
}

############################################################
#Dataset for training
############################################################

data_GPT <- torch::dataset(
  name = "data_GPT",
  initialize = function(df, block_size, test = FALSE) {
    self$f <- function(s) {
      if (!test) {
        s <- sample(1:(length(df) - block_size - 1), size = 1)
      }
      torch::torch_tensor(df[s:(s + block_size)], dtype = torch::torch_int())
    }

    self$df_len <- length(df) - block_size
    self$test <- test
  },
  .getitem = function(i) {
    z <- self$f(i)
    list(x = z[1:(length(z) - 1)], y = z[2:length(z)])
  },
  .length = function() {
    if (self$test) {
      return(64 * 200)
    }
    if (!self$test) {
      return(64 * 500)
    }
  }
)

############################################################
#Defining the cross entropy loss to use the luz package
############################################################


nn_cross_entropy_loss_0 = torch::nn_module(
  initialize = function(weight = NULL, ignore_index = -1, reduction = "mean"){
      self$D = torch::nn_cross_entropy_loss(weight, ignore_index, reduction)
  },
  forward = function(input, target){
    self$D(input$flatten(end_dim = 2), target$flatten())
  })


Generate = function(idx, Model, block_size , max_new_tokens = 100, temperature=0.7, top_k = 3, device0="cuda"){
	idx = torch::torch_tensor(idx, dtype=torch::torch_int(), device=device0)
	idx = torch::torch_unsqueeze(idx, 1)
	idx0= idx
	torch::with_no_grad({
	for(i in 1:max_new_tokens){
            if(idx$size(2) <= block_size){ 
                idx_cond = idx
	    } else{
		    k1=idx$size(2)-block_size+1; k2 =idx$size(2)
			    idx_cond = idx[,k1:k2]}

            logits = Model$eval()(idx_cond)
            logits = logits[, -1, ] / temperature
            if(!is.null(top_k)){
                v = torch::torch_topk(logits, min(top_k, logits$size(-1)))
                logits[, -v[[2]][1,]] = -Inf
	    }

            probs = torch::nnf_softmax(logits,-1)
            idx_next = torch::torch_multinomial(probs, num_samples=1)
            idx = torch::torch_cat(list(idx, idx_next), 2)
	    }
	idx  = idx$to(device = 'cpu')
	idx  = as.integer(idx)[-c(1:length(idx0))]
	return(idx)
	})
}



