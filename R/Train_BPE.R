#AUTHOR: Alexandre Galvão Patriota
#IME-USP

############################################################
#Finding the tokens of the vocabulary. 
#It will save a file youtokentome.bpe in your folder
############################################################
if(Train_tokens){
  model     <- bpe(fileName, coverage = 0.999, vocab_size = nvoc0)
}

if(!Train_tokens){
  model     <- bpe_load_model("youtokentome.bpe")
}

############################################################
#Encoding Shakespeare into token ids
############################################################
File      <- readChar(fileName, file.info(fileName)$size)
Encoded   <- bpe_encode(model, x = File, type = "ids")[[1]]
nvoc0     <- length(model$vocabulary[,2])


############################################################
#Defining train data and test data
############################################################
set.seed(1)
prop <- 0.9
n    <- round(length(Encoded)*prop)
train_data <- Encoded[1:n]
test_data  <- Encoded[(n+1):length(Encoded)]
BD.train   <- data_GPT(train_data, block_size0, test=FALSE)
BD.test    <- data_GPT(test_data,  block_size0, test=TRUE)

############################################################
#Setting the model
############################################################

torch_manual_seed(42)
lr0   <-  0.0003/2

Model <- GPT %>%
  setup(loss = nn_cross_entropy_loss_0(), optimizer = optim_adam) %>%
  set_hparams(block_size = block_size0, ncol = ncol0 ,N_Layers = N_Layers0,  nvoc = nvoc0,Head=Head0, p0 = p00) %>%
  set_opt_hparams(lr = lr0)

############################################################
#Training the model. 
#Change the number of workers.
############################################################

clip  <- luz_callback_gradient_clip(max_norm = 1, norm_type = 2)

fitted  <- fit(Model, 
	       BD.train, 
	       epochs = epochs0, 
	       accelerator = luz::accelerator(), 
	       valid_data = BD.test, 
	       dataloader_options = list(batch_size = batch_size0, num_workers = Workers0, shuffle = FALSE),
	       callbacks = list(clip))

luz_save_model_weights(fitted, "Model-Shakes_weights.pt")

