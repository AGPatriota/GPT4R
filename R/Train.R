# AUTHOR: Alexandre Galvão Patriota
# IME-USP

library(magrittr)

source("R/Generators.R")
source("R/GPT.R")
source("R/aux.R")
source("config.R")


############################################################
#Encoding Shakespeare into token ids
############################################################
if(!config$BPE) {
	File <- base::readChar(config$file_name, file.info(config$file_name)$size)
	Voc = c("<PAD>",sort(unique(unlist(strsplit(File, "")))))
	Encoded <- Encoder(File = File, Vocabulary = Voc)		
}

if(config$BPE) {
	if (config$Train_tokens) {
		model <- tokenizers.bpe::bpe(config$file_name,
		coverage = config$coverage,
		vocab_size = config$vocab_size,
		model_path = config$tokenizer_model_path
		)
	} else {
  		model <- tokenizers.bpe::bpe_load_model(config$tokenizer_model_path)
	}
	Voc  <- model$vocabulary[, 2]
	File <- base::readChar(config$file_name, file.info(config$file_name)$size)
	Encoded <- tokenizers.bpe::bpe_encode(model, x = File, type = "ids")[[1]]+1
}


nvoc0   <- length(Voc)


############################################################
#Defining train data and test data
############################################################
data <- train_test_split(prop = config$split_prop,
                         tokens = Encoded,
                         dataset_generator = data_GPT,
                         block_size = config$block_size,
                         seed = 1)
BD.train <- data$train
BD.test <- data$test

############################################################
#Setting the model
############################################################
torch::torch_manual_seed(42)

Model <- GPT %>%
  luz::setup(loss = nn_cross_entropy_loss_0(), optimizer = torch::optim_adam) %>%
  luz::set_hparams(
    block_size = config$block_size,
    n_embd = config$n_embd,
    N_Layers = config$N_Layers,
    nvoc = nvoc0,
    Head = config$Head,
    p0 = config$p0
  ) %>%
  luz::set_opt_hparams(lr = config$lr)

############################################################
# Training the model.
# Change the number of workers.
############################################################
clip <- luz::luz_callback_gradient_clip(max_norm = 1, norm_type = 2)
amp <- luz::luz_callback_mixed_precision()

fitted <- luz::fit(Model,
  BD.train,
  epochs = config$epochs,
  accelerator = luz::accelerator(),
  valid_data = BD.test,
  dataloader_options = list(
    batch_size = config$batch_size,
    num_workers = config$num_workers,
    shuffle = FALSE
  ),
  callbacks = if(config$AMP) list(clip, amp) else list(clip, amp)
)

if(!config$BPE) 
	luz::luz_save_model_weights(fitted, "Model-Shakes_weights_Character.pt")
if(config$BPE) 
	luz::luz_save_model_weights(fitted, "Model-Shakes_weights_BPE.pt")

