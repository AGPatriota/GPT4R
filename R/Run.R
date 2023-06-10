#AUTHOR: Alexandre Galvão Patriota
#IME-USP

source("R/GPT.R")
source("R/Generators.R")
source("config.R")


if(!config$BPE) {
	File = readChar(config$file_name, file.info(config$file_name)$size)
	Voc = c("<PAD>",sort(unique(unlist(strsplit(File, "")))))
	nvoc0   <- length(Voc)
}

if(config$BPE) {
	model_bpe <- tokenizers.bpe::bpe_load_model(config$tokenizer_model_path)
	nvoc0 <- length(model_bpe$vocabulary[, 2])
}



############################################################
#Loading the model
############################################################
Model <- GPT(
  block_size = config$block_size,
  n_embd = config$n_embd,
  N_Layers = config$N_Layers,
  nvoc = nvoc0,
  Head = config$Head
)
 
############################################################
#Updating the modeol with trained parameters
############################################################
if(!config$BPE) {
	Model$load_state_dict(state_dict = torch::torch_load("Model-Shakes_weights_Character.pt")$parameters)
	Context = Encoder(config$initial_context, Voc)
}

if(config$BPE) {
	Model$load_state_dict(state_dict = torch::torch_load("Model-Shakes_weights_BPE.pt")$parameters)
	Context = tokenizers.bpe::bpe_encode(model_bpe, x = config$initial_context, type = "ids")[[1]]+1
}

############################################################
#Predicting tokens like Shakespeare
############################################################
Model  = if (torch::cuda_is_available()) Model$cuda() else Model$cpu()


Tokens  = Generate(
	  Context,
	  Model,
	  config$block_size,
	  max_new_tokens=config$max_new_tokens,
	  temperature = config$temperature,
	  top_k = config$top_k,
	  device0= if (torch::cuda_is_available()) "cuda" else "cpu"
	  )

#if(!config$BPE)
#	cat(paste(c(config$initial_context,Decoder(Tokens)), collapse=""))

#if(config$BPE) 
#	cat(paste(config$initial_context,tokenizers.bpe::bpe_decode(model_bpe, x = as.integer(Tokens-1)), collpase=""))
