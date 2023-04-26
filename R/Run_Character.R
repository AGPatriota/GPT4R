#AUTHOR: Alexandre Galvão Patriota
#IME-USP

source("R/GPT.R")
source("R/Generators.R")
source("config.R")

File = readChar(config$file_name, file.info(config$file_name)$size)
Voc = c("<PAD>",sort(unique(unlist(strsplit(File, "")))))
nvoc0   <- length(Voc)


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
Model$load_state_dict(state_dict = torch::torch_load("Model-Shakes_weights-2.pt")$parameters)

############################################################
#Predicting tokens like Shakespeare
############################################################
Model  = if (torch::cuda_is_available()) Model$cuda() else Model$cpu()
Context = Encoder(initial_context, Voc)

Tokens  = Generate(
	  Context,
	  Model,
	  config$block_size,
	  max_new_tokens=config$max_new_tokens,
	  temperature = config$temperature,
	  top_k = config$top_k,
	  device0= if (torch::cuda_is_available()) "cuda" else "cpu"
	  )


cat(paste(c(initial_context,Decoder(Tokens)), collapse=""))

