#AUTHOR: Alexandre Galvão Patriota
#IME-USP

source("config.R")
source("R/GPT.R")
source("R/Generators.R")

model_bpe <- tokenizers.bpe::bpe_load_model(config$tokenizer_model_path)
nvoc0 <- length(model_bpe$vocabulary[, 2])

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
Model$load_state_dict(state_dict = torch::torch_load("Model-Shakes_weights.pt")$parameters)

############################################################
#Predicting tokens like Shakespeare
############################################################
Model = if (torch::cuda_is_available()) Model$cuda() else Model$cpu()
Context = tokenizers.bpe::bpe_encode(model_bpe, x = initial_context, type = "ids")[[1]]
Tokens  = Generate(
  Context,Model,
  config$block_size,
  max_new_tokens=config$max_new_tokens,
  temperature = config$temperature,
  top_k = config$top_k,
  device0=if (torch::cuda_is_available()) "cuda" else "cpu"
  )

cat(paste(initial_context,tokenizers.bpe::bpe_decode(model_bpe, x = as.integer(Tokens)), collpase=""))



