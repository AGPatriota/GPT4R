#AUTHOR: Alexandre Galvão Patriota
#IME-USP

model <- bpe_load_model("youtokentome.bpe")
nvoc0     <- length(model$vocabulary[,2])

############################################################
#Loading the model
############################################################
Model <- GPT(block_size = block_size0, ncol = ncol0 ,N_Layers = N_Layers0,  nvoc = nvoc0,Head=Head0)
 
############################################################
#Updating the modeol with trained parameters
############################################################
Model$load_state_dict(state_dict = torch_load("Model-Shakes_weights.pt")$parameters )

############################################################
#Predicting tokens like Shakespeare
############################################################
Model = if (cuda_is_available()) Model$cuda() else Model$cpu()
Context0 = "My lord"
Context = bpe_encode(model, x = Context0, type = "ids")[[1]]
Tokens  = Generate(
  Context,Model,
  block_size0,
  max_new_tokens=n_tokens0,
  temperature = 0.7,
  top_k = 3,
  device0=if (cuda_is_available()) "cuda" else "cpu"
  )

cat(paste(Context0,bpe_decode(model, x = as.integer(Tokens)), collpase=""))



