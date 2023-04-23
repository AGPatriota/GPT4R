#AUTHOR: Alexandre Galvão Patriota
#IME-USP


File = readChar(fileName, file.info(fileName)$size)
Voc = c("<PAD>",sort(unique(unlist(strsplit(File, "")))))
nvoc0   <- length(Voc)


############################################################
#Loading the model
############################################################
Model <- GPT(block_size = block_size0, ncol = ncol0 ,N_Layers = N_Layers0,  nvoc = nvoc0,Head=Head0)
 
############################################################
#Updating the modeol with trained parameters
############################################################
Model$load_state_dict(state_dict = torch_load("Model-Shakes_weights-2.pt")$parameters )

############################################################
#Predicting tokens like Shakespeare
############################################################
Model  = if (cuda_is_available()) Model$cuda() else Model$cpu()
Context0 = "My lord"
Context = Encoder(Context0, Voc)

Tokens  = Generate(
	  Context,
	  Model,
	  block_size0,
	  max_new_tokens=n_tokens0,
	  temperature = 0.7,
	  top_k = 3,
	  device0= if (cuda_is_available()) "cuda" else "cpu"
	  )


cat(paste(c(Context0,Decoder(Tokens)), collapse=""))

