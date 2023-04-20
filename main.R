require(tokenizers.bpe)
require(torch)
require(luz)
source('R/GPT.R')
source('R/Generators.R')

BPE   = FALSE
Train = FALSE
Run   = TRUE
fileName   <- "Shakespeare.txt"
block_size0 <- 256   #Maximum Context 
ncol0       <- 384   #Embedding dimension 
N_Layers0   <- 6     #Number of layers 
Head0       <- 6     #Number of heads


############################################################
#training
############################################################
if(Train){
	batch_size0 <- 32    #Batch size
	if(BPE){
		nvoc0 <- 68    #Size of the vocabulary BPE
	}
	p00        <- 0.2   #Dropout proportion
	epochs0    <- 9     #Number of epochs
	Workers0   <- 6     #Number of CPU workers

	if(BPE){
		#Train the Tokens?
		Train_tokens = FALSE
		source('R/Train_BPE.R')
	}
	if(!BPE){
		source('R/Train_Character.R')
	}
}

############################################################
#Running/predicting
############################################################

if(Run){
        n_tokens0   <- 700   #Number of generated tokens	
	if(BPE){
		if(file.exists("Model-Shakes_weights.pt")){
			source('R/Run_BPE.R')		
		} else{
	
cat('
Make sure you have downloaded the required weights:  
https://drive.google.com/file/d/1XKr__cI4ZBZEiv1EGZc1Yc-bCQGOobKX  
')
}
	}
	if(!BPE){
		if(file.exists("Model-Shakes_weights-2.pt")){
			source('R/Run_Character.R')
		}
		else{
	
cat('
Make sure you have downloaded the required weights:  
https://drive.google.com/file/d/1VDVRLk0o6wsdrlGTfF8xxqjtLo8n03y1
')
}

	}
	
}


