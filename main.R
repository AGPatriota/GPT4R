require(tokenizers.bpe)
require(torch)
require(luz)
source('R/GPT.R')
source('R/Generators.R')

BPE   = FALSE
Train = FALSE
Run   = TRUE


############################################################
#training
############################################################
if(Train){
	batch_size0 <- 32    #Batch size
	block_size0 <- 256   #Maximum Context 
	ncol0       <- 384   #Embedding dimension 
	N_Layers0   <- 6     #Number of layers 
	Head0       <- 6     #Number of heads 
	if(BPE){
		nvoc0 <- 68    #Size of the vocabulary BPE
	}
	if(!BPE){
		nvoc0 <- 66    #Size of the vocabulary Character
	}
	p00        <- 0.2   #Dropout proportion
	epochs0    <- 9     #Number of epochs
	Workers0   <- 6     #Number of CPU workers
	fileName   <- "Shakespeare.txt"

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
	fileName    <- "Shakespeare.txt"
	block_size0 <- 256   #Maximum Context 
	ncol0       <- 384   #Embedding dimension 
	N_Layers0   <- 6     #Number of layers 
	Head0       <- 6     #Number of heads
        n_tokens0   <- 700   #Number of generated tokens	
	if(BPE){
		nvoc0 <- 68    #Size of the vocabulary BPE
		source('R/Run_BPE.R')		
	}
	if(!BPE){
		nvoc0 <- 66    #Size of the vocabulary Character
		source('R/Run_Character.R')
	}
	
}


