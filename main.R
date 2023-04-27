require(tokenizers.bpe)
require(torch)
source('R/GPT.R')
source('R/Generators.R')
source('config.R')

############################################################
#training
############################################################
if (config$Train) {
	require(luz)
	cat("Training with", ifelse(config$BPE, "BPE tokenizer \n", "Character-based tokenizer \n"))
  source("R/Train.R")
}

############################################################
#Running/predicting
############################################################

if (config$Run) {
  if (config$BPE) {
    if (!file.exists("Model-Shakes_weights_BPE.pt")){ 
	    cat("
Make sure you have downloaded the required weights:  
https://drive.google.com/file/d/1XKr__cI4ZBZEiv1EGZc1Yc-bCQGOobKX  
")} else {
          source("R/Run.R")
    }
  }

  if (!config$BPE) {
    if (!file.exists("Model-Shakes_weights_Character.pt")) {
	    cat("
Make sure you have downloaded the required weights:  
https://drive.google.com/file/d/1VDVRLk0o6wsdrlGTfF8xxqjtLo8n03y1
")} else {
      source("R/Run.R")
    } 
  }
}


