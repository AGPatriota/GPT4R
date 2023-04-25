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
if (Train) {
  if (BPE) {
    # Train the Tokens?
    Train_tokens <- FALSE
    source("R/Train_BPE.R")
  } else {
    source("R/Train_Character.R")
  }
}

############################################################
#Running/predicting
############################################################

if (Run) {
  initial_context = "My lord"
  if (BPE) {
    if (file.exists("Model-Shakes_weights.pt")) {
      source("R/Run_BPE.R")
    } else {
      cat("
Make sure you have downloaded the required weights:  
https://drive.google.com/file/d/1XKr__cI4ZBZEiv1EGZc1Yc-bCQGOobKX  
")
    }
  }
  if (!BPE) {
    if (file.exists("Model-Shakes_weights-2.pt")) {
      source("R/Run_Character.R")
    } else {
      cat("
Make sure you have downloaded the required weights:  
https://drive.google.com/file/d/1VDVRLk0o6wsdrlGTfF8xxqjtLo8n03y1
")
    }
  }
}


