config <- list(
  #corpus for training (global)
  file_name = "Shakespeare.txt",
  
  BPE = !TRUE,
  Train = !TRUE,
  Run   = TRUE,

  #GPT parameters (global)
  block_size = 256, # Maximum Context
  n_embd = 384,     # Embedding dimension
  N_Layers = 6,     # Number of layers
  Head = 6,         # Number of heads
  vocab_size = 68,  # Vocabulary size for BPE (for trainning)

  #Training parameters (global)
  lr = 0.00015,     # Learning rate  
  batch_size = 32,  # Batch size
  p0 = 0.2,         # Dropout proportion
  epochs = 9,       # Number of epochs
  num_workers = 6,  # Number of CPU workers

  #Parameters for BPE algorithm (Specific)
  coverage = 0.999, # Coverage for BPE algorithm
  split_prop = 0.9, # Proportion of training data
  tokenizer_model_path = "youtokentome.bpe",
  Train_tokens = FALSE, # Train the Tokens in the BPE algorithm?

  #Parameters for generating tokens
  initial_context = "My lord!",
  max_new_tokens = 700,
  temperature = 0.7,
  top_k = 3
)
