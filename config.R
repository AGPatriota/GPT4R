config <- list(
  file_name = "Shakespeare.txt",
  block_size = 256, # Maximum Context
  ncol = 384, # Embedding dimension
  N_Layers = 6, # Number of layers
  Head = 6, # Number of heads
  batch_size = 32,
  vocab_size = 68,
  p0 = 0.2, # Dropout proportion
  epochs = 9, # Number of epochs
  num_workers = 6, # Number of CPU workers
  n_tokens0 = 700,
  coverage = 0.999,
  split_prop = 0.9,
  lr = 0.00015,
  tokenizer_model_path = "youtokentome.bpe",
  max_new_tokens = 700,
  temperature = 0.7,
  top_k = 3
)
