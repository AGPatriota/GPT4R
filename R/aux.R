train_test_split <- function(prop, tokens, dataset_generator, block_size, seed = 1) {
  set.seed(seed)
  n <- round(length(tokens) * prop)
  train_data <- tokens[1:n]
  test_data <- tokens[(n + 1):length(tokens)]
  BD.train <- data_GPT(train_data, block_size = block_size, test = FALSE)
  BD.test <- data_GPT(test_data, block_size = block_size, test = TRUE)

  return(list(train = BD.train, test = BD.test))
}
