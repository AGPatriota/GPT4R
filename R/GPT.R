# AUTHOR: Alexandre Galvão Patriota
# IME-USP

GPT <- torch::nn_module(
  initialize = function(block_size, n_embd, N_Layers, nvoc, Head, p0 = 0.1) {
    self$N <- N_Layers
    self$block_size <- block_size
    self$PE <- torch::nn_embedding(block_size, n_embd)
    self$L0 <- torch::nn_embedding(nvoc, n_embd, padding_idx = 1)
    self$MH <- torch::nn_module_list(lapply(
      1:N_Layers,
      function(x) torch::nn_multihead_attention(n_embd, Head, dropout = p0)
    ))
    self$scale1 <- torch::nn_module_list(lapply(
      1:N_Layers,
      function(x) torch::nn_layer_norm(n_embd)
    ))
    self$scale2 <- torch::nn_module_list(lapply(
      1:N_Layers,
      function(x) torch::nn_layer_norm(n_embd)
    ))
    self$scale3 <- torch::nn_layer_norm(n_embd, elementwise_affine = TRUE)
    self$FFN <- torch::nn_module_list(lapply(
      1:N_Layers,
      function(x) {
        torch::nn_sequential(
          torch::nn_linear(n_embd, 4 * n_embd),
          torch::nn_gelu(),
          torch::nn_linear(4 * n_embd, n_embd),
          torch::nn_dropout(p0)
        )
      }
    ))
    self$A <- torch::nn_linear(n_embd, nvoc, bias = FALSE)
    self$drop0 <- torch::nn_dropout(p = p0)
  },
  forward = function(x) {
    x1 <- torch::torch_arange(1, self$block_size,
      dtype = torch::torch_int(),
      device = x$device
    )$unsqueeze(1)
    wei <- torch::torch_zeros(self$block_size, self$block_size,
      dtype = torch::torch_bool(), device = x$device
    )
    wei[upper.tri(wei)] <- 1
    if (x$size(2) < self$block_size) {
      zeros <- torch::torch_tensor(rep(1, self$block_size - x$size(2)),
        dtype = torch::torch_int(),
        device = x$device
      )$unsqueeze(1)
      x <- torch::torch_cat(list(zeros, x), 2)
    }
    output <- self$L0(x) + self$PE(x1)
    output <- self$drop0(output)
    for (j in 1:self$N) {
      Q <- torch::torch_transpose(self$scale1[[j]](output), 1, 2)
      output <- output + torch::torch_transpose(
        self$MH[[j]](Q, Q, Q,
          attn_mask = wei,
          need_weights = FALSE)[[1]],
        1, 2
      )
      output <- output + self$FFN[[j]](self$scale2[[j]](output))
    }
    output <- self$A(self$scale3(output))
    output
  }
)
