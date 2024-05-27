# AUTHOR: Alexandre Galvão Patriota
# IME-USP

GPT <- torch::nn_module(
  initialize = function(block_size, n_embd, N_Layers, nvoc, Head, p0 = 0.1) {
    self$N <- N_Layers
    self$block_size <- block_size
    self$wpe <- torch::nn_embedding(block_size, n_embd)
    self$wte <- torch::nn_embedding(nvoc, n_embd, padding_idx = 1)
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
    self$ln_f <- torch::nn_linear(n_embd, nvoc, bias = FALSE)
    self$drop0 <- torch::nn_dropout(p = p0)
  },
  forward = function(x) {
    x1 <- torch::torch_arange(1, x$size(2),
      dtype = torch::torch_int(),
      device = x$device
    )$unsqueeze(1)
    wei <- torch::torch_zeros(x$size(2), x$size(2),
      dtype = torch::torch_bool(), device = x$device
    )
    wei[upper.tri(wei)] <- 1
    output <- self$wte(x) + self$wpe(x1)
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
    output <- self$ln_f(self$scale3(output))
    output
  }
)
