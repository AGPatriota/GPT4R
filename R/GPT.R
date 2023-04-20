#AUTHOR: Alexandre Galvão Patriota
#IME-USP

GPT <- nn_module(
  initialize = function(block_size, ncol, N_Layers, nvoc, Head, p0=0.1) {
    self$N     <- N_Layers
    self$block_size <-block_size
    self$PE    <- nn_embedding(block_size,ncol)
    self$L0    <- nn_embedding(nvoc,ncol, padding_idx = 1)
    self$MH    <- nn_module_list(lapply(1:N_Layers, function(x) nn_multihead_attention(ncol,Head, dropout = p0)))
    self$scale1<- nn_module_list(lapply(1:N_Layers, function(x) nn_layer_norm(ncol)))
    self$scale2<- nn_module_list(lapply(1:N_Layers, function(x) nn_layer_norm(ncol)))
    self$scale3<- nn_layer_norm(ncol,elementwise_affine = TRUE)    
    self$FFN   <- nn_module_list(lapply(1:N_Layers, function(x) nn_sequential(nn_linear(ncol,4*ncol),nn_gelu(), nn_linear(4*ncol,ncol), nn_dropout(p0))))
    self$A     <- nn_linear(ncol, nvoc, bias = FALSE) 
    self$drop0 <- nn_dropout(p=p0)
  },
  forward = function(x){
    x1    <- torch_arange(1,self$block_size, dtype = torch_int(),  device = x$device)$unsqueeze(1)
    wei   <- torch_zeros(self$block_size,self$block_size, dtype = torch_bool(), device = x$device)
    wei[upper.tri(wei)] <- 1
    if(x$size(2)<self$block_size){
      zeros <- torch_tensor(rep(1,self$block_size-x$size(2)), dtype=torch_int(), device = x$device)$unsqueeze(1)
      x     <- torch_cat(list(zeros,x),2)
    }
    output <- self$L0(x) + self$PE(x1)
    output <- self$drop0(output)
    for(j in 1:self$N){
      Q       <- torch_transpose(self$scale1[[j]](output), 1,2)
      output  <- output + torch_transpose(self$MH[[j]](Q,Q,Q, attn_mask = wei, need_weights = FALSE)[[1]],1,2)
      output  <- output + self$FFN[[j]](self$scale2[[j]](output))
    }
    output <- self$A(self$scale3(output))
    output
  }
)
