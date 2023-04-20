# GPT for R


A simple GPT code in R trained with Shakespeare text. You can use the trained model to generate Shakespeare-like texts or you can train from scratch with other hyperparameters (vocabulary size, context size, embedding dimension, etc) and other texts. In this repository, it is provided a training with BPE and without BPE (based only with characters like in Karpathy's [video](https://www.youtube.com/watch?v=kCc8FmEb1nY)

The Shakespeare text was downloaded [here](https://github.com/karpathy/ng-video-lecture/blob/master/input.txt)


## Dependencies:

- [tokenizers.bpe](https://cran.r-project.org/web/packages/tokenizers.bpe/index.html)
- [torch](https://cran.r-project.org/web/packages/torch/index.html) 
- [luz](https://cran.r-project.org/web/packages/luz/vignettes/get-started.html) 

## quick start

If you have a GPU, you can try training this baby GPT model from scratch. You need to define a batch size that fits into your GPU memory (default is 32). Your results might be different for other batch sizes. Before training the model, make sure the number of workers is in order with your machine. Default is 6 workers. In order to train with BPE (Byte-Pair Encoding) or without BPE, open the file `main.R` and set `BPE = TRUE` or `BPE = FALSE`. If you want to train set `Train = TRUE`. If you want to predict tokens set `Run = TRUE`. After setting all hyperparameters in the `main.R` file, run the following in the main folder

```
source('main.R')
```

The file `Train_BPE.R` trains the model by using a vocabulary tokenized by BPE (from the package `tokenizers.bpe`). The file `youtokentome.bpe` contains the tokenized vocabulary (of size 68 tokens) for the Shakespeare text. The weights for this version, that generates tokens with no line breaks, can be downloaded [here](https://drive.google.com/file/d/1XKr__cI4ZBZEiv1EGZc1Yc-bCQGOobKX)

The file `Train_Character.R`  trains the model by using a vocabulary with 65 single characters plus a PAD character. This procedure is closer to what Karpathy does in his video and the generated text contains proper line breaks which makes the reading a little bit more pleasant. Download the trained weights for this model [here](https://drive.google.com/file/d/1VDVRLk0o6wsdrlGTfF8xxqjtLo8n03y1)

You can also use the Automatic Mixing Precision (AMP) which is implemented in `torch` package but not in `luz` yet (19/04/2023). This helps increasing the capacity of your training.

## Examples of Shakespeare-like texts

An example of 700 tokens generated by setting `BPE = FALSE`, `Train = FALSE` and `Run = TRUE` is given below:

> My lord, ungland!
> 
> KING RICHARD IIII:
> Now, you before my son, bear my troth, the blood
> And had all the common ears.
> 
> KING RICHARD III:
> Ay, my good lord, then all my lord,
> Imparel and the execution of our own.
> 
> KING RICHARD III:
> I do not hear them as I seem to fear;
> And then bid by the sons that I am king.
> 
> KING RICHARD III:
> What will not make thee from that thou stand as my
> I will consent to be and his brother?
> 
> QUEEN ELIZABETH:
> And thou shalt say not? and she doth make my country's
> name to the truth, that our life and arrival forget
> Untimely thou thinking them, and that was I do proceed
> The princes of the throng of their sons,
> I have to leve them still out at once that you


An example of 700 tokens generated by setting `BPE = TRUE`, `Train = FALSE` and `Run = TRUE` is given below:

...


### License

MIT
