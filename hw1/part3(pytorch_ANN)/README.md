The execution time was the real concern for me when writing this program. The data or the model was large enough 
for CPU in some cases. Therefore, I used cuda. The program chooses an available device with the call below
```
DEVICE = device = 'cuda' if torch.cuda.is_available() else 'cpu'
```
I use the hyperparameters depth, breadth, and the activation function in a function `Classifier2dModel`. 
I could have use a class inheriting from `nn.Module` but preferred to initialize my model using `nn.Sequential`.
Input is 784, and the output is 10 as stated in the assignment text. 

I get the model, other hyperparameters not used in the model, loss function and optimizer in a class `Classifier2d`. I also use `DataLoader` to load the data. I set `pin_memory = True` in order that the data to be used in the GPU is stored directly in the pinned memory without being transferred from pageable memory. Note that we need to send each batch to the GPU. For each batch, the process of transferring to the pinned memory is eliminated. This could reduce the time execution cost. For details, please refer [here](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/). I use the batch size for the training set, in its data loader, and shuffle the data to avoid overfitting as I state in the report. 

As also stated in the report, I scan all the batches in one epoch. I came across this convention on the internet. `loss.backward` calls accumulate the the gradients into the `grad` attribute. Before the next call, this attribute should be cleared with `optimizer.zero_grad`, which is used with the parameter `set_to_none` set `True`. This improves the performance by using less memory operations. Instead of zeroing out each individual parameter in the attribute, it is set to `None`, its default value. For details, please refer [here](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html) and [here](https://pytorch.org/docs/stable/generated/torch.Tensor.grad.html). 

Similarly to training, we scan all the batches when evaluating, or testing (`eval` method). This is trivial since we have a single batch for testing data. Instead of `torch.no_grad`, I preferred `torch.inference_mode`, as it is stated [here](https://pytorch.org/docs/stable/generated/torch.inference_mode.html) that "code run under this mode gets better performance by disabling view tracking and version counter bumps". 

In the `loop` method, I loop through all the epochs, saving each loss and accuracy value to an array. 

I specify the hyperparameters name using key items and the list of values they can take as their corresponding value items in the `configs` variable. Given `configs` to `product_dict` returns all the combinations with the value items in the key order. 

For each configuration, `test` function creates a classifier, trains and evaluates it 10 times as stated in the assignment text, obtaines a 95% confidence metric out of the data and save the plots as PNG files. In `evalConfig` function, the return data is then used to construct a LATEX representation, which is written to a file with the name `[RESULTSFILE]` in `iterConfigs`. The information is also printed to stdout for convenience with the elapsed time in seconds.

I took the preprocessing stage as it was given. In the first step, all configurations are iterated over with `iterConfigs`. In this function, the maximum configuration is returned as a dictionary in a variable `maxconfig`. One additional `evalConfig` call is executed on the new data with this configuration, and the result is printed.