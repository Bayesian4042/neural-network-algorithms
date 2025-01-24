1. Neural networks relies on matrix multiplication to do the calculations that needed to be done for passing the signal forward through a neural network, and also for the backward pass that updates the network link weights

2. These calculation could be written using matrix multiplication. Instead of writing out hundreds, or even thousands, of separate calculations, we could just write down a simple matrix multiplication.

3. Many real world neural networks are much larger and process much more data, in those cases, the training times can be very long, taking hours days or weeks even with libraries like numpy to do fast matrix calculation.

4. In search for more speed, machine learning researchers started taking advantage of special hardware found in some computers, originally designed to improve graphics performance, called graphic cards.

# NVIDIA CUDA
Those graphics cards contain a GPU, or graphic processing unit. Unlike CPU, A GPU is designed to perform specific tasks, and do them well. One of the task is to carry out arithmetic, including matrix multiplication, in a highly parallel way.
The software tools for taking advantage of NVIDIA GPUs are called CUDA.

You can check if CUDA is available for not:
```
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    print("using cuda: torch.cuda.get_device_name(0)")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```