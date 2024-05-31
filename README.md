# MEFK.jl

This is the julia package repository for fitting K-th order MEF.


### Installation
Install julia 1.10.2 or higher.
To add, simply start a julia session in your project home directory and enter the following (do not copy and paste the ] character):
```
]
activate .
add https://github.com:TenzinCHW/MEFK.jl.git
```


### Quick start
For 2nd order MEF, we initialize the object as follows
```
import MEFK: MEF2T
num_neurons = 10
model = MEF2T(num_neurons)
```

We use `MEF3T` for 3rd order. The more general `MPNK` requires another argument, `K`, or the order.
```
import MEFK: MPNK
num_neurons = 20
K = 4
model = MPNK(num_neurons, K)
```

To fit the model, we must prepare a dataset and dataloader from Flux. Data should contain `num_neurons` columns.
```
optim = Flux.setup(Flux.Adam(0.1), model)
x = [[1, 0, 1, 0] [0, 1, 0, 1]]' .|> Int8
for i in 1:100
    loss, grads = model(x; reset_grad=true)
    Flux.update!(optim, model, grads)
end
```
Set `reset_grad` to `false` if you need to batch the data.

To converge data `x` to attractors after training `model`
```
ŷ = dynamics(model, x)
```


### In-depth usage notes
To use GPU for training, first import the `CUDA` or `Metal` (on MacOS) packages and initialize with 
```
import CUDA: cu  # import Metal: mtl
model = MEF2T(num_neurons; array_cast=cu)
model = MEF3T(num_neurons, array_cast=cu)
model = MPNK(num_neurons, K; array_cast=cu)
```

In order to save the model, all parameters must be on CPU. To transfer an instantiated model to the CPU or GPU, pass the model instance and the device array constructor into the model's respective constructor
```
model = typeof(model)(model, Array)  # transfer to CPU
model = typeof(model)(model, cu)  # transfer to CUDA GPU
```

You may also pass in an `Vector{Vector{AbstractMatrix{Int}}}` of `indices` to the constructor for `MPNK` to specify sparsity.
This parameter must be of length `num_neurons`. Each inner vector contains `K-1` matrices which have at least 1 row each, and in increasing order have 1 to `K-1` columns.
Each inner vector at index $i$ is a list of the combination of bits of each order that include the bit $i$.
Each matrix at index j of the $i^{th}$ inner vector corresponds to the indices of bits for the $j+1^{th}$ order.
For example, in a 4 neuron model with order 2.
```
indices = [[[2;;]], [[1; 3;;]], [[2; 4;;]], [[3;;]]]
```
will connect each bit to its adjacent bits.

```
indices = [[[2; 3;;], [2 3;;]], [[1; 3; 4;;], [1 3; 1 4; 3 4;;]], [[1; 2; 4;;], [1 2; 1 4; 2 4;;]], [[2; 3;;], [2 3;;]]]
```
will connect each bit with bits that are up to 2 indices away for 2nd and 3rd order.
Note that the index $i$ does not appear in the $i^{th}$ element of `indices` since it is assumed to be part of any combination that appears.

