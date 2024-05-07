import MEFK, Flux, LinearAlgebra, CUDA

array_cast = CUDA.cu
n = 4
model = MEFK.MEF3T(n; array_cast=array_cast)
optim = Flux.setup(Flux.Adam(0.1), model)
x = [[1, 0, 1, 0] [0, 1, 0, 1]]' .|> Int8
counts = ones(2)

for i in 1:100
    loss, grads = model(x, counts; reset_grad=true)
    Flux.update!(optim, model, grads)
end

en = MEFK.energy(model, x)
x_hat = MEFK.dynamics(model, x) |> Array
@assert all(x_hat == x)


convx = convergedynamics(model, x)
Test.@test all(convx == x)


# test partial iteration feature and no symmetrization of gradients
model = MEFK.MEF3T(4; array_cast=array_cast)
optim = Flux.setup(Flux.Adam(0.1), model)

for i in 1:100
    loss, grads = model(x, counts; iterate_nodes=3:4, symmetrize_grad=false, reset_grad=true)
    Flux.update!(optim, model, grads)
end

xp = [[1, 0, 0, 1] [0, 1, 1, 0]]' .|> Int8 |> array_cast
x_hat = MEFK.dynamics(model, xp; iterate_nodes=3:4) |> Array
@assert all(x_hat == x)

