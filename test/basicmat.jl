import MEFK, Flux, LinearAlgebra

model = MEFK.MEF2T(4)
optim = Flux.setup(Flux.Adam(0.1), model)
x = [[1, 0, 1, 0] [0, 1, 0, 1]]' .|> Int8
counts = ones(2)


for i in 1:100
    loss, grads = model(x, counts; reset_grad=true)
    Flux.update!(optim, model, grads)
end

en = MEFK.energy(model, x)
x_hat = MEFK.dynamics(model, x)
@assert all(x_hat == x)

# test partial iteration feature and no symmetrization of gradients
model = MEFK.MEF3T(4)
optim = Flux.setup(Flux.Adam(0.1), model)

for i in 1:100
    loss, grads = model(x, counts; iterate_nodes=3:4, symmetrize_grad=false, reset_grad=true)
    Flux.update!(optim, model, grads)
end

xp = [[1, 0, 0, 1] [0, 1, 1, 0]]' .|> Int8
x_hat = MEFK.dynamics(model, xp; iterate_nodes=3:4)
@assert all(x_hat == x)

