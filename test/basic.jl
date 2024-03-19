import MEFK: MEFMPNK, dynamics, make_inds_winds, energy, energy2prob
import Flux, Test


n = 4
N = 2
model = MEFMPNK(n, N)
inds, _ = make_inds_winds(n, N)
model = MEFMPNK(n, N, inds) # Test both constructors
optim = Flux.setup(Flux.Adam(0.1), model)
x = [[1, 0, 1, 0] [0, 1, 0, 1]]' .|> Int8

for i in 1:100
    loss, grads = model(x, i==1; reset_grad=true)
    Flux.update!(optim, model, grads)
end

pred = dynamics(model, x)

gt = dynamics(model, x)
Test.@test all(pred == gt == x)


model = MEFMPNK(n, N, inds)
optim = Flux.setup(Flux.Adam(0.1), model)
for i in 1:100
    loss, grads = model(x, [2., 1], i==1; reset_grad=true)
    Flux.update!(optim, model, grads)
end
en = energy(model, [[0, 0, 0, 0] [0, 0, 0, 1] [0, 0, 1, 0] [0, 0, 1, 1] [0, 1, 0, 0] [0, 1, 0, 1] [0, 1, 1, 0] [0, 1, 1, 1] [1, 0, 0, 0] [1, 0, 0, 1] [1, 0, 1, 0] [1, 0, 1, 1] [1, 1, 0, 0] [1, 1, 0, 1] [1, 1, 1, 0] [1, 1, 1, 1]]' |> collect .|> Int8)
println(energy2prob(en))

