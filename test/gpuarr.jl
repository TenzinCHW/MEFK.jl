import MEFK: MEFMPNK, dynamics, make_inds_winds
import Flux, Test, CUDA


n = 4
N = 2
array_cast = CUDA.cu

inds, winds = make_inds_winds(n, N)
model = MEFMPNK(n, N, inds; array_cast=array_cast)
model = MEFMPNK(n, N; array_cast=array_cast)
optim = Flux.setup(Flux.Adam(0.1), model)
x = [[1, 0, 1, 0] [0, 1, 0, 1]]' .|> Int8

for i in 1:100
    loss, grads = model(x; reset_grad=true)
    Flux.update!(optim, model, grads)
end

pred = dynamics(model, x; array_cast=array_cast)

W1 = Float32[-12.1044, -12.3351, -24.5112, -11.0210]
W2 = Float32[-3.1486, 15.7777,  3.5976, 18.6813,  0.5786,  6.8495] |> array_cast
model.W[1] = W1
model.W[2] = W2
gt = dynamics(model, x; array_cast=array_cast)
Test.@test all(pred == gt == y)

