import MEFK
import Flux


nbits = 4
embed_vals = Iterators.product(fill(Int8[0, 1], nbits)...) |> collect |> vec .|> collect
x = hcat(embed_vals...) |> transpose |> Array
counts = zeros(length(embed_vals))
counts[4] = 1
counts[13] = 1

net_full = MEFK.MEF3T(nbits)
optim = Flux.setup(Flux.Adam(0.1), net_full)


for it in 1:100
    loss = net_full(x, counts)
    grads = MEFK.retrieve_reset_gradients!(net_full)
    Flux.update!(optim, net_full, grads)
end

energies = MEFK.energy(net_full, x)
probs_full = MEFK.energy2prob(energies)
@assert abs(0.5 - probs_full[4]) < 0.001
@assert abs(0.5 - probs_full[13]) < 0.001


net = MEFK.MEFMPNK(nbits, 3)
for i in 1:net.N
    empty!(net.W[i])
end
for w in net_full.W1
    push!(net.W[1], w)
end
for i in 1:net.n-1
    for j in i+1:net.n
        push!(net.W[2], net_full.W2[i, j])
    end
end
for i in 1:net.n-2
    for j in i+1:net.n-1
        for k in j+1:net.n
            push!(net.W[3], net_full.W3[i, j, k])
        end
    end
end

energies = MEFK.energy(net, x)
probs = MEFK.energy2prob(energies)

@assert abs(0.5 - probs[4]) < 0.001
@assert abs(0.5 - probs[13]) < 0.001

