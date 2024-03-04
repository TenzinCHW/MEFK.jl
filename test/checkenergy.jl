import MEFK, Combinatorics
import Flux


nbits = 4
embed_vals = Iterators.product(fill(Int8[0, 1], nbits)...) |> collect |> vec .|> collect
x = hcat(embed_vals...) |> transpose |> Array
counts = zeros(length(embed_vals))
counts[4] = 1
counts[13] = 1

net = MEFK.MEFMPNK(nbits, 3)
optim = Flux.setup(Flux.Adam(0.1), net)

for it in 1:100
    loss, grads = net(x, counts, it==1)
    Flux.update!(optim, net, grads)
end

energies = MEFK.energy(net, x)
probs = MEFK.energy2prob(energies)
@assert abs(0.5 - probs[4]) < 0.001
@assert abs(0.5 - probs[13]) < 0.001


net_full = MEFK.MEF3T(nbits)

net_full.W1 .= net.W[1][:]
W2 = zeros(size(net_full.W2)...)
W3 = zeros(size(net_full.W3)...)
ind2 = 1
for i in 1:nbits-1
    for j in i+1:nbits
        global ind2
        W2[i, j] = net.W[2][ind2]
        ind2 += 1
    end
end
ind3 = 1
for i in 1:nbits-2
    for j in i+1:nbits-1
        for k in j+1:nbits
            global ind3
            W3[i, j, k] = net.W[3][ind3]
            ind3 += 1
        end
    end
end
W2 += W2'
W3 = sum([permutedims(W3, p) for p in Combinatorics.permutations([1,2,3], 3)])
net_full.W2[:, :] .= W2
net_full.W3[:,:,:] .= W3

energies = MEFK.energy(net_full, x)
probs_full = MEFK.energy2prob(energies)
@assert abs(0.5 - probs_full[4]) < 0.001
@assert abs(0.5 - probs_full[13]) < 0.001

