import Flux
import Combinatorics, LinearAlgebra

struct MEF3T{F<:AbstractFloat}
    n::Int
    W1::AbstractVector{F}
    W2::AbstractMatrix{F}
    W3::AbstractArray{F}
    W2_mask::AbstractMatrix{F}
    W3_mask::AbstractArray{F}
    gradients::AbstractVector
    array_cast


    function MEF3T(n::Int, W1, W2, W3, mW2, mW3, gradients, array_cast)
        new{Float32}(n, W1, W2, W3, mW2, mW3, gradients, array_cast)
    end

    function MEF3T(n::Int, W1, W2, W3; array_cast=Array)
        W2_mask = (ones(n, n) - LinearAlgebra.I) |> array_cast
        W3_mask = MEFK.one_m_I_3D(n) |> array_cast
        gradients = [
            zeros(n),
            zeros(n, n) |> array_cast,
            zeros(n, n, n) |> array_cast]
        MEF3T(n, W1, W2, W3, W2_mask, W3_mask, gradients, array_cast)
    end

    function MEF3T(n::Int; array_cast=Array)
        W1 = zeros(n)
        W2 = zeros(n, n) |> array_cast
        W3 = zeros(n, n, n) |> array_cast
        MEF3T(n, W1, W2, W3; array_cast=array_cast)
    end

    function MEF3T(net::MEF3T, array_cast=Array)
        MEF3T(net.n,
              net.W1,
              net.W2 |> array_cast,
              net.W3 |> array_cast,
              net.W2_mask |> array_cast,
              net.W3_mask |> array_cast,
              net.gradients .|> array_cast,
              array_cast)
    end
end


Flux.@functor MEF3T


function diag(a::AbstractMatrix)
    [a[i, i] for i in 1:size(a)[1]]
end


function one_m_I_3D(n::Int)
    nI = ones(n, n, n)
    for i in 1:n
        for j in 1:n
            nI[i, j, j] = 0
            nI[j, i, j] = 0
            nI[j, j, i] = 0
        end
    end
    nI
end


function adjust_W3(net::MEF3T, symmetrize::Bool)
    g3 = net.gradients[3][:, :, :]
    if symmetrize
        g3 = sum([permutedims(g3, p) for p in
                Combinatorics.permutations([1,2,3], 3) |> collect]) ./ 6
    end
    g3 .* net.W3_mask
end
    


function adjust_W2(net::Union{MEF3T, MEF2T}, symmetrize::Bool)
    g2 = net.gradients[2][:, :]
    if symmetrize
        g2 = (g2 + g2') ./ 2
    end
    g2 .* net.W2_mask
end


function retrieve_reset_gradients!(net::MEF3T; symmetrize=true, reset_grad=false)
    g1 = net.gradients[1][:]
    g2 = adjust_W2(net, symmetrize)
    g3 = adjust_W3(net, symmetrize)

    if reset_grad
        net.gradients[1] .= 0
        net.gradients[2] .= 0
        net.gradients[3] .= 0
    end
    (n=nothing, W1=g1, W2=g2, W3=g3, W2_mask=nothing, W3_mask=nothing,
        gradients=nothing, array_cast=nothing)
end


function (net::MEF3T)(x::AbstractMatrix; iterate_nodes=nothing, reset_grad=false)
    sz = size(x)[1]
    counts = ones(sz)
    net(x, counts; iterate_nodes=iterate_nodes, reset_grad=reset_grad)
end


function (net::MEF3T)(x::AbstractMatrix, counts::AbstractVector;
                      iterate_nodes=nothing, symmetrize_grad=true, reset_grad=false)
    x = x |> net.array_cast
    W1 = net.W1
    W2 = net.W2
    W3 = net.W3
    loss = 0
    counts = counts |> net.array_cast
    #counts ./= sum(counts)
    iterate_nodes = isnothing(iterate_nodes) ? (1:net.n) : iterate_nodes
    for i in iterate_nodes
        flip = 1 .- 2 .* x[:, i]
        use_W1 = W1[i]
        use_W2 = W2[i, :]
        use_W3 = W3[i, :, :]
        bit_obj = mef_obj(use_W1, use_W2, use_W3, x, flip, counts, net.array_cast)
        loss += sum(bit_obj)

        flip_bit_obj = flip .* bit_obj .* counts
        net.gradients[1][i] += sum(flip_bit_obj)
        net.gradients[2][i, :] += x' * flip_bit_obj
        net.gradients[3][i, :, :] += broadcast(*, flip_bit_obj, x)' * x ./ 2
    end

    return loss, retrieve_reset_gradients!(net; symmetrize=symmetrize_grad, reset_grad=reset_grad)
end


function mef_obj(W1, W2, W3, x, flip, counts, array_cast)
    #out = net.W1[pos] * flip .+ x * use_W2 .* flip + sum((x * use_W3) .* x .* flip, dims=2)

    #tmp1 = sum((x * W3) .* x, dims=2)
    tmp1 = diag((x * W3) * x' |> Array) |> array_cast
    out = flip .* (tmp1 ./ 2 + x * W2 .+ W1)
    exp.(clamp.(out .* counts, -10, 10))
end


function dynamics(net::MEF3T, x::AbstractMatrix; iterate_nodes=nothing)
    x_ = x[:, :] |> net.array_cast
    iterate_nodes = isnothing(iterate_nodes) ? (1:net.n) : iterate_nodes
    for i in iterate_nodes
        out = dyn_step(net, x_, i)
        x_[:, i] .= out .> 0
    end
    x_
end


function dyn_step(net::MEF3T, x_, i)
    use_W2 = net.W2[i, :]
    use_W3 = net.W3[i, :, :]
    sum((x_ * use_W3) .* x_, dims=2) ./ 2 + x_ * use_W2 .+ net.W1[i]
end


function energy(net::MEF3T, x::AbstractMatrix)
    m, n = size(x)
    x = x |> net.array_cast
    energy1 = x * (net.W1 |> net.array_cast)
    energy2 = diag((x * net.W2) * x' |> Array) |> net.array_cast

    tmp1 = zeros(m, n, n) |> net.array_cast
    for i in 1:n
        tmp1[:, :, i] = x * net.W3[:, :, i]
    end
    tmp2 = zeros(m, n) |> net.array_cast
    for i in 1:m
        tmp2[i, :] = x[i, :]' * tmp1[i, :, :]
    end
    energy3 = diag((x * tmp2') |> Array) |> net.array_cast

    -(energy1 + energy2 + energy3)
end

