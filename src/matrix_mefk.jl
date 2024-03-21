import Flux
import Combinatorics, LinearAlgebra

struct MEF2T{F<:AbstractFloat}
    n::Int
    W1::AbstractVector{F}
    W2::AbstractMatrix{F}
    W2_mask::AbstractMatrix{F}
    gradients::AbstractVector
    array_cast


    function MEF2T(n::Int, W1, W2, mW2, gradients, array_cast)
        new{Float32}(n, W1, W2, mW2, gradients, array_cast)
    end

    function MEF2T(n::Int, W1, W2; array_cast=Array)
        W2_mask = (ones(n, n) - LinearAlgebra.I) |> array_cast
        gradients = [
            zeros(n),
            zeros(n, n) |> array_cast]
        MEF2T(n, W1, W2, W2_mask, gradients, array_cast)
    end

    function MEF2T(n::Int; array_cast=Array)
        W1 = zeros(n)
        W2 = zeros(n, n) |> array_cast
        MEF2T(n, W1, W2; array_cast=array_cast)
    end
end


Flux.@functor MEF2T


#function diag(a::AbstractMatrix)
#    [a[i, i] for i in 1:size(a)[1]]
#end


#function adjust_W2(net::MEF2T, symmetrize::Bool)
#    g2 = net.gradients[2][:, :]
#    if symmetrize
#        g2 = (g2 + g2') ./ 2
#    end
#    g2 .* net.W2_mask
#end


function retrieve_reset_gradients!(net::MEF2T; symmetrize=true, reset_grad=false)
    g1 = net.gradients[1][:]
    g2 = adjust_W2(net, symmetrize)

    if reset_grad
        net.gradients[1] .= 0
        net.gradients[2] .= 0
    end
    (n=nothing, W1=g1, W2=g2, W2_mask=nothing, gradients=nothing, array_cast=nothing)
end


function (net::MEF2T)(x::AbstractMatrix; iterate_nodes=nothing, reset_grad=false)
    sz = size(x)[1]
    counts = ones(sz)
    net(x, counts; iterate_nodes=iterate_nodes, reset_grad=reset_grad)
end


function (net::MEF2T)(x::AbstractMatrix, counts::AbstractVector;
                      iterate_nodes=nothing, symmetrize_grad=true, reset_grad=false)
    x = x |> net.array_cast
    W1 = net.W1
    W2 = net.W2
    loss = 0
    counts = counts |> net.array_cast
    iterate_nodes = isnothing(iterate_nodes) ? (1:net.n) : iterate_nodes
    for i in iterate_nodes
        flip = 1 .- 2 .* x[:, i]
        use_W1 = W1[i]
        use_W2 = W2[i, :]
        bit_obj = mef_obj(use_W1, use_W2, x, flip, counts)
        loss += sum(bit_obj)

        flip_bit_obj = flip .* bit_obj .* counts
        net.gradients[1][i] += sum(flip_bit_obj)
        net.gradients[2][i, :] += x' * flip_bit_obj
    end

    return loss, retrieve_reset_gradients!(net; symmetrize=symmetrize_grad, reset_grad=reset_grad)
end


function mef_obj(W1, W2, x, flip, counts)
    #out = net.W1[pos] * flip .+ x * use_W2 .* flip + sum((x * use_W3) .* x .* flip, dims=2)

    #tmp1 = sum((x * W3) .* x, dims=2)
    out = flip .* (x * W2 .+ W1)
    exp.(clamp.(out .* counts, -10, 10))
end


function dynamics(net::MEF2T, x::AbstractMatrix; iterate_nodes=nothing)
    x_ = x[:, :] |> net.array_cast
    iterate_nodes = isnothing(iterate_nodes) ? (1:net.n) : iterate_nodes
    for i in iterate_nodes
        out = dyn_step(net, x_, i)
        x_[:, i] .= out .> 0
    end
    x_
end


function dyn_step(net::MEF2T, x_, i)
    use_W2 = net.W2[i, :]
    x_ * use_W2 .+ net.W1[i]
end


function energy(net::MEF2T, x::AbstractMatrix)
    x = x |> net.array_cast
    energy1 = x * (net.W1 |> net.array_cast)
    energy2 = diag((x * net.W2) * x' |> Array) |> net.array_cast

    -(energy1 + energy2)
end

