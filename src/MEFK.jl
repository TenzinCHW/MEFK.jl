module MEFK
    import Combinatorics: combinations
    import Flux: @functor
    include("tensor_mefk.jl")


    abstract type MPNK end


    function dynamics(net::MPNK, x::AbstractMatrix{Int8}; array_cast=Array,
        iterate_nodes=nothing)
        m, n = size(x)
        x_ = x[:, :]
        iterate_nodes = isnothing(iterate_nodes) ? (1:n) : iterate_nodes
        for i in iterate_nodes
            en = (ones(m) |> array_cast) .* net.W[1][i]
            for (order, (inds, winds)) in enumerate(zip(net.indices[i], net.windices[i]))
                mult = prod_bits(x_, inds)
                en += mult * net.W[order+1][winds] .* factorial(order+1)
            end
            x_[:, i] = en .>= 0
        end
        x_
    end


    # TODO define the probability distribution for MPNK
    function energy(net::MPNK, x::AbstractMatrix{Int8}; array_cast=Array)
        x = x |> array_cast
        en = x * (net.W[1] |> array_cast)
        for i in 1:net.n
            for (order, (inds, winds)) in enumerate(zip(net.indices[i], net.windices[i]))
                mult = x[:, i] .* prod_bits(x, inds)
                en += mult * net.W[order+1][winds] .* factorial(order+1)
            end
        end
        -en
    end


    function energy2prob(energies)
        nenergies = -energies
        norm_nen = nenergies .- max(nenergies...)
        ex_en = exp.(norm_nen)
        Z = sum(ex_en)
        ex_en ./ Z
    end


    struct MEFMPNK{F<:AbstractFloat}<:MPNK
        n::Int
        N::Int
        W::Vector{AbstractVector{F}}
        grad::Vector{AbstractVector{F}}
        indices::Vector{Vector{AbstractMatrix{Int}}}
        windices::Vector{Vector{AbstractVector{Int}}}

        function MEFMPNK(n::Int, N::Int, W, grad, indices, windices)
            F = Float32
            new{F}(n, N, W, grad, indices, windices)
        end

        function MEFMPNK(n::Int, N::Int, W, indices, windices)
            grad = [w[:] for w in W]
            MEFMPNK(n, N, W, grad, indices, windices)
        end

        """Creates all weights"""
        function MEFMPNK(n::Int, N::Int, dtype::D=Float32; array_cast=Array) where {D<:DataType}
            inds, winds = make_inds_winds(n, N)
            inds = [[i |> array_cast for i in ind] for ind in inds]
            winds = [[wi |> array_cast for wi in wind] for wind in winds]
            W = DenseArray[zeros(dtype, binomial(n, i)) |> array_cast for i in 2:N]
            pushfirst!(W, zeros(dtype, n))
            MEFMPNK(n, N, W, inds, winds)
        end

        """Given a set of indices, create weights and windices"""
        function MEFMPNK(n::Int, N::Int, inds::Vector{Vector{Matrix{Int}}}, dtype::D=Float32; array_cast=Array) where {D<:DataType}
            # collect unique inds in each order and create Dict, index into it to make winds
            orig_inds_byorder = [[[push!(val |> Array, i) |> sort for val in eachslice(ind[ord], dims=1)] for (i, ind) in enumerate(inds)] for ord in 1:N-1]
            unique_inds = [Iterators.flatten(uind) |> collect |> unique |> Array
                           for uind in orig_inds_byorder]
            W = DenseArray[zeros(dtype, length(uind)) |> array_cast for uind in unique_inds]
            pushfirst!(W, zeros(dtype, n))
            ind2wind = [Dict(uindi=>i for (i, uindi) in enumerate(uind)) for uind in unique_inds]
            orig_inds = zip(orig_inds_byorder...) |> collect
            winds = [[[ind2wind[ord][indordi] for indordi in indord] |> array_cast for (ord, indord) in enumerate(ind)] for ind in orig_inds]
            inds = [[i |> array_cast for i in ind] for ind in inds]
            MEFMPNK(n, N, W, inds, winds)
        end
    end


    @functor MEFMPNK


    function make_inds_winds(n::Int, N::Int)
        @assert 0 < N <= n
        combs = [collect(enumerate(combinations(1:n, i))) for i in 1:N]
        indices = [[filter(x->!(j in x[2]), comb) for comb in combs[1:end-1]] for j in 1:n]
        windices = [[filter(x->j in x[2], comb) for comb in combs[2:end]] for j in 1:n]
        [[reduce(vcat, transpose.([i[2] for i in ind])) for ind in inds] for inds in indices],
        [[Int.([wo[1] for wo in wind]) for wind in winds] for winds in windices]
    end


    function prod_bits(x::AbstractMatrix{Int8}, inds)
        xv = @view x[:, inds]
        mult = prod(xv, dims=3)
        mult[:, :, 1]
    end


    function (net::MEFMPNK)(x::AbstractMatrix{Int8}, first_iter::Bool=false; array_cast=Array, iterate_nodes=nothing)
        sz = size(x)[1]
        counts = ones(sz)
	    net(x, counts, first_iter; array_cast=array_cast, iterate_nodes=iterate_nodes)
    end


    function (net::MEFMPNK)(x::AbstractMatrix{Int8}, counts::AbstractVector, first_iter::Bool=false; array_cast=Array, iterate_nodes=nothing)
	    # compute objective
        obj = 0
        x = x |> array_cast
        counts = counts |> array_cast
        counts ./= sum(counts)
        for g in net.grad
            fill!(g, 0)
        end
        grad_inp = []
        iterate_nodes = isnothing(iterate_nodes) ? (1:net.n) : iterate_nodes
        for i in iterate_nodes
            flip = 1 .- 2 .* x[:, i]

            if first_iter
                obj_i = ones(size(counts)...) |> array_cast
            else
            # TODO maybe make this loop async? Then grads need to be synced/replicated
                obj_i = mef_obj(net, x, flip, i, counts; per_bit=true)
            end
            push!(grad_inp, (obj_i, x[:, :], flip, i))
            obj += sum(obj_i)
        end

        for gr in grad_inp
            gradient(net, gr..., counts)
        end

        @sync for (i, g) in enumerate(net.grad)
            @async g ./= factorial(i)
        end
        grads = (n=nothing, N=nothing, W=net.grad, grad=nothing, indices=nothing, windices=nothing)
        obj, grads
    end


    function compute_grad_order(x, flip, inds, a)
        xtnx = flip .* prod_bits(x, inds)
        sum(a .* xtnx, dims=1)[1, :]
    end


    function compute_energy_contrib(net::MEFMPNK, x, flip, inds, winds, order)
        xtnx = flip .* prod_bits(x, inds)
        xtnx * net.W[order+1][winds] .* factorial(order+1)
    end


    function mef_obj(net::MEFMPNK, x, flip, i, counts; per_bit=false)
        energy = Any[flip .* net.W[1][i]]
        for _ in 2:net.N
            push!(energy, 0)
        end
        @sync for (order, (inds, winds)) in enumerate(zip(net.indices[i], net.windices[i]))
            # TODO this is the slowest line in the loop. Need to hide the latency to transfer data into contiguous arrays
            energy[order+1] = compute_energy_contrib(net, x, flip, inds, winds, order)
        end
        energy = sum(energy)

        energy .*= counts
        obj = exp.(clamp.(energy, -10, 10))
        if per_bit
            return obj
        else
            return sum(obj)
        end
    end


    function gradient(net::MEFMPNK, obj_bit, x, flip, i, counts)
        a = obj_bit .* counts
        @sync for (g, inds, winds) in zip(net.grad[2:end], net.indices[i], net.windices[i])
            @async begin
                gwv = @view g[winds]
                gwv .+= compute_grad_order(x, flip, inds, a)
            end
        end

        net.grad[1][i] += sum(flip .* a)
    end


    function obj_gradient(net::MEFMPNK, x, flip, i, counts)
        tmp = mef_obj(net, x, flip, i, counts; per_bit=true)
        gradient(net, tmp, x, flip, i, counts)
        return sum(tmp)
    end
end
