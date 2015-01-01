module LinearChainCRF

using Optim
import ArrayViews.getindex

export
    Sequence,
    calc_margs_and_logZ,
    suffstats,
    expectedstats,
    loglikelihood,
    gradient,
    params_to_vector,
    vector_to_params,
    get_optim_func

# include("loaddata.jl")

type Sequence
    xl :: AbstractMatrix
    yl :: AbstractVector

    # Don't touch
    _margs :: Union(Nothing, (Array{Float64, 2}, Array{Float64, 3}, Float64))

    Sequence(xl, yl) = new(xl, yl, nothing)
end

function calc_margs_and_logZ(xl::AbstractMatrix, θ::AbstractMatrix, γ::AbstractMatrix, D::Int)
#    println("calc_margs")
    T = size(xl, 2)

    logϕ = reshape(θ, (D, D, 1)) .+ reshape(γ * xl, (1, D, T))
    logϕ[:,:,2] .+= reshape(γ * xl[:,1], (D, 1))

    # Log-sum-exp like trick
    logϕMaxes = maximum(logϕ, (1, 2))
    ϕ = exp(logϕ .- logϕMaxes)

    # The forward and backward messages
    α = ones(D, T)
    β = ones(D, T)

    logZ :: Float64 = sum(logϕMaxes[:,:,2:end])
    for t = 2:T
        unnorm_msg :: Array{Float64, 1} = ϕ[:,:,t]' * α[:,t-1]
        sum_t = sum(unnorm_msg)
        α[:,t] = unnorm_msg / sum_t
        logZ += log(sum_t)
    end

    for t = (T-1):-1:1
        unnorm_msg :: Array{Float64, 1} = ϕ[:,:,t+1] * β[:,t+1]
        β[:,t] = unnorm_msg / sum(unnorm_msg)
    end

    # Single-node marginals
    p_1 :: Array{Float64, 2} = α .* β
    p_1 ./= sum(p_1, 1)
    
    # Pairwise-node marginals
    p_2 :: Array{Float64, 3} = (ϕ[:,:,2:T]
                                .* reshape(α[:,1:(T-1)], (D, 1, T-1))
                                .* reshape(β[:,2:T], (1, D, T-1)))
    p_2 ./= sum(p_2, (1, 2))

    p_1, p_2, logZ
end

function _clear_margs!(seq :: Sequence)
#    println("clear margs")
    seq._margs = nothing
end

# function _calc_margs!(seq :: Sequence)
    
# end

function _get_margs!(seq :: Sequence, θ, γ, D)
    if seq._margs == nothing
        seq._margs = calc_margs_and_logZ(seq.xl, θ, γ, D)
    end
    seq._margs
end

function suffstats(data :: AbstractVector{Sequence}, K::Int, D::Int)
    function helper(seq)
#        println("ss")
        xl, yl = (seq.xl, seq.yl)
        ymat = hcat([yl .== i for i = 1:D]...)
        θ_ss_part = ymat[1:(end-1),:]' * ymat[2:end,:]
        γ_ss_part = (xl * ymat)'
        θ_ss_part, γ_ss_part
    end
    
    θ_ss, γ_ss = [zip(map(helper, data)...)...]
    sum(θ_ss), sum(γ_ss)
end

function expectedstats(data :: AbstractVector{Sequence}, θ, γ, K::Int, D::Int)
#=
    θ_es = zeros(D, D)
    γ_es = zeros(D, K)
    for seq = data
        xl = seq.xl
        p_1, p_2, _ = _get_margs!(seq, θ, γ, D)
        θ_es += sum(p_2, 3)
        γ_es += p_1 * xl'
    end
    θ_es, γ_es
=#

    function helper(seq)
#        println("es")
        xl = seq.xl
        p_1, p_2, _ = _get_margs!(seq, θ, γ, D)
        θ_es_part = sum(p_2, 3)
        γ_es_part = p_1 * xl'
        θ_es_part, γ_es_part
    end

    θ_es, γ_es = [zip(map(helper, data)...)...]
    sum(θ_es), sum(γ_es)
end

function logprob(seq :: Sequence,
                 θ :: Array{Float64, 2},
                 γ :: Array{Float64, 2},
                 D :: Int)
    println("logprob")
    xl :: Array{Float64, 2} = seq.xl
    yl :: Array{Int64, 1} = seq.yl
    K, Tl = size(xl)
    
    θ_term :: Float64 = 0.0
    for t = 2:Tl
        θ_term += θ[yl[t-1], yl[t]]
    end
    
    γ_term :: Float64 = 0.0

#    d_time :: Float64 = 0.0
#    a_time :: Float64 = 0.0
#    b_time :: Float64 = 0.0
#    mult_time :: Float64 = 0.0
    @time for k :: Int64 = 1:K, t :: Int64 = 1:Tl
#        s = time()
        #@inbounds d :: Int64 = yl[t]
#        d_time += time() - s

#        s = time()
        #@inbounds a :: Float64 = γ[d,k]
#        a_time += time() - s

#        s = time()
        #@inbounds b :: Float64 = xl[k,t]
#        b_time += time() - s

#        s = time()
        #γ_term += a * b
#        mult_time += time() - s

        @inbounds γ_term += γ[yl[t],k] * xl[k,t]
    end

#    println((d_time, a_time, b_time, mult_time))

    logZ = _get_margs!(seq, θ, γ, D)[3]

    println("done logprob")
    θ_term + γ_term - logZ
end

function loglikelihood(data, θ, γ, D)
    sum(map(seq -> logprob(seq, θ, γ, D), data))
end

function ll_gradient(data, θ, γ, K, D; _suffstats = ())
    θ_ss, γ_ss = if length(_suffstats) > 0
        _suffstats
    else
        suffstats(data, K, D)
    end

    θ_es, γ_es = expectedstats(data, θ, γ, K, D)
    return θ_ss - θ_es, γ_ss - γ_es
end

params_to_vector(θ, γ) = [θ[:], γ[:]]

function vector_to_params(vec, K, D)
    θ = reshape(vec[1:D*D], (D, D))
    γ = reshape(vec[(D*D+1):end], (D, K))
    θ, γ
end

function get_optim_func(data, K, D, λ)
    L = length(data)
    ss = suffstats(data, K, D)
    
    # TODO fix nll, grad!
    function nll(vec::Vector)
#        println("nll")
        θ, γ = vector_to_params(vec, K, D)
        margs = {l => calc_margs_and_logZ(x[l], θ, γ, D) for l = 1:L}
        return -1.0 * loglikelihood(x, y, θ, γ, D, _margs = margs)
    end

    function grad!(vec::Vector, grad_storage)
#        println("grad!")
        θ, γ = vector_to_params(vec, K, D)
        margs = {l => calc_margs_and_logZ(x[l], θ, γ, D) for l = 1:L}
        θ_grad, γ_grad = ll_gradient(x, y, θ, γ, K, D, _suffstats = ss, _margs = margs)
        copy!(grad_storage, -1.0 * params_to_vector(θ_grad, γ_grad))
    end

    function nll_and_grad!(vec::Vector, grad_storage)
        println("nll_and_grad!")
        θ, γ = vector_to_params(vec, K, D)
        
        # Clear the old _margs
        map(_clear_margs!, data)

#        margs = {l => calc_margs_and_logZ(x[l], θ, γ, D) for l = 1:L}
#        margs = Dict(pmap((l) -> (l, calc_margs_and_logZ(x[l], θ, γ, D)), 1:L))
#        margs = map((dat) -> calc_margs_and_logZ(dat[1], θ, γ, D), data)

        θ_grad, γ_grad = @time ll_gradient(data, θ, γ, K, D; _suffstats = ss)
        grad = params_to_vector(θ_grad, γ_grad)
        copy!(grad_storage, -1.0 * (grad - 2 * λ * vec))

        ret = -1.0 * (@time loglikelihood(data, θ, γ, D) - λ * sum(vec .^ 2))
        println("done nll_and_grad!")
        ret
    end
    
    return DifferentiableFunction(nll, grad!, nll_and_grad!)
end

end
