module LinearChainCRF

using Optim
import ArrayViews.getindex

export
    calc_margs_and_logZ,
    suffstats,
    expectedstats,
    loglikelihood,
    gradient,
    params_to_vector,
    vector_to_params,
    get_optim_func

# include("loaddata.jl")

function calc_margs_and_logZ(xl::AbstractMatrix, θ::AbstractMatrix, γ::AbstractMatrix, D::Int)
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

function suffstats(data, K::Int, D::Int)
#    L = length(x)
    θ_ss = zeros(D, D)
    γ_ss = zeros(D, K)
    for (xl, yl) = data
        ymat = hcat([yl .== i for i = 1:D]...)
        θ_ss += ymat[1:(end-1),:]' * ymat[2:end,:]
        γ_ss += (xl * ymat)'
    end
    θ_ss, γ_ss
end

function expectedstats(data, θ, γ, K::Int, D::Int; _margs = [])
#    println("expectedstats")
    θ_es = zeros(D, D)
    γ_es = zeros(D, K)
    for (l, (xl, _)) = enumerate(data)
        p_1, p_2, _ = if length(_margs) > 0
            _margs[l]
        else
#            println("expectedstats: recalc margs")
            calc_margs_and_logZ(xl, θ, γ, D)
        end
        θ_es += sum(p_2, 3)
        γ_es += p_1 * xl'
    end
#    println("done expectedstats")
    θ_es, γ_es
end

function logprob(xl, yl, θ, γ, D; _logZ = -1.0)
    K, Tl = size(xl)
    
    θ_term :: Float64 = 0.0
    for t = 2:Tl
        θ_term += θ[yl[t-1], yl[t]]
    end
    
    γ_term :: Float64 = 0.0
    for t = 1:Tl, k = 1:K
        γ_term += γ[yl[t],k] * xl[k,t]
    end

    logZ = if _logZ >= 0.0
        _logZ
    else
        calc_margs_and_logZ(xl, θ, γ, D)[3]
    end
    
    θ_term + γ_term - logZ
end

function loglikelihood(data, θ, γ, D; _margs = [])
    if length(_margs) > 0
        sum(map(enumerate(data)) do el
            (l, (xl, yl)) = el
            logprob(xl, yl, θ, γ, D; _logZ = _margs[l][3])
        end)
    else
        sum(map(enumerate(data)) do el
            (l, (xl, yl)) = el
            logprob(xl, yl, θ, γ, D)
        end)
    end
end

function ll_gradient(data, θ, γ, K, D; _suffstats = (), _margs = [])
    θ_ss, γ_ss = if length(_suffstats) > 0
        _suffstats
    else
        suffstats(data, K, D)
    end

    θ_es, γ_es = expectedstats(data, θ, γ, K, D; _margs = _margs)
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
#        println("nll_and_grad! $(myid())")
        θ, γ = vector_to_params(vec, K, D)
#        margs = {l => calc_margs_and_logZ(x[l], θ, γ, D) for l = 1:L}
#        margs = Dict(pmap((l) -> (l, calc_margs_and_logZ(x[l], θ, γ, D)), 1:L))
        margs = map((dat) -> calc_margs_and_logZ(dat[1], θ, γ, D), data)

        θ_grad, γ_grad = ll_gradient(data, θ, γ, K, D; _suffstats = ss, _margs = margs)
        grad = params_to_vector(θ_grad, γ_grad)
        copy!(grad_storage, -1.0 * (grad - 2 * λ * vec))

        return -1.0 * (loglikelihood(data, θ, γ, D; _margs = margs) - λ * sum(vec .^ 2))
    end
    
    return DifferentiableFunction(nll, grad!, nll_and_grad!)
end

end
