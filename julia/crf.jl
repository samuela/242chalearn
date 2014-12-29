module LinearChainCRF

export suffstats, calc_margs_and_logZ

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

function suffstats(x, y, K::Int, D::Int)
    L = length(x)
    θ_ss = zeros(D, D)
    γ_ss = zeros(D, K)
    for l = 1:L
        ymat = 1.0 * hcat([y[l] .== i for i = 1:D]...)
        θ_ss += ymat[:,1:(end-1)] * ymat[:,2:end]'
        γ_ss += ymat * x[l]'
    end
    θ_ss, γ_ss
end

function expectedstats(x, θ, γ, K::Int, D::Int, _margs = Dict())
    L = length(x)
    θ_es = zeros(D, D)
    γ_es = zeros(D, K)
    for l = 1:L
        p_1, p_2, _ = get(() -> calc_margs_and_logZ(x[l], θ, γ, D), _margs, l)
        θ_es += p_1 * x[l]'
        γ_es += sum(p_2, 3)
    end
    θ_es, γ_es
end

function loglikelihood(x, y, θ, γ, D, _margs = Dict())
    L = length(x)
    ll = 0.0
    for l = 1:L
        ymat = 1.0 * hcat([y[l] .== i for i = 1:D]...)
        n_ij = ymat[:,1:(end-1)] * ymat[:,2:end]'
        θ_term = sum(θ .* n_ij)
        γ_term = sum((ymat * x[l]') .* γ)
        _, _, logZ = get(() -> calc_margs_and_logZ(x[l], θ, γ, D), _margs, l)
        ll += θ_term + γ_term - logZ
    end
    ll
end

function gradient(x, y, θ, γ, K, D, _suffstats = (), _margs = Dict())
    θ_ss, γ_ss = if length(_suffstats) > 0
        _suffstats
    else
        suffstats(x, y, K, D)
    end

    θ_ss, γ_es = expectedstats(x, θ, γ, K, D, _margs)
    return θ_ss - θ_es, γ_ss - γ_es
end

params_to_vector(θ, γ) = [θ[:], γ[:]]

function vector_to_params(vec, K, D)
    θ = reshape(vec[1:D*D], (D, D))
    γ = reshape(vec[(D*D+1):end], (D, K))
    θ, γ
end

function get_optim_func(x, y, K, D)
    ss = suffstats(x, y, K, D)
    
    function nll(vec::Vector)
        θ, γ = vector_to_params(vec, K, D)
        margs = {l => calc_margs_and_logZ(x[l], θ, γ, D) for l = 1:L}
        return -1.0 * loglikelihood(x, y, θ, γ, D, _margs = margs)
    end

    function grad!(vec::Vector, grad_storage)
        θ, γ = vector_to_params(vec, K, D)
        margs = {l => calc_margs_and_logZ(x[l], θ, γ, D) for l = 1:L}
        θ_grad, γ_grad = gradient(x, y, θ, γ, K, D, _suffstats = ss, _margs = margs)
        copy!(grad_storage, -1.0 * params_to_vector(θ_grad, γ_grad))
    end

    function nll_and_grad!(vec::Vector, grad_storage)
        θ, γ = vector_to_params(vec, K, D)
        margs = {l => calc_margs_and_logZ(x[l], θ, γ, D) for l = 1:L}
        θ_grad, γ_grad = gradient(x, y, θ, γ, K, D, _suffstats = ss, _margs = margs)
        copy!(grad_storage, -1.0 * params_to_vector(θ_grad, γ_grad))
        return -1.0 * loglikelihood(x, y, θ, γ, D, _margs = margs)
    end
    
    return DifferentiableFunction(nll, grad!, nll_and_grad!)
end

end
