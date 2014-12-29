using LinearChainCRF
import NumericExtensions.logsumexp
using Base.Test
using Base.Cartesian

function _brute_unnorm_logprob(xl, yl, θ, γ)
    T = size(xl, 2)

    logprob = 0.0
    for t = 1:(T-1)
        logprob += θ[yl[t], yl[t+1]]
    end

    for t = 1:T
        logprob += (γ[yl[t],:] * xl[:,t])[1]
    end
    logprob
end

function _brute_logZ(xl, θ, γ, K)

end

function test_calc_margs_and_logZ(Ntests = 50)
    function _brute(xl::AbstractMatrix, θ::AbstractMatrix, γ::AbstractMatrix, D::Int, K::Int)
        Tl = size(xl, 2)
        latent_states = map(x -> (yl = 1 + digits(x, D, Tl);
                                  lp = _brute_unnorm_logprob(xl, yl, θ, γ);
                                  (yl, lp)),
                            0:(D^Tl - 1)) :: Array{(Array{Int,1},Float64),1}

        logZ = logsumexp([lp for (_, lp) in latent_states])
#        logZ = log(sum(exp([lp for (_, lp) in latent_states])))
        
        p_1 = zeros(D, Tl)
        for t = 1:Tl
            for d = 1:D
                lps = [lp - logZ for (yl, lp) in filter(a -> ((yl, lp) = a; yl[t] == d), latent_states)]
                p_1[d,t] = sum(exp(lps))
            end
        end

        p_2 = zeros(D, D, Tl - 1)
        for t = 1:(Tl - 1)
            for i = 1:D, j = 1:D
                lps = [lp - logZ for (yl, lp) in filter(a -> ((yl, lp) = a; yl[t] == i && yl[t+1] == j), latent_states)]
                p_2[i,j,t] = sum(exp(lps))
            end
        end

        p_1, p_2, logZ
    end

    for i = 1:Ntests
        K = rand(5:10)
        D = rand(2:5)
        T = rand(3:5)
        xl = randn(K, T)
        θ = randn(D, D)
        γ = randn(D, K)
        
        bf_p_1, bf_p_2, bf_logZ = _brute(xl, θ, γ, D, K)
        dp_p_1, dp_p_2, dp_logZ = calc_margs_and_logZ(xl, θ, γ, D)

        @test_approx_eq_eps sum(abs(bf_p_1 .- dp_p_1)) 0.0 1e-10
        @test_approx_eq_eps sum(abs(bf_p_2 .- dp_p_2)) 0.0 1e-10
        @test_approx_eq bf_logZ dp_logZ
    end
end

function test_suffstats()
    K = rand(5:10)
    D = rand(2:5)
    T = rand(3:5)
    xl = randn(K, T)
    yl = rand(1:D, T)

    suffstats(Array[xl], Array[yl], K, D)
end

test_calc_margs_and_logZ()
test_suffstats()
