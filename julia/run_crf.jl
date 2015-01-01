using Optim
# @everywhere using LazyMap
@everywhere using LinearChainCRF
@everywhere include("loaddata.jl")
@everywhere include("arrayviewsfix.jl")

λ = 1.0
maxiter = 500
@everywhere window_size = 30

println("... Gathering data")
train_data = map(distribute(train_files[1:25])) do fn
    println("load $fn")
    xl, yl = windowify(diffify(loadfile(fn)), window_size)
    Sequence(xl, yl)
end

#train_x, train_y = [zip(train_data...)...]

# @show typeof(train_x)
# @show typeof(train_y)

# train_x = LazyMapping(train_data, (dat) -> ((xl, yl) = dat; copy(xl)));
# train_y = LazyMapping(train_data, (dat) -> ((xl, yl) = dat; yl));

K = size(train_data[1].xl, 1)
D = length(all_labels)

println("... Training model")
optfunc = get_optim_func(train_data, K, D, λ)
optimize(optfunc, zeros(D*(D+K)), method = :l_bfgs, show_trace = true, iterations = maxiter)
