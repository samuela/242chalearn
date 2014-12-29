using MAT

glob(regex, path) = [joinpath(path, fn) for fn in filter((fn) -> ismatch(regex, fn), readdir(path))]

train_files = glob(r".*\.mat", "/users/skainswo/data/skainswo/chalearn/train")
valid_files = glob(r".*\.mat", "/users/skainswo/data/skainswo/chalearn/validation")

flatten(a::AbstractArray) = reshape(a, prod(size(a)))
trace(a) = (println(a); a)

loadfiles(files) =
    [begin
       # println(fn)
       mat = matread(fn)
       frames = mat["Video"]["Frames"]["Skeleton"]
       hcat([begin
               wp = frame["WorldPosition"]
               wr = frame["WorldRotation"]
               wp_hipped = (wp .- wp[1,:])[2:end,:]
               [flatten(wp_hipped), flatten(wr)]
             end for frame in frames]...)
     end for fn in files]
