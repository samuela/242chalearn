using MAT
using ArrayViews

glob(regex, path) = [joinpath(path, fn) for fn in filter((fn) -> ismatch(regex, fn), readdir(path))]

train_files = glob(r".*\.mat", "/users/skainswo/data/skainswo/chalearn/train")
valid_files = glob(r".*\.mat", "/users/skainswo/data/skainswo/chalearn/validation")

# flatten(a::AbstractArray) = reshape(a, prod(size(a)))

all_labels = ["*NONE*",
              "vattene",
              "vieniqui",
              "perfetto",
              "furbo",
              "cheduepalle",
              "chevuoi",
              "daccordo",
              "seipazzo",
              "combinato",
              "freganiente",
              "ok",
              "cosatifarei",
              "basta",
              "prendere",
              "noncenepiu",
              "fame",
              "tantotempo",
              "buonissimo",
              "messidaccordo",
              "sonostufo"]

num_for_label = {l => i for (i, l) in enumerate(all_labels)}

joint_types = ["HipCenter",
               "Spine",
               "ShoulderCenter",
               "Head",
               "ShoulderLeft",
               "ElbowLeft",
               "WristLeft",
               "HandLeft",
               "ShoulderRight",
               "ElbowRight",
               "WristRight",
               "HandRight",
               "HipLeft",
               "KneeLeft",
               "AnkleLeft",
               "FootLeft",
               "HipRight",
               "KneeRight",
               "AnkleRight",
               "FootRight"]

num_for_joint = {l => i for (i, l) in enumerate(joint_types)}

function loadfile(fn)
    mat = matread(fn)
    frames = mat["Video"]["Frames"]["Skeleton"]
    xl = hcat([begin
               wp = frame["WorldPosition"]
               wr = frame["WorldRotation"]
               wp_hipped = (wp .- wp[1,:])[2:end,:]
               wr_hipped = (wr .- wr[1,:])[2:end,:]
               # [flatten(wp_hipped), flatten(wr)]
               [wp_hipped[:], wr_hipped[:]]
               end for frame in frames]...)

    labels = mat["Video"]["Labels"]
    yl = ones(length(frames))
    for (startix, endix, name) = zip(labels["Begin"], labels["End"], labels["Name"])
        yl[startix:endix] = num_for_label[name]
    end
    
    xl, yl
end

function moving_window(arr, n)
    shp = (size(arr, 1) * n, size(arr, 2) - n + 1)
#    strided_view(arr, shp, ArrayViews.contrank(arr), strides(arr))
    copy(strided_view(arr, shp, ArrayViews.contrank(arr), strides(arr)))
end

diffify(dat) = ((xl, yl) = dat; (diff(xl, 2), yl[2:end]))

function windowify(dat, n)
    @assert n % 2 == 0
    (xl, yl) = dat
    a :: Int = n / 2
    moving_window(xl, n), yl[a:end-a]
end
