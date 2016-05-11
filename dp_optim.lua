require 'dp'
require 'optim'

torch.setnumthreads(1) -- this speeds up small networks by removing omp overhead

local dataSource = dp.Cifar10{}

-- command line config
local cmd = torch.CmdLine()
cmd:option('-algo', 'sgd', 'optim algorithm: sgd or rprop')
cmd:option('-bs', 64, 'batch size')

params = cmd:parse(arg)
print(params)

local batchSize = params.bs
local optimize;
if params.algo == 'sgd' then
    optimize = optim.sgd
elseif params.algo == 'rprop' then
    optimize = optim.rprop
end

-- define the network
local cnn = nn.Sequential()
cnn:add(nn.SpatialConvolution(dataSource:imageSize('c'), 16, 5, 5, 1, 1, 2, 2))
cnn:add(nn.ReLU())
cnn:add(nn.SpatialMaxPooling(2, 2, 2, 2))
cnn:add(nn.SpatialConvolution(16, 32, 3, 3, 1, 1, 1, 1))
cnn:add(nn.ReLU())
cnn:add(nn.SpatialMaxPooling(2, 2, 2, 2))
cnn:add(nn.SpatialConvolution(32, 64, 3, 3, 1, 1, 1, 1))
cnn:add(nn.ReLU())
cnn:add(nn.SpatialMaxPooling(2, 2, 2, 2))
cnn:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1))
cnn:add(nn.ReLU())
cnn:add(nn.SpatialMaxPooling(2, 2, 2, 2))
outsize = cnn:outside{1, dataSource:imageSize('c'), dataSource:imageSize('h'), dataSource:imageSize('w')}
cnn:add(nn.Collapse(3))
cnn:add(nn.Linear(outsize[2]*outsize[3]*outsize[4], 256))
cnn:add(nn.Linear(256, #dataSource:classes()))
cnn:add(nn.LogSoftMax())
cnn:insert(nn.Convert(dataSource:ioShapes(), 'bchw'), 1)

-- things needed by the optim package
local x, dl_dx = cnn:getParameters()
local optim_state = {learningRate = 0.01, stepsize = 0.01}

-- things needed by the dp package
local train = dp.Optimizer{
    loss = nn.ModuleCriterion(nn.ClassNLLCriterion(), nil, nn.Convert()),
    callback = function(model, report) -- this is where we glue optim and dp together
        feval = function(x_new)
            if x ~= x_new then
                x:copy(x_new)
            end
            return model.err, dl_dx
        end
        optimize(feval, x, optim_state)
        dl_dx:zero()
    end,
    feedback = dp.Confusion(),
    sampler = dp.ShuffleSampler{batch_size = batchSize},
    progress = true
}
local valid = dp.Evaluator{
    feedback = dp.Confusion(),
    sampler = dp.Sampler{batch_size = batchSize}
}
test = dp.Evaluator{
    feedback = dp.Confusion(),
    sampler = dp.Sampler{batch_size = batchSize}
}

local xp = dp.Experiment{
    model = cnn,
    optimizer = train,
    validator = valid,
    tester = test,
    observer = dp.EarlyStopper{
        error_report = {'validator', 'feedback', 'confusion', 'accuracy'},
        maximize = true,
        max_epochs = 10
    },
    random_seed = os.time(),
    max_epoch = 100
}

dl_dx:zero() -- make sure we start with 0 gradient

print(cnn)
xp:verbose(true)
xp:run(dataSource) -- run the experiment

