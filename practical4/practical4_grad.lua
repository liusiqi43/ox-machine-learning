---------------------------------------------------------------------------------------
-- Practical 4 - Learning to use different optimizers with logistic regression
--
-- to run: th -i practical3.lua
-- or:     luajit -i practical3.lua
---------------------------------------------------------------------------------------

require 'torch'
require 'math'
require 'nn'
require 'optim'
require 'gnuplot'
require 'dataset-mnist'

------------------------------------------------------------------------------
-- INITIALIZATION AND DATA
------------------------------------------------------------------------------

torch.manualSeed(1)    -- fix random seed so program runs the same every time

-- TODO: play with these optimizer options for the second handin item, as described in the writeup
-- NOTE: see below for optimState, storing optimiser settings
local opt = {}         -- these options are used throughout
opt.optimization = 'sgd'
opt.batch_size = 256
opt.train_size = 0 -- set to 0 or 60000 to use all 60000 training data
opt.test_size = 0      -- 0 means load all data
opt.epochs = 5        -- **approximate** number of passes through the training data (see below for the `iterations` variable, which is calculated from this)

-- NOTE: the code below changes the optimization algorithm used, and its settings
local optimState       -- stores a lua table with the ]ptimization algorithm's settings, and state during iterations
local optimMethod      -- stores a function corresponding to the optimization routine
-- remember, the defaults below are not necessarily good
if opt.optimization == 'lbfgs' then
  optimState = {
    learningRate = 1e-1,
    maxIter = 2,
    nCorrection = 10
  }
  optimMethod = optim.lbfgs
elseif opt.optimization == 'sgd' then
  optimState = {
    learningRate = 1e-1,
    weightDecay = 1e-1, -- lambda
    momentum = 1e-3,
    learningRateDecay = 1e-7
  }
  optimMethod = optim.sgd
elseif opt.optimization == 'adagrad' then
  optimState = {
    learningRate = 1e-1,
  }
  optimMethod = optim.adagrad
else
  error('Unknown optimizer')
end

mnist.download()       -- download dataset if not already there

-- load dataset using dataset-mnist.lua into tensors (first dim of data/labels ranges over data)
local function load_dataset(train_or_test, count)
    -- load
    local data
    if train_or_test == 'train' then
        data = mnist.loadTrainSet(count, {32, 32})
    else
        data = mnist.loadTestSet(count, {32, 32})
    end

    -- shuffle the dataset
    local shuffled_indices = torch.randperm(data.data:size(1)):long()
    -- creates a shuffled *copy*, with a new storage
    data.data = data.data:index(1, shuffled_indices):squeeze()
    data.labels = data.labels:index(1, shuffled_indices):squeeze()

    -- TODO: (optional) UNCOMMENT to display a training example
    -- for more, see torch gnuplot package documentation:
    -- https://github.com/torch/gnuplot#plotting-package-manual-with-gnuplot
    --gnuplot.imagesc(data.data[10])

    -- vectorize each 2D data point into 1D
    data.data = data.data:reshape(data.data:size(1), 32*32)

    print('--------------------------------')
    print(' loaded dataset "' .. train_or_test .. '"')
    print('inputs', data.data:size())
    print('targets', data.labels:size())
    print('--------------------------------')

    return data
end

local train = load_dataset('train', opt.train_size)
local test = load_dataset('test', opt.test_size)

------------------------------------------------------------------------------
-- MODEL
------------------------------------------------------------------------------

local n_train_data = train.data:size(1) -- number of training data
local n_inputs = train.data:size(2)     -- number of cols = number of dims of input
local n_outputs = train.labels:max()    -- highest label = # of classes

print('train.labels:max()' .. train.labels:max())
print('train.labels:min()' .. train.labels:min())

local lin_layer = nn.Linear(n_inputs, n_outputs)
local model = nn.Sequential()
model:add(lin_layer)


------------------------------------------------------------------------------
-- LOSS FUNCTION
------------------------------------------------------------------------------
criterion = nn.MultiMarginCriterion()

------------------------------------------------------------------------------
-- TRAINING
------------------------------------------------------------------------------

local parameters, gradParameters = model:getParameters()

------------------------------------------------------------------------
-- Define closure with mini-batches 
------------------------------------------------------------------------

local counter = 0
local feval = function(x)

  if x ~= parameters then
    parameters:copy(x)
  end

  -- get start/end indices for our minibatch (in this code we'll call a minibatch a "batch")
  --           ------- 
  --          |  ...  |
  --        ^ ---------<- start index = i * batchsize + 1
  --  batch | |       |
  --   size | | batch |       
  --        v |   i   |<- end index (inclusive) = start index + batchsize
  --          ---------                         = (i + 1) * batchsize + 1
  --          |  ...  |                 (except possibly for the last minibatch, we can't 
  --          --------                   let that one go past the end of the data, so we take a min())
  local start_index = counter * opt.batch_size + 1
  local end_index = math.min(n_train_data, (counter + 1) * opt.batch_size + 1)
  if end_index == n_train_data then
    counter = 0
  else
    counter = counter + 1
  end

  local batch_inputs = train.data[{{start_index, end_index}, {}}]
  local batch_targets = train.labels[{{start_index, end_index}}]
  -- reset gradients for each minibatch.
  gradParameters:zero()

  -- In order, these lines compute:
  -- 1. compute outputs (log probabilities) for each data point
  local batch_outputs = model:forward(batch_inputs)

  local batch_losses = torch.Tensor(batch_outputs:size()):zero()
  local dloss_doutputs = torch.Tensor(batch_outputs:size()):zero()
  for i = 1, batch_outputs:size(1) do
    -- 2. compute the loss of these outputs, measured against the true labels in batch_target
    batch_losses[i] = criterion:forward(batch_outputs[i], batch_targets[i])
    -- 3. compute the derivative of the loss wrt the outputs of the model
    dloss_doutputs[i] = criterion:backward(batch_outputs[i], batch_targets[i])
  end
  -- 4. use gradients to update weights, we'll understand this step more next week
  model:backward(batch_inputs, dloss_doutputs)
  -- add lambda w term in the gradient.
  -- gradParameters:add(lambda, parameters)


  
  -- optim expects us to return
  --     loss, (gradient of loss with respect to the weights that we're optimizing)
  return batch_losses, gradParameters
end

function eval(inputs, targets)
  local outputs = model:forward(inputs)
  local _, predicted_labels = outputs:max(2)
  predicted_labels = torch.squeeze(predicted_labels):double()
  local misclass = 1 - predicted_labels:eq(targets:double()):sum() / predicted_labels:size(1)
  return misclass
end
  
------------------------------------------------------------------------
-- OPTIMIZE: FIRST HANDIN ITEM
------------------------------------------------------------------------

function gradient_checker(batch_inputs, batch_targets, gradParameters)
    local h = 1e-12
    local eps = gradParameters:size(1) * 1e-4

    local grad = {}
    for i = 1, parameters:size(1) do
        local cur = parameters[i]
        parameters[i] = cur + h
        local batch_loss_upper = criterion:forward(model:forward(batch_inputs), batch_targets)
        parameters[i] = cur - h
        local batch_loss_lower = criterion:forward(model:forward(batch_inputs), batch_targets)
        grad[i]  = (batch_loss_upper - batch_loss_lower) / (2 * h)
        -- reset
        parameters[i] = cur
    end
    local diff_norm = (gradParameters - torch.Tensor(grad)):norm()
    if diff_norm > eps then
        print('[gradient checker]: diff_norm with eps ' .. eps .. ' exceeded ' .. diff_norm)
    else
        print('[gradient checker]: passed (' .. diff_norm .. ' < ' .. eps .. ')')
    end
end


local losses = {}          -- training losses for each iteration/minibatch
local eval_iter = {}          -- test losses for each iteration/minibatch
local test_losses = {}          -- test losses for each iteration/minibatch
local train_misclass = {}
local test_misclass = {}
local epochs = opt.epochs  -- number of full passes over all the training data
local iterations = epochs * math.ceil(n_train_data / opt.batch_size) -- integer number of minibatches to process
-- (note: number of training data might not be divisible by the batch size, so we round up)

-- In each iteration, we:
--    1. call the optimization routine, which
--      a. calls feval(parameters), which
--          i. grabs the next minibatch
--         ii. returns the loss value and the gradient of the loss wrt the parameters, evaluated on the minibatch
--      b. the optimization routine uses this gradient to adjust the parameters so as to reduce the loss.
--    3. then we append the loss to a table (list) and print it
for i = 1, iterations do
  -- optimMethod is a variable storing a function, either optim.sgd or optim.adagrad or ...
  -- see documentation for more information on what these functions do and return:
  --   https://github.com/torch/optim
  -- it returns (new_parameters, table), where (TODO) there is table[0]? table[0] is the value of the function being optimized
  -- and we can ignore new_parameters because `parameters` is updated in-place every time we call 
  -- the optim module's function. It uses optimState to hide away its bookkeeping that it needs to do
  -- between iterations.
  local _, minibatch_loss = optimMethod(feval, parameters, optimState)

  -- TIP: use this same idea of not saving the test loss in every iteration if you want to increase speed.
  -- Then you can get, 10 (for example) times fewer values than the training loss. If you do this,
  -- you just have to be careful to give the correct x-values to the plotting function, rather than
  -- Tensor{1,2,...,#losses}. HINT: look up the torch.linspace function, and note that torch.range(1, #losses)
  -- is the same as torch.linspace(1, #losses, #losses).

  losses[#losses + 1] = minibatch_loss[1] -- append the new loss

  -- Our loss function is cross-entropy, divided by the number of data points,
  -- therefore the units (units in the physics sense) of the loss is "loss per data sample".
  -- Since we evaluate the loss on a different minibatch each time, the loss will sometimes 
  -- fluctuate upwards slightly (i.e. the loss estimate is noisy).
  if i % 10 == 0 then -- don't print *every* iteration, this is enough to get the gist
      test_misclass[#test_misclass + 1] = eval(test.data, test.labels)
      train_misclass[#train_misclass + 1] = eval(train.data, train.labels)
      eval_iter[#eval_iter + 1] = i
      print(string.format("minibatches processed: %6s, test_misclass: %6s", i, test_misclass[#test_misclass]))
  end
end

gnuplot.figure(1)
gnuplot.title('misclassification')
gnuplot.plot(
    {'train-misclass', torch.Tensor(eval_iter), torch.Tensor(train_misclass), '-'},
    {'test-misclass', torch.Tensor(eval_iter), torch.Tensor(test_misclass), '-'}
)
