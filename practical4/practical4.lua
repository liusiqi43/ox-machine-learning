require 'nn';
require 'svm';
require 'gnuplot';

torch.setnumthreads(4)
print('<torch> set nb threads to ', torch.getnumthreads())

C_searched = {}
G_searched = {}
Valid_acc = {}
Train_acc = {}

-- TODO:
-- Implement a function that takes a training and validation set and a range of
-- parameters C and G The output should be indices i, j, such that C[i], G[i]
-- were the best parameters for accuracy on the validation set.
--
function grid_search(train_data, validation_data, C, G) 
        local best_acc, best_i, best_j = 0, -1, -1
        for i = 1, #C do
                for j = 1, #G do
                        local cmd = '-q -g ' .. G[j] .. ' -c ' .. C[i]
                        print(cmd)
                        local model = libsvm.train(train_data, cmd)
                        local _, acc, _ = libsvm.predict(validation_data, model)
                        if acc[1] > best_acc then
                                best_acc, best_i, best_j = acc[1], i, j
                                print('best_acc: ' .. best_acc)
                        end
                        C_searched[#C_searched + 1] = C[i]
                        G_searched[#G_searched + 1] = G[j]
                        Valid_acc[#Valid_acc + 1] = acc[1]
                        local _, acc, _ = libsvm.predict(train_data, model)
                        Train_acc[#Train_acc + 1] = acc[1]
                end
        end
        return best_i, best_j, best_acc
end

-- Load the datasets
function load_data()
	--TODO:
	-- You can change these sizes if your program is taking too long to run
	small_train_size = 2000
	small_validation_size = 2000
	medium_train_size = 10000
	medium_validation_size = 2000
	large_train_size = 48000
	large_validation_size = 12000

	full_train_data = svm.ascread('practical4-data/train_data')
	full_test_data = svm.ascread('practical4-data/test_data')

	small_train_data = nn.NarrowTable(1, small_train_size):forward(full_train_data)
	small_validation_data = nn.NarrowTable(small_train_size + 1, small_validation_size):forward(full_train_data)

	medium_train_data = nn.NarrowTable(1, medium_train_size):forward(full_train_data)
	medium_validation_data = nn.NarrowTable(medium_train_size + 1, medium_validation_size):forward(full_train_data)

	large_train_data = nn.NarrowTable(1, large_train_size):forward(full_train_data)
	large_validation_data = nn.NarrowTable(large_train_size + 1, large_validation_size):forward(full_train_data)
end

print('Loading data ..')
load_data()
print('Finished loading data.\n\n')

-- Use these to do grid search on the small training and validation sets
-- Set up C_small and G_small
-- C*:2 G*:0.03125 (2^-5) best_acc: 94.85
C_small = {}
G_small = {}
 
C_small[1] = 2^(-5)
G_small[1] = 2^(-13)
for i=2, 10 do
	C_small[i] = C_small[i-1] * 4
 	G_small[i] = G_small[i-1] * 4
end


print('Performing grid search: Large grid; small dataset\n')
local i, j, best_acc = grid_search(small_train_data, small_validation_data, C_small, G_small)
print('C_small*:' .. C_small[i] .. ' G_small*:' .. G_small[j] .. ' best_acc:', best_acc)

tC_searched = torch.Tensor(C_searched):log()
tG_searched = torch.Tensor(G_searched):log()
tValid_acc = torch.Tensor(Valid_acc)
tTrain_acc = torch.Tensor(Train_acc)
gnuplot.figure(1)
gnuplot.xlabel('log(C)')
gnuplot.ylabel('log(G)')
gnuplot.scatter3({'validation', tC_searched, tG_searched, tValid_acc}, 
                 {'train', tC_searched, tG_searched, tTrain_acc})
gnuplot.figprint('plot_grid_1.fig')


-- Use these to do grid search on the medium training and validation sets
C_medium = {}
G_medium = {}
C_medium[1] = C_small[i]/4
G_medium[1] = G_small[j]/4
for i=2, 5 do
	C_medium[i] = C_medium[i-1] * 2
 	G_medium[i] = G_medium[i-1] * 2
end


print('\nPerforming grid search: Medium grid; medium dataset\n')
-- Use these to do grid search on the large training and validation sets
local i, j, best_acc = grid_search(medium_train_data, medium_validation_data, C_medium, G_medium)
print('C_medium*['..i..']:' .. C_medium[i] .. ' G_medium*['..j..']:' .. G_medium[j] .. ' best_acc:', best_acc)


C_large = {}
G_large = {}
C_large[1] = C_medium[i]/1.5
G_large[1] = G_medium[j]/1.5
for i=2, 3 do
	C_large[i] = C_large[i-1] * 1.5
 	G_large[i] = G_large[i-1] * 1.5
end
print('\nPerforming grid search: Small grid; medium dataset\n')
local i, j, best_acc = grid_search(medium_train_data, medium_validation_data, C_large, G_large)
print('C_large*['..i..']:' .. C_large[i] .. ' G_large*['..j..']:' .. G_large[j] .. ' best_acc:', best_acc)

best_C = C_large[i]
best_gamma = G_large[j]

print('\n\nTraining on full training data with best parameters picked by grid search.\n\n')
flags= string.format('-q -c %f -g %f', best_C, best_gamma)
model = libsvm.train(full_train_data, flags)
libsvm.predict(full_test_data, model)

tC_searched = torch.Tensor(C_searched):log()
tG_searched = torch.Tensor(G_searched):log()
tValid_acc = torch.Tensor(Valid_acc)
tTrain_acc = torch.Tensor(Train_acc)
gnuplot.figure(2)
gnuplot.xlabel('log(C)')
gnuplot.ylabel('log(G)')
gnuplot.scatter3({'validation', tC_searched, tG_searched, tValid_acc}, 
                 {'train', tC_searched, tG_searched, tTrain_acc})
gnuplot.figprint('plot_grid_2.fig')


-------------- polynomial kernel ------------------
function poly_grid_search(train_data, validation_data, C) 
        local best_acc, best_i = 0, -1
        for i = 1, #C do
          local cmd = '-q -t 1 -d 2 -c ' .. C[i]
          print(cmd)
          local model = libsvm.train(train_data, cmd)
          local _, acc, _ = libsvm.predict(validation_data, model)
          if acc[1] > best_acc then
                  best_acc, best_i, best_j = acc[1], i, j
                  print('best_acc: ' .. best_acc)
          end
        end
        return best_i, best_acc
end


-- Use these to do grid search on the small training and validation sets
-- Set up C_small and G_small
-- C*:2 G*:0.03125 (2^-5) best_acc: 94.85
C_small = {}
 
C_small[1] = 8
for i=2, 6 do
	C_small[i] = C_small[i-1] * 4
end


print('Performing grid search: Large grid; small dataset\n')
local i, best_acc = poly_grid_search(small_train_data, small_validation_data, C_small)
print('C_small*:' .. C_small[i] .. ' best_acc:', best_acc)

-- Use these to do grid search on the medium training and validation sets
C_medium = {}
C_medium[1] = C_small[i]/4
for i=2, 5 do
	C_medium[i] = C_medium[i-1] * 2
end


print('\nPerforming grid search: Medium grid; medium dataset\n')
-- Use these to do grid search on the large training and validation sets
local i, best_acc = poly_grid_search(medium_train_data, medium_validation_data, C_medium)
print('C_medium*['..i..']:' .. C_medium[i] .. ' best_acc:', best_acc)


C_large = {}
C_large[1] = C_medium[i]/1.5
for i=2, 3 do
	C_large[i] = C_large[i-1] * 1.5
end
print('\nPerforming grid search: Small grid; medium dataset\n')
local i, best_acc = poly_grid_search(medium_train_data, medium_validation_data, C_large)
print('C_large*['..i..']:' .. C_large[i] .. ' best_acc:', best_acc)

best_C = C_large[i]

print('\n\nTraining on full training data with best parameters picked by grid search.\n\n')
flags= string.format('-q -c %f -t 1 -d 2', best_C)
model = libsvm.train(full_train_data, flags)
libsvm.predict(full_test_data, model)
