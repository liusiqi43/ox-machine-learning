require 'gnuplot';
require 'image';
require 'nn';

function transform(vec)
	vmin = vec:min()
	vmax = vec:max()
	vec:apply(function (x) return ((x - vmin)/(vmax - vmin)) end)
end
	

-- Load normalised dataset
data = torch.load('p6_normalised_data.t7', 'ascii')

-- svd contains U, S, V
svd = torch.load('svd_data.t7', 'ascii')

-- For three-d plots
-- For 2d plots only use first two PCs
data3 = torch.mm(data, svd.V[{{}, {1, 3}}])

gnuplot.figure(1)
gnuplot.plot(
{'0', data3[{{1,500}, {1}}]:squeeze(), data3[{{1, 500}, {2}}]:squeeze(), '+'},
{'1', data3[{{501,1000}, {1}}]:squeeze(), data3[{{501, 1000}, {2}}]:squeeze(), '+'}, 
{'7', data3[{{3501,4000}, {1}}]:squeeze(), data3[{{3501, 4000}, {2}}]:squeeze(), '+'},
{'8', data3[{{4001,4500}, {1}}]:squeeze(), data3[{{4001, 4500}, {2}}]:squeeze(), '+'}
)

gnuplot.figure(2)
gnuplot.scatter3(
{'0', data3[{{1,500}, {1}}]:squeeze(), data3[{{1, 500}, {2}}]:squeeze(), data3[{{1, 500}, {3}}]:squeeze()},
{'1', data3[{{501,1000}, {1}}]:squeeze(), data3[{{501, 1000}, {2}}]:squeeze(), data3[{{501, 1000}, {3}}]:squeeze()},
{'2', data3[{{1001,1500}, {1}}]:squeeze(), data3[{{1001, 1500}, {2}}]:squeeze(), data3[{{1001, 1500}, {3}}]:squeeze()},
{'3', data3[{{1501,2000}, {1}}]:squeeze(), data3[{{1501, 2000}, {2}}]:squeeze(), data3[{{1501, 2000}, {3}}]:squeeze()},
{'4', data3[{{2001,2500}, {1}}]:squeeze(), data3[{{2001, 2500}, {2}}]:squeeze(), data3[{{2001, 2500}, {3}}]:squeeze()},
{'5', data3[{{2501,3000}, {1}}]:squeeze(), data3[{{2501, 3000}, {2}}]:squeeze(), data3[{{2501, 3000}, {3}}]:squeeze()},
{'6', data3[{{3001,3500}, {1}}]:squeeze(), data3[{{3001, 3500}, {2}}]:squeeze(), data3[{{3001, 3500}, {3}}]:squeeze()},
{'7', data3[{{3501,4000}, {1}}]:squeeze(), data3[{{3501, 4000}, {2}}]:squeeze(), data3[{{3501, 4000}, {3}}]:squeeze()},
{'8', data3[{{4001,4500}, {1}}]:squeeze(), data3[{{4001, 4500}, {2}}]:squeeze(), data3[{{4001, 4500}, {3}}]:squeeze()},
{'9', data3[{{4501,5000}, {1}}]:squeeze(), data3[{{4501, 5000}, {2}}]:squeeze(), data3[{{4501, 5000}, {3}}]:squeeze()}
)

-- View a few different PCs as images
-- We will do a crude renormalization to avoid very dark images
components  = {1, 2, 3, 25, 100, 500, 1000}
for i=1,(#components) do
        idx = components[i]
        filename = string.format('PC%04d.png', idx)
        image.save(filename, (svd.V[{{}, {idx}}]/(svd.V[{{}, {idx}}]:max())):reshape(1, 32, 32))
end

-- Advanced
-- TODO pick one of image of each digit and project onto first 1, 2, 4, 8, 16, 20 PCs and see the quality of reconstruction you get
pcs = {1, 2, 4, 8, 16, 20, 100}
for digit=1, 9 do
    for i=1,(#pcs) do
            filename = string.format('reconstructions/PC%d-with-%d-pcs.png', digit-1, pcs[i])
            vk = svd.V[{{}, {1, pcs[i]}}]
            sampled_img = data[{{25 + 500*(digit-1)}, {}}]
            image.save(filename, (vk * vk:t() * sampled_img:t()):reshape(1, 32, 32))
    end
end
