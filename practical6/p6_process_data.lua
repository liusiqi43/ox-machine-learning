--
-- The data consists of 5000 points each of size 1024. Each point is actually a
-- 32x32 image from MNIST dataset The first 500 images are of 0, the next 500
-- or 1, and so on
data = torch.load('practical6_data.t7', 'ascii')

-- Normalise data
-- Actually many components of the data have 0 variance. Ideally we should
-- throw them out. However, since we would like to reconstruct images later and
-- view them, we'll keep them. They won't contribute to the PCs anyway
means = data:mean(1):squeeze()
stds = data:std(1):squeeze()
for i =1, (#stds)[1] do
	data[{{}, {i}}]:add(-means[i])
	if stds[i] > 0 then
		data[{{}, {i}}]:mul(1/stds[i])
	end
end

-- Save normalised dataset
-- We will use the normalised dataset to plot projections onto PCs
torch.save('p6_normalised_data.t7', data, 'ascii')

-- We perform svd on the normalized dataset 
-- Save the decomposition so that we can experiment with it later and avoid
-- having to run an svd repeatedly
svd_data = {}
svd_data.U, svd_data.S, svd_data.V = torch.svd(data)
torch.save('svd_data.t7', svd_data, 'ascii')
