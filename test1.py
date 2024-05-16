from config import parse_opt
import numpy as np
from config import *
from Data import SIG17AlignDataset
from model import SelfHDR2Model
# from models import create_model
# from models.selfhdr2_model import SelfHDR2Model
import time
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from util.util import range_compressor, calculate_ssim

from torch.utils.data import DataLoader
import pilpline
import numpy as np
from config import *
import time
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from util.util import range_compressor, calculate_ssim

opt = parse_opt()
# for l, r in vars(opt).items(): print(l, ':', r)
opt.isTrain = False

dataset = SIG17AlignDataset(opt, 'test')
dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=True)
dataset_size_test = len(dataset)
print(dataset_size_test)

model = SelfHDR2Model(opt)
model.setup(opt)
model.eval()

log_dir = '%s/%s/log_epoch_%d.txt' % (opt.checkpoints_dir, opt.name, 1)
os.makedirs(os.path.split(log_dir)[0], exist_ok=True)
f = open(log_dir, 'a')


#
print('='*80)
# print(dataset_name + ' dataset')

psnr_l = [0.0] * dataset_size_test
psnr_mu = [0.0] * dataset_size_test
ssim_l = [0.0] * dataset_size_test
ssim_mu = [0.0] * dataset_size_test
time_val =0

for i, data in enumerate(dataloader):
    torch.cuda.empty_cache()
    model.set_input(data)
    torch.cuda.synchronize()
    time_val_start = time.time()
    model.test()
    torch.cuda.synchronize()
    time_val += time.time() - time_val_start
    # res = model.get_current_visuals()

    output = model.data_out[0].detach().cpu().numpy().astype(np.float32)  # [::-1,:,:]
    gt = model.data_color_label[0].detach().cpu().numpy().astype(np.float32)

    if opt.calc_metrics:
        # psnr-l and psnr-\mu
        psnr_l[i] = compare_psnr(gt, output, data_range=1.0)
        label_mu = range_compressor(gt)
        output_mu = range_compressor(output)
        psnr_mu[i] = compare_psnr(label_mu, output_mu, data_range=1.0)
        # ssim-l
        output_l = np.clip(output * 255.0, 0., 255.).transpose(1, 2, 0)
        label_l = np.clip(gt * 255.0, 0., 255.).transpose(1, 2, 0)
        ssim_l[i] = calculate_ssim(output_l, label_l)
        # ssim-\mu
        output_mu = np.clip(output_mu * 255.0, 0., 255.).transpose(1, 2, 0)
        label_mu = np.clip(label_mu * 255.0, 0., 255.).transpose(1, 2, 0)
        ssim_mu[i] = calculate_ssim(output_mu, label_mu)
#
avg_psnr_l = '%.2f'%np.mean(psnr_l)
avg_psnr_mu = '%.2f'%np.mean(psnr_mu)
avg_ssim_l = '%.4f'%np.mean(ssim_l)
avg_ssim_mu = '%.4f'%np.mean(ssim_mu)

f = open(log_dir, 'a')

f.write('AVG Time: %.3f ms \n avg_psnr_l: %s, avg_psnr_mu: %s \n avg_ssim_l: %s, avg_ssim_mu: %s \n'
        % (time_val / dataset_size_test * 1000, avg_psnr_l, avg_psnr_mu, avg_ssim_l, avg_ssim_mu))
print('AVG Time: %.3f ms \n avg_psnr_l: %s, avg_psnr_mu: %s \n avg_ssim_l: %s, avg_ssim_mu: %s \n'
      % (time_val / dataset_size_test * 1000, avg_psnr_l, avg_psnr_mu, avg_ssim_l, avg_ssim_mu))
f.flush()
f.write('\n')
f.close()


