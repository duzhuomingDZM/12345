import torch
import torch.optim as optim
from models import networks as N
from util.util import range_compressor_cuda
from models.ahdrnet import AHDR
from models.fshdr import FSHDR
from models.hdr_transformer import HDRTransformer
from models.sctnet import SCTNet
import os


class SelfHDR2Model:
    def __init__(self, opt):
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain

        self.device = torch.device('cuda', self.gpu_ids)

        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'
        self.start_epoch = 0

        self.backwarp_tenGrid = {}
        self.backwarp_tenPartial = {}

        self.opt = opt
        self.loss_names = ['Color', 'Stru', 'Total']
        self.visual_names = ['data_out', 'data_color_label', 'data_input0', 'data_input1', 'data_input2']

        self.model_names = ['SelfHDR']
        self.optimizer_names = ['SelfHDR_optimizer_%s' % opt.optimizer]

        if self.opt.network == 'AHDRNet':
            selfhdr = AHDR()
            opt.chop = False
        elif self.opt.network == 'FSHDR':
            selfhdr = FSHDR()
            opt.chop = False
        elif self.opt.network == 'HDR-Transformer':
            selfhdr = HDRTransformer()
            opt.chop = True
        elif self.opt.network == 'SCTNet':
            selfhdr = SCTNet()
            opt.chop = True

        self.netSelfHDR = N.init_net(selfhdr, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.noise_adder = N.AugmentNoise(style='gauss0.5')

        if self.opt.isTrain:
            self.optimizer_netSelfHDR = optim.Adam(self.netSelfHDR.parameters(),
                                                   lr=opt.lr,
                                                   betas=(opt.beta1, opt.beta2),
                                                   weight_decay=opt.weight_decay)
            self.optimizers = [self.optimizer_netSelfHDR]

            self.criterionL1 = N.init_net(N.L1MuLoss(), gpu_ids=opt.gpu_ids)
            self.criterionVGG = N.init_net(N.VGGLoss(), gpu_ids=opt.gpu_ids)

    def set_input(self, input):
        self.data_input0 = input['input0'].to(self.device)
        self.data_input1 = input['input1'].to(self.device)
        self.data_input2 = input['input2'].to(self.device)
        self.data_color_label = input['label'].to(self.device)
        self.data_stru_label = input['other_label'].to(self.device)
        self.data_expo = input['expo'].to(self.device, dtype=torch.float)
        self.image_name = input['fname']

    def forward(self):
        if self.isTrain:
            in_1_1 = self.noise_adder.add_train_noise(self.data_input1[:, 3:6, ...]).clamp(0, 1)
            in_1_0 = (in_1_1 ** 2.2) / (self.data_expo.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + 1e-8)
            in_1 = torch.cat([in_1_0, in_1_1], 1)
            self.data_out = self.netSelfHDR(self.data_input0, in_1, self.data_input2)
        elif not self.opt.chop:
            self.data_out = self.netSelfHDR(self.data_input0, self.data_input1, self.data_input2)
        else:
            N, C, H, W = self.data_input0.shape
            pad_w = 8 - W % 8
            pad_h = 8 - H % 8
            paddings = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            new_input0 = torch.nn.ReflectionPad2d(paddings)(self.data_input0)
            new_input1 = torch.nn.ReflectionPad2d(paddings)(self.data_input1)
            new_input2 = torch.nn.ReflectionPad2d(paddings)(self.data_input2)
            out = self.netSelfHDR(new_input0, new_input1, new_input2)
            self.data_out = out[:, :, pad_h // 2:pad_h // 2 + H, pad_w // 2:pad_w // 2 + W]

    def backward(self):
        diff = range_compressor_cuda(self.data_color_label) - range_compressor_cuda(self.data_stru_label)
        label_mask = torch.mean(torch.abs(diff), 1, keepdim=True)
        label_mask[label_mask < 10 / 255] = 0
        label_mask[label_mask >= 10 / 255] = 1
        label_mask = 1 - label_mask

        self.loss_Color = self.criterionL1(self.data_out, self.data_color_label, label_mask).mean()
        self.loss_Stru = self.criterionVGG(self.data_out, self.data_stru_label).mean() * 1

        self.loss_Total = self.loss_Color + self.loss_Stru
        self.loss_Total.backward()

    def optimize_parameters(self, epoch):
        self.forward()
        self.optimizer_netSelfHDR.zero_grad()
        self.backward()
        self.optimizer_netSelfHDR.step()

    def get_current_losses(self):
        errors_ret = {}
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = getattr(self, 'loss_' + name).item()
        return errors_ret

    def update_learning_rate(self, epoch):
        for i, scheduler in enumerate(self.schedulers):
            if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                scheduler.step(self.metric)
            elif scheduler.__class__.__name__ == 'CosineLRScheduler':
                scheduler.step(epoch)
            else:
                scheduler.step()
            print('lr of %s = %.7f' % (
                self.optimizer_names[i], self.optimizers[i].param_groups[0]['lr']))

    def save_networks(self, epoch):
        for name in self.model_names:
            save_filename = '%s_model_%d.pth' % (name, epoch)
            save_path = os.path.join(self.save_dir, save_filename)
            net = getattr(self, 'net' + name)
            if self.device.type == 'cuda':
                state = {'state_dict': net.module.cpu().state_dict()}
                torch.save(state, save_path)
                net.to(self.device)
            else:
                state = {'state_dict': net.state_dict()}
                torch.save(state, save_path)
        self.save_optimizers(epoch)

    def load_networks(self, epoch):
        name = 'SelfHDR'
        load_filename = '%s_model_%d.pth' % (name, epoch)
        if self.opt.load_path != '':
            load_path = self.opt.load_path
        else:
            load_path = os.path.join(self.save_dir, load_filename)
        net = getattr(self, 'net' + name)

        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        state_dict = torch.load(load_path, map_location=self.device)
        print('loading the model from %s' % (load_path))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        net_state = net.state_dict()
        is_loaded = {n: False for n in net_state.keys()}
        for name, param in state_dict['state_dict'].items():
            if name in net_state:
                try:
                    if net_state[name].shape == param.shape:
                        net_state[name].copy_(param)
                        is_loaded[name] = True
                except Exception:
                    print('While copying the parameter named [%s], '
                          'whose dimensions in the model are %s and '
                          'whose dimensions in the checkpoint are %s.'
                          % (name, list(net_state[name].shape),
                             list(param.shape)))
                    raise RuntimeError
            else:
                print('Saved parameter named [%s] is skipped' % name)
            mark = True
            for name in is_loaded:
                if not is_loaded[name]:
                    print('Parameter named [%s] is randomly initialized' % name)
                    mark = False
            if mark:
                print('All parameters are initialized using [%s]' % load_path)

            self.start_epoch = epoch

    def save_optimizer(self, epoch):
        assert len(self.optimizers) == len(self.optimizer_names)
        for id, optimizer in enumerate(self.optimizers):
            save_filename = self.optimizer_names[id]
            state = {'name': save_filename,
                     'epoch': epoch,
                     'state_dict': optimizer.state_dict()}
            save_path = os.path.join(self.save_dir, save_filename + '.pth')
            torch.save(state, save_path)

    def load_optimizer(self, epoch):
        assert len(self.optimizers) == len(self.optimizer_names)
        for id, optimizer in enumerate(self.optimizer_names):
            load_filename = self.optimizer_names[id]
            load_path = os.path.join(self.save_dir, load_filename + '.pth')
            print('loading the optimizer from %s' % load_path)
            state_dict = torch.load(load_path)
            assert optimizer == state_dict['name']
            assert epoch == state_dict['epoch']
            self.optimizers[id].load_state_dict(state_dict['state_dict'])

    def setup(self, opt):
        opt = opt
        if self.isTrain:
            self.schedulers = [N.get_scheduler(optimizer, opt) \
                               for optimizer in self.optimizers]
            for scheduler in self.schedulers:
                scheduler.last_epoch = opt.load_iter

        load_suffix = opt.load_iter
        self.load_networks(load_suffix)
        if opt.load_optimizers:
            self.load_optimizers(opt.load_iter)

        # self.print_networks(opt.verbose)

    def test(self):
        self.isTrain = False
        with torch.no_grad():
            self.forward()

    def eval(self):
        for name in self.model_names:
            net = getattr(self, 'net' + name)
            net.eval()

    def train(self):
        self.isTrain = True
        for name in self.model_names:
            net = getattr(self, 'net' + name)
            net.train()
