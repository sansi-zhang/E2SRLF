import time
from datetime import datetime
import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from einops import rearrange
from torchvision.transforms import ToPILImage

import torch.nn.functional as F
import torch 
import numpy as np



class Train(object):
    def __init__(self, cfg):
        Train.train(cfg)
        
    def save_ckpt(state, save_path='./param/', subdirectory='', filename='checkpoint.pth.tar'):
        save_dir = os.path.join(save_path, subdirectory)
        os.makedirs(save_dir, exist_ok=True)  # 创建子目录，如果不存在的话
        torch.save(state, os.path.join(save_dir, filename))
    
    def downsample(dispGT, scale_factor):
        B, C, H, W = dispGT.shape
        dispGT_down_list = []
        for b in range(B):
            for c in range(C):
                dispGT_slice = dispGT[b, c, :, :]
                # 进行双三次插值下采样
                down_slice = F.interpolate(dispGT_slice.unsqueeze(0).unsqueeze(0), scale_factor=scale_factor, mode='bicubic').squeeze()
                dispGT_down_list.append(down_slice)

        dispGT_down = torch.stack(dispGT_down_list).view(B, C, *down_slice.shape)
        return dispGT_down
    
    def downsample_lf(lf, scale_factor):
        a1, a2, H, W = lf.shape
        lf_down_list = []
        for u in range(a1):
            for v in range(a2):
                # 获取当前通道的（H，W）切片
                lf_slice = lf[u, v]
                # 使用双三次插值进行下采样
                down_slice = zoom(lf_slice, (scale_factor, scale_factor), order=3)
                lf_down_list.append(down_slice)

        # 将下采样后的切片重新组合成原始形状的张量
        lf_down = np.array(lf_down_list).reshape(a1, a2, *down_slice.shape)
        return lf_down
    

    def train(cfg):
        print("--------------------")
        print("train start!!!")
        if cfg.parallel:
            cfg.device = 'cuda:0'
        if 'SACAT' in cfg.net:
            from model.model_SACAT import Net 
        elif 'NAT' in cfg.net:
            from model.model_NAT import Net 
        else:
            from model.model import Net
            print("Using sub_pix + attention model.")
            print("Use pre-agg-sr!!")
            
        net = Net(cfg)

        net.to(cfg.device)
        cudnn.benchmark = True
        epoch_state = 0
        min_loss = float('inf')
        avg_mse = float('inf')
        min_mse = min_mse_loss = float('inf')
        min_loss_epoch = min_mse_epoch = 0
        scale_factor = cfg.scale_factor
        
        if cfg.load_continuetrain:
            filename_para = ('../param/' + cfg.net + '/' + cfg.net + '_lr' + str(cfg.lr) + '_n_steps' + str(cfg.n_steps) + '_' + str(cfg.n_epochs) + 'best_loss' + '.pth.tar')
            
            if os.path.isfile(filename_para):
                model = torch.load(filename_para, map_location={'cuda:0': cfg.device})
                net.load_state_dict(model['state_dict'], strict=False)
                epoch_state = model["epoch"]
                print("=> model found at '{}'".format(filename_para))
            else:
                print("=> no model found at '{}'".format(filename_para))
        
        if cfg.load_pretrain:
            if os.path.isfile(cfg.model_path):
                # model = torch.load(cfg.model_path, map_location={'cuda:0': cfg.device})
                model = torch.load(cfg.model_path, map_location=lambda storage, loc: storage.cuda(int(cfg.device.split(':')[-1])))
                net.load_state_dict(model['state_dict'], strict=False)
                # epoch_state = model["epoch"]
                print("=> model found at '{}'".format(cfg.model_path))
            else:
                print("=> no model found at '{}'".format(cfg.load_model))

        if cfg.parallel:
            net = torch.nn.DataParallel(net, device_ids=[1, 2])
            
            
            
            

        Super_criterion_Loss = torch.nn.L1Loss().to(cfg.device)
        Low_criterion_Loss = torch.nn.L1Loss().to(cfg.device)

        optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)
        scheduler._step_count = epoch_state

        loss_list = []
        epoch_history = []

        L1_loss_rate = 1.0
        Super_loss_rate = 1.0
        Sub_loss_rate = 0.5
        
        for idx_epoch in range(epoch_state, cfg.n_epochs):
            
            train_set = TrainSetLoader(cfg)
            train_loader = DataLoader(dataset=train_set, num_workers=cfg.num_workers, batch_size=cfg.batch_size, shuffle=True)
            loss_epoch = []
            Super_L1_epoch = []
            Low_Loss_epoch = []
            for idx_iter, (data, dispGT) in tqdm(enumerate(train_loader), total=len(train_loader)): 
                # 其中 data 为小尺寸输入数据，dispGT 为大尺寸 label
                data, dispGT = Variable(data).to(cfg.device), Variable(dispGT).to(cfg.device)
                
                # 放缩带来的视差变化
                dispGT_mini = Train.downsample(dispGT, cfg.scale_factor) / (1/cfg.scale_factor)
                
                super_disp_final_down, super_disp_final = net(data)
                
                
                lf = rearrange(data, 'b c (a1 h) (a2 w) -> b c a1 a2 h w', a1=cfg.angRes, a2=cfg.angRes).contiguous()
                    
                # 超分的L1损失（有监督）    
                Super_L1 = Super_criterion_Loss(super_disp_final,  dispGT[:, 0, :, :].unsqueeze(1))
                
                # 以下采样后的深度图构建损失（有监督+无监督）
                ## 总的小尺寸视差图
                L1_loss = Low_criterion_Loss(super_disp_final_down,  dispGT_mini[:, 0, :, :].unsqueeze(1))
                
                low_loss = L1_loss_rate*L1_loss
                
                loss = (Super_loss_rate*Super_L1 + Sub_loss_rate*low_loss)
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_epoch.append(loss.data.cpu())
                Super_L1_epoch.append(Super_L1.data.cpu())
                Low_Loss_epoch.append(low_loss.data.cpu())

            filename_para = cfg.net + '_lr' + str(cfg.lr) + '_n_steps' + str(cfg.n_steps) 
                
            
            if idx_epoch % 1 == 0:         
                print(time.ctime()[4:-5] + ' Epoch----%5d, loss---%f' % (idx_epoch + 1, 
                                                                        float(np.array(loss_epoch).mean())))
                loss_list.append(float(np.array(loss_epoch).mean()))
                epoch_history.append(idx_epoch)
                plt.figure()
                plt.title('loss')
                plt.plot(epoch_history, loss_list, label='Training loss')  # 添加标签
                plt.legend()  # 显示图例
                plt.savefig('../loss/' + filename_para + '.jpg')
                plt.close()
                
                if cfg.parallel:
                    Train.save_ckpt({
                    'epoch': idx_epoch + 1,
                    'state_dict': net.module.state_dict(),
                }, save_path='../param/', subdirectory=cfg.net, filename=filename_para + '_' + str(cfg.n_epochs) + '.pth.tar')
                else:
                    Train.save_ckpt({
                        'epoch': idx_epoch + 1,
                        'state_dict': net.state_dict(),
                    }, save_path='../param/', subdirectory=cfg.net, filename=filename_para + '_' + str(cfg.n_epochs) + '.pth.tar')
                    
                    
                loss_now = np.array(loss_epoch).mean()  
                Super_L1_now = np.array(Super_L1_epoch).mean() 
                Low_Loss_now = np.array(Low_Loss_epoch).mean() 
    
                filename_txt = (cfg.model_name + cfg.net + '_lr' + str(cfg.lr) + '_n_steps' + str(cfg.n_steps) 
                                + '_epoch' + str(cfg.n_epochs) + '.txt')
                txtfile = open(filename_txt, 'a')
                txtfile.write('Epoch={}:\t' 'time:{}\t' 'net:{}\t' 'Super_L1:{}\t' 'Low_Loss:{}\t'
                              'loss_now:{}\t' 'loss_min:{}\t'.format((idx_epoch+1), datetime.now().strftime('%Y%m%d_%H%M%S'), 
                                                                                                      cfg.net, Super_L1_now, Low_Loss_now, loss_now, min_loss))
                txtfile.write('\n')
                txtfile.close()
            if idx_epoch % 20 == 19:     
                avg_mse_now = Train.valid(net, cfg)
                if avg_mse_now < avg_mse:
                    if cfg.parallel:
                        Train.save_ckpt({
                            'epoch': idx_epoch + 1,
                            'state_dict': net.module.state_dict(),
                        }, save_path='../param/', subdirectory=cfg.net, filename=filename_para + '_' + str(cfg.n_epochs) + 'best_mse' + '.pth.tar')
                    else:
                        Train.save_ckpt({
                            'epoch': idx_epoch + 1,
                            'state_dict': net.state_dict(),
                        }, save_path='../param/', subdirectory=cfg.net, filename=filename_para + '_' + str(cfg.n_epochs) + 'best_mse' + '.pth.tar')
                    avg_mse = avg_mse_now
                                    
            if np.array(loss_epoch).mean() < min_loss:
                min_loss = np.array(loss_epoch).mean()
                min_loss_epoch = idx_epoch
                if cfg.parallel:
                    Train.save_ckpt({
                        'epoch': idx_epoch + 1,
                        'state_dict': net.module.state_dict(),
                    }, save_path='../param/', subdirectory=cfg.net, filename=filename_para + '_' + str(cfg.n_epochs) + 'best_loss' + '.pth.tar')
                else:
                    Train.save_ckpt({
                        'epoch': idx_epoch + 1,
                        'state_dict': net.state_dict(),
                    }, save_path='../param/', subdirectory=cfg.net, filename=filename_para + '_' + str(cfg.n_epochs) + 'best_loss' + '.pth.tar')
                
            if idx_epoch % 100 == 99:
                if cfg.parallel:
                    Train.save_ckpt({
                    'epoch': idx_epoch + 1,
                    'state_dict': net.module.state_dict(),
                }, save_path='../param/', subdirectory=cfg.net, filename=filename_para + '_' +  str(idx_epoch+1) + str(cfg.n_epochs) + '.pth.tar')
                else:
                    Train.save_ckpt({
                            'epoch': idx_epoch + 1,
                            'state_dict': net.state_dict(),
                        }, save_path='../param/', subdirectory=cfg.net, filename=filename_para + '_' + str(idx_epoch+1) +str(cfg.n_epochs) + '.pth.tar')                       
            
            scheduler.step()



    def valid(net, cfg):
        print("--------------------")
        print("valid start!!!")   

        avg_mse_now = Train.execut(net, cfg)
        return avg_mse_now

    def execut(net, cfg):
        scale_factor = cfg.scale_factor
        scene_list = ['boxes', 'cotton', 'dino', 'sideboard', 'stripes', 'pyramids', 'dots', 'backgammon']
        angRes = cfg.angRes
        totalmse = 0
        filename_txt = (cfg.model_name + cfg.net + '_lr' + str(cfg.lr) + '_n_steps' + str(cfg.n_steps)
                        + '_MSE' + '_epoch' + str(cfg.n_epochs) + '.txt')
        
        txtfile = open(filename_txt, 'a')
        
        txtfile.write('Valid:\t' 'time:{}\t' 'net:{}\t' .format(datetime.now().strftime('%Y%m%d_%H%M%S'), cfg.net))
        txtfile.close()

        for scenes in scene_list:
            lf = np.zeros(shape=(9, 9, 512, 512, 3), dtype=int)
            for i in range(81):
                temp = imageio.v2.imread(cfg.validset_dir + scenes + '/input_Cam0%.2d.png' % i)
                lf[i // 9, i - 9 * (i // 9), :, :, :] = temp
                del temp
            lf = np.mean((1 / 255) * lf.astype('float32'), axis=-1, keepdims=False)
            disp_gt = np.float32(
                read_pfm(cfg.validset_dir + scenes + '/gt_disp_lowres.pfm'))  # load groundtruth disparity map

            angBegin = (9 - angRes) // 2

            lf_angCrop = lf[angBegin:  angBegin + angRes, angBegin: angBegin + angRes, :, :]

            lf_angCrop = Train.downsample_lf(lf_angCrop, scale_factor)
            # print(lf_angCrop.shape)

            data = rearrange(lf_angCrop, 'u v h w -> (u h) (v w)')
            data = ToTensor()(data.copy())
            data = data.unsqueeze(0)
            with torch.no_grad():
                _,disp = net(data.to(cfg.device))
            print(disp.shape)
            disp = np.float32(disp[0,0,:,:].data.cpu())

            mse100 = np.mean((disp[15:-15, 15:-15] - disp_gt[15:-15, 15:-15]) ** 2) * 100
            print(mse100)
            totalmse+=mse100
            txtfile = open(filename_txt, 'a')
            txtfile.write('mse_{}={:3f}\t'.format(scenes, mse100))
            txtfile.close()
        avg_mse_now = totalmse/len(scene_list)
        txtfile = open(filename_txt, 'a')
        txtfile.write('avg_mse={:3f}\t'.format(totalmse/len(scene_list)))
        txtfile.write('\n')
        txtfile.close()

        return avg_mse_now




