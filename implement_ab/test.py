from utils import *
from einops import rearrange
import argparse
from torchvision.transforms import ToTensor
import torch.nn as nn
from model.HDModelBlock import downsample_lf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as F
import torch 
import numpy as np



class Test(object):
    def __init__(self, cfg):
        Test.test(cfg)
        
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

        
    def test(cfg):
        print("--------------------")
        print("test start!!!")
        # torch.cuda.empty_cache()
        if cfg.parallel:
            cfg.device = 'cuda:0'
            
        if 'SACAT' in cfg.net:
            from model.model_SACAT import Net 
        elif 'NAT' in cfg.net:
            from model.model_NAT import Net 
        else:
            from model.model import Net
            print("Using sub_pix + attention model.")
            print("Sub3 use pre-agg-sr!!")

        net = Net(cfg)
        
        net.to(cfg.device)
        # print(cfg.device)
        scale_factor = cfg.scale_factor
        filename_para = cfg.net + '_lr' + str(cfg.lr) + '_n_steps' + str(cfg.n_steps) + '_' + str(cfg.n_epochs) + 'best_mse.pth.tar'
        
        model_path = '../param/' + cfg.net +'/' + filename_para

        print('param=', model_path)
      
        # model = torch.load(model_path, map_location={'cuda:0': cfg.device})
        # model = torch.load(model_path, map_location={'cuda:0': 'cuda:4'})
        model = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(int(cfg.device.split(':')[-1])))
        net.load_state_dict(model['state_dict'])
        
        if cfg.parallel:
            net = torch.nn.DataParallel(net, device_ids=[0, 1])
            
        if cfg.test_data_mode == 'real_data':
            testset_dir = cfg.testset_dir + 'others/' + cfg.test_data_mode + '/'
            scene_list = ['Amethyst','bracelet', 'Cards', 'dinosaur', 'Jelly Beans', 'knight', 'Lego Bulldozer', 'Lego Gantry Self Portrait', 
                    'Lego Truck', 'Toy Humvee and soldier', 'tuzi', 'хЕФхнР']
        
            scene_list_17 = ['Amethyst','bracelet', 'Cards', 'Jelly Beans', 'knight', 'Lego Bulldozer', 'Lego Gantry Self Portrait', 
                        'Lego Truck', 'tuzi', 'хЕФхнР']
            scene_list_dinosaur = ['dinosaur']
            angRes = cfg.angRes

            for scenes in scene_list:
                print('Working on scene: ' + scenes + '...')

                # 读取图片，并拼接成17*17的 SAIs 阵列
                if scenes in scene_list_17:
                    temp = imageio.v2.imread(testset_dir + scenes + '/out_00_00.png')
                    lf = np.zeros(shape=(17, 17, temp.shape[0], temp.shape[1], 3), dtype=int)
                    for i in range(17):
                        for j in range(17):
                            temp = imageio.v2.imread(testset_dir + scenes + '/out_%.2d_%.2d.png' % (i, j))
                            lf[i, j, :, :, :] = temp

                    lf_gray = np.mean((1 / 255) * lf.astype('float32'), axis=-1, keepdims=False)
                    # 取其中9*9的阵列
                    angBegin = (17 - angRes) // 2

                elif scenes in scene_list_dinosaur:
                    temp = imageio.v2.imread(testset_dir + scenes + '/2067_01_01.png')
                    lf = np.zeros(shape=(9, 9, temp.shape[0], temp.shape[1], 3), dtype=int)
                    for i in range(1, 9):
                        for j in range(1, 9):
                            temp = imageio.v2.imread(testset_dir + scenes + '/2067_%.2d_%.2d.png' % (i, j))
                            lf[i, j, :, :, :] = temp

                    lf_gray = np.mean((1 / 255) * lf.astype('float32'), axis=-1, keepdims=False)
                    # 取其中9*9的阵列
                    angBegin = (9 - angRes) // 2

                else:
                    temp = imageio.v2.imread(testset_dir + scenes + '/0101.png')
                    lf = np.zeros(shape=(15, 15, temp.shape[0], temp.shape[1], 3), dtype=int)
                    for i in range(225):
                        temp = imageio.v2.imread(testset_dir + scenes + '/%.4d.png' % i)
                        lf[i // 15, i - 15 * (i // 15), :, :, :] = temp

                    lf_gray = np.mean((1 / 255) * lf.astype('float32'), axis=-1, keepdims=False)
                    # 取其中9*9的阵列
                    angBegin = (15 - angRes) // 2

                lf_angCrop = lf_gray[angBegin:  angBegin + angRes, angBegin: angBegin + angRes, :, :]

                lf_angCrop = Test.downsample_lf(lf_angCrop, scale_factor)
                print(lf_angCrop.shape)
                image = lf_angCrop[4,4,:,:]
                # plt.imshow(image, cmap='gray')  # 假设张量是灰度图
                # plt.axis('off')
                # plt.savefig('low.png')
                # plt.show()

                data = rearrange(lf_angCrop, 'u v h w -> (u h) (v w)')
                data = ToTensor()(data.copy())
                data = data.unsqueeze(0)
                with torch.no_grad():
                    _,disp = net(data.to(cfg.device))
                print(disp.shape)
                disp = np.float32(disp[0,0,:,:].data.cpu())

                print('Finished! \n')
                mid_dir = cfg.net + '/' + cfg.best
                save_dir = os.path.join(cfg.save_path, mid_dir)
                os.makedirs(save_dir, exist_ok=True) 
                write_pfm(disp, save_dir + '/scenes%s_net%s_epochs%d.pfm' % (scenes, cfg.net, cfg.n_epochs))
                
        elif cfg.test_data_mode == 'HCI':
            testset_dir = cfg.testset_dir + 'validation/' + '/'
            
            # scene_list = ['boxes']
            # scene_list = ['boxes', 'cotton', 'dino', 'sideboard', 'stripes', 'pyramids', 'dots', 'backgammon']
            scene_list = ['boxes', 'cotton', 'dino', 'sideboard', 'stripes', 'pyramids', 'dots', 'backgammon', 'origami', 'bedroom', 'bicycle', 'herbs']
            # scene_list = os.listdir(cfg.testset_dir)

            angRes = cfg.angRes

            for scenes in scene_list:
                print('Working on scene: ' + scenes + '...')
                temp = imageio.v2.imread(testset_dir + scenes + '/input_Cam000.png')
                lf = np.zeros(shape=(9, 9, temp.shape[0], temp.shape[1], 3), dtype=int)
                for i in range(81):
                    temp = imageio.v2.imread(testset_dir + scenes + '/input_Cam0%.2d.png' % i)
                    lf[i // 9, i - 9 * (i // 9), :, :, :] = temp

                lf_gray = np.mean((1 / 255) * lf.astype('float32'), axis=-1, keepdims=False)
                angBegin = (9 - angRes) // 2
                lf_angCrop = lf_gray[angBegin:  angBegin + angRes, angBegin: angBegin + angRes, :, :]
                lf_angCrop = Test.downsample_lf(lf_angCrop, scale_factor)
                print(lf_angCrop.shape)
                image = lf_angCrop[4,4,:,:]


                data = rearrange(lf_angCrop, 'u v h w -> (u h) (v w)')
                data = ToTensor()(data.copy())
                data = data.unsqueeze(0)
                with torch.no_grad():
                    _,disp = net(data.to(cfg.device))
                print(disp.shape)
                disp = np.float32(disp[0,0,:,:].data.cpu())
                print('Finished! \n')
                mid_dir = cfg.net + '/' + cfg.best
                save_dir = os.path.join(cfg.save_path, mid_dir)
                os.makedirs(save_dir, exist_ok=True) 
                write_pfm(disp, save_dir + '/%s.pfm' % (scenes))
                # write_pfm(disp, save_dir + '/scenes%s_net%s_epochs%d.pfm' % (scenes, cfg.net, cfg.n_epochs))
                
                
        elif cfg.test_data_mode == 'our_data':
            testset_dir = cfg.testset_dir + 'others' + '/' + cfg.test_data_mode + '/'
            scene_list = ['arrange', 'basket', 'basket1', 'bronze', 'cactus', 'doll', 'Dora Aemmon', 'flower', 'towers', 'Toy cars']

            angRes = cfg.angRes

            for scenes in scene_list:
                print('Working on scene: ' + scenes + '...')
                temp = imageio.v2.imread(testset_dir + scenes + '/image_0_0.png')
                lf = np.zeros(shape=(17, 17, temp.shape[0], temp.shape[1], 3), dtype=int)
                for i in range(10):
                    for j in range(10):
                        temp = imageio.v2.imread(testset_dir + scenes + '/image_%d_%d.png' % (i, j))
                        lf[i, j, :, :, :] = temp

                lf_gray = np.mean((1 / 255) * lf.astype('float32'), axis=-1, keepdims=False)
                # 取其中9*9的阵列
                angBegin = (10 - angRes) // 2

                lf_angCrop = lf_gray[angBegin:  angBegin + angRes, angBegin: angBegin + angRes, :, :]
                lf_angCrop = Test.downsample_lf(lf_angCrop, scale_factor)
                print(lf_angCrop.shape)
                image = lf_angCrop[4,4,:,:]


                data = rearrange(lf_angCrop, 'u v h w -> (u h) (v w)')
                data = ToTensor()(data.copy())
                data = data.unsqueeze(0)
                with torch.no_grad():
                    _,disp = net(data.to(cfg.device))
                print(disp.shape)
                disp = np.float32(disp[0,0,:,:].data.cpu())
                print('Finished! \n')
                mid_dir = cfg.net + '/' + cfg.best
                save_dir = os.path.join(cfg.save_path, mid_dir)
                os.makedirs(save_dir, exist_ok=True) 
                write_pfm(disp, save_dir + '/scenes%s_net%s_epochs%d.pfm' % (scenes, cfg.net, cfg.n_epochs))
            
        elif cfg.test_data_mode == 'ours_new':
            testset_dir = cfg.testset_dir + 'others/' + cfg.test_data_mode + '/'
            scene_list = ['pictures-91', 'pictures-152', 'pictures-84', 'pictures-121', 'pictures-82', 'pictures-115', 
                  'pictures-85', 'pictures-86', 'pictures-113', 'pictures-88', 'pictures-92', 'pictures-83', 
                  'pictures-138', 'pictures-87', 'pictures-124', 'pictures-97']

            angRes = cfg.angRes

            for scenes in scene_list:
                print('Working on scene: ' + scenes + '...')
                temp = imageio.v2.imread(testset_dir + scenes + '/input_Cam000.png')
                lf = np.zeros(shape=(11, 11, temp.shape[0], temp.shape[1], 3), dtype=int)
                for i in range(121):
                    temp = imageio.v2.imread(testset_dir + scenes + '/input_Cam%.3d.png' % i)
                    lf[i // 11, i - 11 * (i // 11), :, :, :] = temp

                lf_gray = np.mean((1 / 255) * lf.astype('float32'), axis=-1, keepdims=False)
                # 取其中9*9的阵列
                angBegin = (11 - angRes) // 2

                lf_angCrop = lf_gray[angBegin:  angBegin + angRes, angBegin: angBegin + angRes, :, :]
                lf_angCrop = Test.downsample_lf(lf_angCrop, scale_factor)
                print(lf_angCrop.shape)
                image = lf_angCrop[4,4,:,:]


                data = rearrange(lf_angCrop, 'u v h w -> (u h) (v w)')
                data = ToTensor()(data.copy())
                data = data.unsqueeze(0)
                with torch.no_grad():
                    _,disp = net(data.to(cfg.device))
                print(disp.shape)
                disp = np.float32(disp[0,0,:,:].data.cpu())
                print('Finished! \n')
                mid_dir = cfg.net + '/' + cfg.best
                save_dir = os.path.join(cfg.save_path, mid_dir)
                os.makedirs(save_dir, exist_ok=True) 
                write_pfm(disp, save_dir + '/scenes%s_net%s_epochs%d.pfm' % (scenes, cfg.net, cfg.n_epochs))
        
        elif cfg.test_data_mode == 'new_data':
            testset_dir = cfg.testset_dir + 'others/' + cfg.test_data_mode + '/test/'
            scene_list = ['Black_Fence', 'Bush', 'Caution_Bees', 'Flowers', 'Gravel_Garden', 'Mirabelle_Prune_Tree', 'Palais_du_Luxembourg', 'Poppies', 'Rusty_Handle', 'Swans_1', 'Swans_2', 'Trunk', 'Vine_Wood', 'Wheat_&_Silos']

            angRes = cfg.angRes

            for scenes in scene_list:
                print('Working on scene: ' + scenes + '...')

                # 读取图片，并拼接成15*15的 SAIs 阵列
                temp = imageio.v2.imread(testset_dir + scenes + '/input_Cam000.png')
                lf = np.zeros(shape=(15, 15, temp.shape[0], temp.shape[1], 3), dtype=int)
                for i in range(225):
                    temp = imageio.v2.imread(testset_dir + scenes + '/input_Cam%.3d.png' % i)
                    lf[i // 15, i - 15 * (i // 15), :, :, :] = temp

                lf_gray = np.mean((1 / 255) * lf.astype('float32'), axis=-1, keepdims=False)

                # 取其中9*9的阵列
                angBegin = (15 - angRes) // 2
                lf_angCrop = lf_gray[angBegin:  angBegin + angRes, angBegin: angBegin + angRes, :, :]
                lf_angCrop = Test.downsample_lf(lf_angCrop, scale_factor)
                print(lf_angCrop.shape)
                image = lf_angCrop[4,4,:,:]


                data = rearrange(lf_angCrop, 'u v h w -> (u h) (v w)')
                data = ToTensor()(data.copy())
                data = data.unsqueeze(0)
                with torch.no_grad():
                    _,disp = net(data.to(cfg.device))
                print(disp.shape)
                disp = np.float32(disp[0,0,:,:].data.cpu())
                print('Finished! \n')
                mid_dir = cfg.net + '/' + cfg.best
                save_dir = os.path.join(cfg.save_path, mid_dir)
                os.makedirs(save_dir, exist_ok=True) 
                write_pfm(disp, save_dir + '/scenes%s_net%s_epochs%d.pfm' % (scenes, cfg.net, cfg.n_epochs))

        return 






