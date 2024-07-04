import argparse
import shutil
import os
parser = argparse.ArgumentParser(description='Initate')
parser.add_argument('--name', default='model', type=str, help='save model path')
opt = parser.parse_args()
dirname = os.path.join('./model',opt.name)
#把同级目录下的model_LPN_Gem_denseLPN.py文件用dirname的路径下的model_LPN_Gem_denseLPN.py文件覆盖
shutil.copyfile(os.path.join(dirname,'model_LPN_Gem_denseLPN.py'), 'model_LPN_Gem_denseLPN.py')