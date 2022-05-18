import torch
import os
path_monet = '/home/hulk/slendas/DL/kaggle_gan/gan-getting-started/monet_jpg'
path_photo = '/home/hulk/slendas/DL/kaggle_gan/gan-getting-started/photo_jpg'

path_data = '/home/hulk/slendas/DL/kaggle_gan/model_state'


def save_model(G_XtoY, G_YtoX, D_X, D_Y):

    G_XtoY_path = os.path.join(path_data, 'G_XtoY.pt')
    G_YtoX_path = os.path.join(path_data, 'G_YtoX.pt')
    D_X_path = os.path.join(path_data, 'D_X.pt')
    D_Y_path = os.path.join(path_data, 'D_Y.pt')
    
    torch.save(G_XtoY.state_dict(), G_XtoY_path)
    torch.save(G_YtoX.state_dict(), G_YtoX_path)
    torch.save(D_X.state_dict(), D_X_path)
    torch.save(D_Y.state_dict(), D_Y_path)


def load_model(G_XtoY, G_YtoX, D_X, D_Y):
    G_XtoY_path = os.path.join(path_data, 'G_XtoY.pt')
    G_YtoX_path = os.path.join(path_data, 'G_YtoX.pt')
    D_X_path = os.path.join(path_data, 'D_X.pt')
    D_Y_path = os.path.join(path_data, 'D_Y.pt')
    
    G_XtoY.load_state_dict(torch.load(G_XtoY_path))
    G_YtoX.load_state_dict(torch.load(G_YtoX_path))
    D_X.load_state_dict(torch.load(D_X_path))
    D_Y.load_state_dict(torch.load(D_Y_path))
    
