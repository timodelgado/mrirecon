                                                                                        
#%%
import torch
from graspcg.workspace.device_cfg import DeviceCfg
from graspcg.nufft.organize_k_space_slidingwindow import organize_k_space_slidingwindow
from graspcg.utils.debug import check_memory
from graspcg.solvers.cg_user_init import quick_cg      # path of your solver

device_id = 0
torchdevice = f"cuda:{device_id}"
torch.cuda.set_device(torchdevice)
lr_recon_fov = 0.5
recon_Nx = 112
devices = [device_id]

kdataxy = torch.load('kdataxytest_MID00451.pt', map_location = torchdevice)
[Nch, Ndz, Ndth, Ndr] = kdataxy.shape
maps = torch.load('maps_MID00451_lr_50_112.pt', map_location = torchdevice)
N_keep = int(round(recon_Nx / lr_recon_fov))
edge      = (Ndr - N_keep) // 2

dsfactor = recon_Nx/(Ndr * lr_recon_fov)
print('Downsample Factor: ' + str(dsfactor))
if edge > 0 and edge < Ndr // 2:
    kdataxy = kdataxy[:,:,:,edge:-edge]
[Nch, Ndz, Ndth, Ndr] = kdataxy.shape
print(kdataxy.shape)
voxel_spacing = [6, 1.875 / dsfactor, 1.875 / dsfactor]

org_k, nufft = organize_k_space_slidingwindow(
    kdataxy, maps, 4, 4,
    devices=devices)

del kdataxy, maps


stats_cfg = {
    "tv_t": {"percentile": 0.90, "eps_floor": 1e-6, "kappa": 1.0, "apply_scale": True},
    "tv_s": {"percentile": 0.90, "eps_floor": 1e-6, "kappa": 3.0, "apply_scale": True,
             "voxel_size": voxel_spacing}
}
from graspcg.numerics.continuation import ContinuationConfig, ContinuationManager
cont = ContinuationManager(ContinuationConfig(every=3, alpha=0.6))

x_rec = quick_cg(nufft, org_k,
                 devices=[0,1], stats_cfg=stats_cfg,
                 max_iter=40, line_search="wolfe", direction="prplus",
                 continuation=cont, verbose=True)