# graspcg/tests/test_ws_no_regs.py
import torch
from graspcg.workspace.cg_workspace import CGWorkspace
from graspcg.workspace.device_cfg import DeviceCfg

class DummyNUFFT:
    imshape = (2,3,4,5)
    def attach_arena(self, arena): pass

def test_workspace_shapes_and_scale():
    y = torch.zeros((2,8,128), dtype=torch.complex64)  # whatever your y shape is
    ws = CGWorkspace(y, DummyNUFFT(), device_cfg=DeviceCfg())
    assert ws.dims[0] == 2
    s = ws.s_t
    assert s.shape[0] == ws.dims[0]
    ws.scale.set(2.0)
    assert float(ws.s_t.flatten()[0]) == 2.0

def test_ws_bind_regs_bridge():
    y = torch.zeros((2,8,128), dtype=torch.complex64)
    ws = CGWorkspace(y, DummyNUFFT(), device_cfg=DeviceCfg())
    class R: 
        def __init__(self): self.regs={}
        def add(self,k,**cfg): self.regs[k]=cfg
    regm = R()
    ws.bind_regs(regm)
    ws.add_reg("tv_t", weight=1e-3, eps=1e-2)
    assert "tv_t" in ws.regs and ws.regs["tv_t"]["weight"] == 1e-3