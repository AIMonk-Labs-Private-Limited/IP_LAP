from cuda import cuda 
from cuda import cudart
import pycuda.autoinit
import tensorrt as trt
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import face_alignment
from face_alignment import FaceAlignment
from face_alignment.detection.sfd.bbox import *
from face_alignment.detection.sfd.sfd_detector import SFDDetector
from pathlib import Path
import os

TRT_LOGGER = trt.Logger()
ENGINE_PATH=str(Path( os.path.dirname(os.path.abspath(__file__))).parent/'checkpoints/s3fd_v8601.engine')

def trt_inference(engine, context, data):  
    
    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    nOutput = engine.num_bindings - nInput        
    bufferH = []
    bufferH.append(np.ascontiguousarray(data.reshape(-1)))
    
    for i in range(nInput, nInput + nOutput):
        bufferH.append(np.empty(context.get_binding_shape(i), dtype=trt.nptype(engine.get_binding_dtype(i))))
    
    bufferD = []
    for i in range(nInput + nOutput):
        bufferD.append(cuda.cuMemAlloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cuda.cuMemcpyHtoD(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes)
    
    context.execute_v2(bufferD)

    for i in range(nInput, nInput + nOutput):
        cuda.cuMemcpyDtoH(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes)
        
    for b in bufferD:
        cuda.cuMemFree(b)  
    
    return bufferH

def trt_detect(net,engine, img, device):
    img = img.transpose(2, 0, 1)
    # Creates a batch of 1
    img = np.expand_dims(img, 0)

    img = torch.from_numpy(img.copy()).to(device, dtype=torch.float32)

    return trt_batch_detect(net,engine, img, device)

def trt_batch_detect(net,engine, img_batch, device):
    """
    Inputs:
        - img_batch: a torch.Tensor of shape (Batch size, Channels, Height, Width)
    """
    if 'cuda' in device:
        torch.backends.cudnn.benchmark = True
        
    batch_size = img_batch.size(0)
    img_batch = img_batch.to(device, dtype=torch.float32)
    img_batch = img_batch.flip(-3)  # RGB to BGR
    img_batch = img_batch - torch.tensor([104.0, 117.0, 123.0], device=device).view(1, 3, 1, 1)
    img_batch= img_batch.cpu().numpy() ##speed up
    shape=img_batch.shape ##speed up
    net.set_binding_shape(0, shape) ##speed up
    trt_outputs = trt_inference(engine, net, img_batch) ##speed up
    olist=[torch.tensor(trt_outputs[12])]+[torch.tensor(z) for z in trt_outputs[1:12]] ##speed up
    for i in range(len(olist) // 2):
        olist[i * 2] = F.softmax(olist[i * 2], dim=1)
    olist = [oelem.data.cpu().numpy() for oelem in olist]
    bboxlists = get_predictions(olist, batch_size)
    return bboxlists


def get_predictions(olist, batch_size):
    bboxlists = []
    variances = [0.1, 0.2]
    for j in range(batch_size):
        bboxlist = []
        for i in range(len(olist) // 2):
            ocls, oreg = olist[i * 2], olist[i * 2 + 1]
            stride = 2**(i + 2)    # 4,8,16,32,64,128
            poss = zip(*np.where(ocls[:, 1, :, :] > 0.05))
            for Iindex, hindex, windex in poss:
                axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
                score = ocls[j, 1, hindex, windex]
                loc = oreg[j, :, hindex, windex].copy().reshape(1, 4)
                priors = np.array([[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]])
                box = decode(loc, priors, variances)
                x1, y1, x2, y2 = box[0]
                bboxlist.append([x1, y1, x2, y2, score])

        bboxlists.append(bboxlist)

    bboxlists = np.array(bboxlists)
    return bboxlists

class FaceAlignment_trt(FaceAlignment):
    
    def __init__(self,flip_input=False, device='cuda'):
        super().__init__(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cuda')
        self.face_detector=SFD_Detector_trt(device)
        
        
class SFD_Detector_trt(SFDDetector):
    def __init__(self,device, path_to_detector=None, verbose=False, filter_threshold=0.5):
        super().__init__(device, path_to_detector=None, verbose=False, filter_threshold=0.5)
        with open(ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime: 
            self.engine = runtime.deserialize_cuda_engine(f.read())   
        self.detector = self.engine.create_execution_context()
        
    def detect_from_image(self, tensor_or_path):
        image = self.tensor_or_path_to_ndarray(tensor_or_path)

        bboxlist = trt_detect(self.detector,self.engine, image, device=self.device)[0]
        bboxlist = self._filter_bboxes(bboxlist)

        return bboxlist
        
        
    
        
    
    
