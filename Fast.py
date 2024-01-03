from gfpgan import GFPGANer
from facexlib.utils.face_restoration_helper import FaceRestoreHelper

import torch
# from basicsr.utils import img2tensor, tensor2img
from torchvision.transforms.functional import normalize

import cv2
import kornia
from kornia.geometry.transform import warp_affine
import torchvision.transforms.functional as F
import numpy as np

from facexlib.detection.retinaface import RetinaFace
from facexlib.utils import load_file_from_url
from copy import deepcopy




from facexlib.detection.retinaface_utils import (PriorBox, decode, decode_landm,
                                                 py_cpu_nms)

# from facexlib.utils.misc import img2tensor, imwrite

import os 

class Fast_PriorBox(PriorBox):
    def __init__(self, cfg, image_size=None):
        super().__init__(cfg, image_size)
        
    def process(self,xv,st,min_sizes,image_size):
        v_1,v_2=[(xv+0.5)*st/image_size]*2
        v_1,v_2=v_1.flatten(),v_2.flatten()
        s_k1,s_k2=torch.tensor(min_sizes)/image_size
        return v_1,v_2,s_k1,s_k2
        

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):    
            st=self.steps[k]
            min_sizes = self.min_sizes[k]        
            xv,yv=torch.meshgrid([torch.arange(0,f[0]),torch.arange(0,f[1])])
            xv_1,xv_2,s_kx1,s_kx2=self.process(xv,st,min_sizes,self.image_size[1]) ##for getting in x-direction
            yv_1,yv_2,s_ky1,s_ky2=self.process(yv,st,min_sizes,self.image_size[0]) ##for getting in y-direction           
            arr_1=torch.stack([yv_1,xv_1,s_kx1*torch.ones_like(xv_1),s_ky1*torch.ones_like(yv_1)],dim=1)
            arr_2=torch.stack([yv_2,xv_2,s_kx2*torch.ones_like(xv_2),s_ky2*torch.ones_like(yv_2)],dim=1)     
            anchors.append(torch.cat((arr_1,arr_2),dim=1).flatten())
        output=torch.Tensor(torch.cat(anchors,dim=0)).view(-1,4)  
        if self.clip:
            output.clamp_(max=1, min=0)   
        return output
 
def get_largest_face(det_faces, h, w):

    def get_location(val, length):
        if val < 0:
            return 0
        elif val > length:
            return length
        else:
            return val

    face_areas = []
    for det_face in det_faces:
        left = get_location(det_face[0], w)
        right = get_location(det_face[2], w)
        top = get_location(det_face[1], h)
        bottom = get_location(det_face[3], h)
        face_area = (right - left) * (bottom - top)
        face_areas.append(face_area)
    largest_idx = face_areas.index(max(face_areas))
    return det_faces[largest_idx], largest_idx


def get_center_face(det_faces, h=0, w=0, center=None):
    if center is not None:
        center = np.array(center)
    else:
        center = np.array([w / 2, h / 2])
    center_dist = []
    for det_face in det_faces:
        face_center = np.array([(det_face[0] + det_face[2]) / 2, (det_face[1] + det_face[3]) / 2])
        dist = np.linalg.norm(face_center - center)
        center_dist.append(dist)
    center_idx = center_dist.index(min(center_dist))
    return det_faces[center_idx], center_idx

def init_detection_model(model_name, half=False, device='cuda', model_rootpath=None):
    if model_name == 'retinaface_resnet50':
        model = Fast_RetinaFace(network_name='resnet50', half=half, device=device)
        model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth'
    elif model_name == 'retinaface_mobile0.25':
        model = Fast_RetinaFace(network_name='mobile0.25', half=half, device=device)
        model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_mobilenet0.25_Final.pth'
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    model_path = load_file_from_url(
        url=model_url, model_dir='facexlib/weights', progress=True, file_name=None, save_dir=model_rootpath)

    # TODO: clean pretrained model
    load_net = torch.load(model_path, map_location=lambda storage, loc: storage)
    # remove unnecessary 'module.'
    for k, v in deepcopy(load_net).items():
        if k.startswith('module.'):
            load_net[k[7:]] = v
            load_net.pop(k)
    model.load_state_dict(load_net, strict=True)
    model.eval()
    model = model.to(device)
    return model


class Fast_RetinaFace(RetinaFace):
    def __init__(self,network_name='resnet50', half=False, device="cuda"):
        super().__init__(network_name, half, device)
        self.priors = None
        self.previous_image_size = None
    
    def __detect_faces(self, inputs):
        # get scale
        height, width = inputs.shape[2:]
        self.scale = torch.tensor([width, height, width, height], dtype=torch.float32, device=self.device)
        tmp = [width, height, width, height, width, height, width, height, width, height]
        self.scale1 = torch.tensor(tmp, dtype=torch.float32, device=self.device)
            
        # forawrd
        # print("Is Half Inferenceing", self.half_inference)
        inputs = inputs.to(self.device)
        if self.half_inference:
            inputs = inputs.half()
        loc, conf, landmarks = self(inputs)
        priorbox=Fast_PriorBox(self.cfg, image_size=inputs.shape[2:])
        priors = priorbox.forward().to(self.device)
        return loc, conf, landmarks, priors    
    
    def detect_faces(
        self,
        image,
        conf_threshold=0.8,
        nms_threshold=0.4,
        use_origin_size=True,
    ):
        image, self.resize = self.transform(image, use_origin_size)
        image = image.to(self.device)
        if self.half_inference:
            image = image.half()
        image = image - self.mean_tensor
        loc, conf, landmarks, priors = self.__detect_faces(image)

        boxes = decode(loc.data.squeeze(0), priors.data, self.cfg['variance'])
        boxes = boxes * self.scale / self.resize
        boxes = boxes.cpu().numpy()

        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        
        landmarks = decode_landm(landmarks.squeeze(0), priors, self.cfg['variance'])
        landmarks = landmarks * self.scale1 / self.resize
        landmarks = landmarks.cpu().numpy()
        # ignore low scores
        inds = np.where(scores > conf_threshold)[0]
        boxes, landmarks, scores = boxes[inds], landmarks[inds], scores[inds]

        # sort
        order = scores.argsort()[::-1]
        boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]

        # do NMS
        bounding_boxes = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(bounding_boxes, nms_threshold)
        bounding_boxes, landmarks = bounding_boxes[keep, :], landmarks[keep]
      
        retina_net_output=np.concatenate((bounding_boxes, landmarks), axis=1)            
        return retina_net_output


class FAST_FaceRestoreHelper(FaceRestoreHelper):
    
    def __init__(self,upscale,face_size=512,crop_ratio=(1, 1),det_model='retinaface_resnet50',
                    save_ext='png',
                    use_parse=True,
                    device="cuda:0",
                    model_rootpath='gfpgan/weights'):
        upscale=1
        model_rootpath='gfpgan/weights'
        self.device=device
        super().__init__(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            use_parse=True,
            device=self.device,
            model_rootpath='gfpgan/weights')

        det_model='retinaface_resnet50'
        self.face_det = init_detection_model(det_model, half=False, device=self.device, model_rootpath=model_rootpath)
    
    def get_face_landmarks_5(self,
                             only_keep_largest=False,
                             only_center_face=False,
                             resize=None,
                             blur_ratio=0.01,
                             eye_dist_threshold=None):
        if resize is None:
            scale = 1
            input_img = self.input_img
        else:
            h, w = self.input_img.shape[0:2]
            scale = min(h, w) / resize
            h, w = int(h / scale), int(w / scale)
            input_img = cv2.resize(self.input_img, (w, h), interpolation=cv2.INTER_LANCZOS4)
        with torch.no_grad():
            bboxes = self.face_det.detect_faces(input_img, 0.97) * scale
        for bbox in bboxes:
            # remove faces with too small eye distance: side faces or too small faces
            eye_dist = np.linalg.norm([bbox[5] - bbox[7], bbox[6] - bbox[8]])
            if eye_dist_threshold is not None and (eye_dist < eye_dist_threshold):
                continue

            if self.template_3points:
                landmark = np.array([[bbox[i], bbox[i + 1]] for i in range(5, 11, 2)])
            else:
                landmark = np.array([[bbox[i], bbox[i + 1]] for i in range(5, 15, 2)])
            self.all_landmarks_5.append(landmark)
            self.det_faces.append(bbox[0:5])
            
        if len(self.det_faces) == 0:
            return 0
        if only_keep_largest:
            h, w, _ = self.input_img.shape
            self.det_faces, largest_idx = get_largest_face(self.det_faces, h, w)
            self.all_landmarks_5 = [self.all_landmarks_5[largest_idx]]
            
        elif only_center_face:
            h, w, _ = self.input_img.shape
            self.det_faces, center_idx = get_center_face(self.det_faces, h, w)
            self.all_landmarks_5 = [self.all_landmarks_5[center_idx]]
    
        # pad blurry images
        if self.pad_blur:
            self.pad_input_imgs = []
            for landmarks in self.all_landmarks_5:
                # get landmarks
                eye_left = landmarks[0, :]
                eye_right = landmarks[1, :]
                eye_avg = (eye_left + eye_right) * 0.5
                mouth_avg = (landmarks[3, :] + landmarks[4, :]) * 0.5
                eye_to_eye = eye_right - eye_left
                eye_to_mouth = mouth_avg - eye_avg

                # Get the oriented crop rectangle
                # x: half width of the oriented crop rectangle
                x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
                #  - np.flipud(eye_to_mouth) * [-1, 1]: rotate 90 clockwise
                # norm with the hypotenuse: get the direction
                x /= np.hypot(*x)  # get the hypotenuse of a right triangle
                rect_scale = 1.5
                x *= max(np.hypot(*eye_to_eye) * 2.0 * rect_scale, np.hypot(*eye_to_mouth) * 1.8 * rect_scale)
                # y: half height of the oriented crop rectangle
                y = np.flipud(x) * [-1, 1]

                # c: center
                c = eye_avg + eye_to_mouth * 0.1
                # quad: (left_top, left_bottom, right_bottom, right_top)
                quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
                # qsize: side length of the square
                qsize = np.hypot(*x) * 2
                border = max(int(np.rint(qsize * 0.1)), 3)

                # get pad
                # pad: (width_left, height_top, width_right, height_bottom)
                pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                       int(np.ceil(max(quad[:, 1]))))
                pad = [
                    max(-pad[0] + border, 1),
                    max(-pad[1] + border, 1),
                    max(pad[2] - self.input_img.shape[0] + border, 1),
                    max(pad[3] - self.input_img.shape[1] + border, 1)
                ]

                if max(pad) > 1:
                    # pad image
                    pad_img = np.pad(self.input_img, ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
                    # modify landmark coords
                    landmarks[:, 0] += pad[0]
                    landmarks[:, 1] += pad[1]
                    # blur pad images
                    h, w, _ = pad_img.shape
                    y, x, _ = np.ogrid[:h, :w, :1]
                    mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0],
                                                       np.float32(w - 1 - x) / pad[2]),
                                      1.0 - np.minimum(np.float32(y) / pad[1],
                                                       np.float32(h - 1 - y) / pad[3]))
                    blur = int(qsize * blur_ratio)
                    if blur % 2 == 0:
                        blur += 1
                    blur_img = cv2.boxFilter(pad_img, 0, ksize=(blur, blur))
                    # blur_img = cv2.GaussianBlur(pad_img, (blur, blur), 0)

                    pad_img = pad_img.astype('float32')
                    pad_img += (blur_img - pad_img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
                    pad_img += (np.median(pad_img, axis=(0, 1)) - pad_img) * np.clip(mask, 0.0, 1.0)
                    pad_img = np.clip(pad_img, 0, 255)  # float32, [0, 255]
                    self.pad_input_imgs.append(pad_img)
                else:
                    self.pad_input_imgs.append(np.copy(self.input_img))
        
        return len(self.all_landmarks_5)
     
    def paste_faces_to_input_image(self, save_path=None, upsample_img=None):
        
        from facexlib.utils.misc import img2tensor, imwrite

        h, w, _ = self.input_img.shape
        h_up, w_up = int(h * self.upscale_factor), int(w * self.upscale_factor)
        
        if upsample_img is None:
            # simply resize the background
            upsample_img = cv2.resize(self.input_img, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)
        else:
            upsample_img = cv2.resize(upsample_img, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)

        assert len(self.restored_faces) == len(
            self.inverse_affine_matrices), ('length of restored_faces and affine_matrices are different.')

        
        for restored_face, inverse_affine in zip(self.restored_faces, self.inverse_affine_matrices):
            # cnt__+=1
            # Add an offset to inverse affine matrix, for more precise back alignment            
            if self.upscale_factor > 1:
                extra_offset = 0.5 * self.upscale_factor
            else:
                extra_offset = 0
            inverse_affine[:, 2] += extra_offset
            
            inv_restored = cv2.warpAffine(restored_face, inverse_affine, (w_up, h_up))
            if self.use_parse:
                # inference

                face_input = cv2.resize(restored_face, (512, 512), interpolation=cv2.INTER_LINEAR)
                face_input = img2tensor(face_input.astype('float32') / 255., bgr2rgb=True, float32=True)
                normalize(face_input, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                face_input = torch.unsqueeze(face_input, 0).to(self.device) 
                with torch.no_grad():
                    out = self.face_parse(face_input)[0]
                # out = out.argmax(dim=1).squeeze().cpu().numpy()
                out = out.argmax(dim=1).squeeze()  ##speed up part
                # inv_soft_mask, pasted_face=numba_speedup_full(out, restored_face, inverse_affine, w_up, h_up, inv_restored)
                # mask = np.zeros(out.shape)
                mask = torch.zeros_like(out) ##speed up
                MASK_COLORMAP = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0]
                for idx, color in enumerate(MASK_COLORMAP):
                    mask[out == idx] = color
                #  blur the mask
                gauss = kornia.filters.GaussianBlur2d((101, 101),(11,11))  ##speed up part
                mask=mask.unsqueeze(0).unsqueeze(0).float()  ##speed up part
                mask=gauss(mask)  ##speed up part

                # remove the black borders
                thres = 10 ##speed up part
    
                mask[:,:,:thres, :] = 0  ##speed up part
                mask[:,:,-thres:, :] = 0  ##speed up part
                mask[:,:,:, :thres] = 0  ##speed up part
                mask[:,:,:, -thres:] = 0  ##speed up part
                mask = mask / 255.   ##speed up part

                mask=F.resize(mask,size=restored_face.shape[:2]) ##speed up
                # mask = cv2.warpAffine(mask, inverse_affine, (w_up, h_up), flags=3)
                inverse_affine=torch.tensor(inverse_affine).to(self.device).unsqueeze(0) ##speed up
                inverse_affine=inverse_affine.to(dtype=torch.float32) ##speed up
                mask=kornia.geometry.transform.warp_affine(mask,inverse_affine,(h_up, w_up)) ##speed up
                # inv_soft_mask = mask[:, :, None]
                mask=mask.squeeze(0).squeeze(0) ##speed up
                
                # inv_soft_mask = mask[:, :, None]
                inv_soft_mask=mask.unsqueeze(2) ##speed up
                pasted_face = inv_restored
            else:  # use square parse maps
                mask = np.ones(self.face_size, dtype=np.float32)
                inv_mask = cv2.warpAffine(mask, inverse_affine, (w_up, h_up))
                # remove the black borders
                inv_mask_erosion = cv2.erode(
                    inv_mask, np.ones((int(2 * self.upscale_factor), int(2 * self.upscale_factor)), np.uint8))
                pasted_face = inv_mask_erosion[:, :, None] * inv_restored
                total_face_area = np.sum(inv_mask_erosion)  # // 3
                # compute the fusion edge based on the area of face
                w_edge = int(total_face_area**0.5) // 20
                erosion_radius = w_edge * 2
                inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
                blur_size = w_edge * 2
                inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)
                if len(upsample_img.shape) == 2:  # upsample_img is gray image
                    upsample_img = upsample_img[:, :, None]
                inv_soft_mask = inv_soft_mask[:, :, None]
            if len(upsample_img.shape) == 3 and upsample_img.shape[2] == 4:  # alpha channel
                alpha = upsample_img[:, :, 3:]
                upsample_img = inv_soft_mask * pasted_face + (1 - inv_soft_mask) * upsample_img[:, :, 0:3]
                upsample_img = np.concatenate((upsample_img, alpha), axis=2)
            else:

                pasted_face=torch.tensor(pasted_face).to(self.device) ##speed up part
                upsample_img=torch.tensor(upsample_img).to(self.device) ##speed up part
 
                upsample_img = inv_soft_mask * pasted_face + (1 - inv_soft_mask) * upsample_img
                upsample_img=torch.tensor(upsample_img).cpu().numpy()
            
        if np.max(upsample_img) > 256:  # 16-bit image
            upsample_img = upsample_img.astype(np.uint16)
        else:
            upsample_img = upsample_img.astype(np.uint8)
        if save_path is not None:
            path = os.path.splitext(save_path)[0]
            save_path = f'{path}.{self.save_ext}'
            imwrite(upsample_img, save_path)
        return upsample_img
        
    



class FAST_GFGGaner(GFPGANer):
    
    def __init__(self, model_path, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None, device=None):
        super().__init__(model_path=model_path,
            upscale=upscale,
            arch=arch,
            channel_multiplier=channel_multiplier,
            bg_upsampler=bg_upsampler)
        
        # initialize face helper
        self.face_helper = FAST_FaceRestoreHelper(upscale=upscale,face_size=512,crop_ratio=(1, 1),det_model='retinaface_resnet50',
                    save_ext='png',
                    use_parse=True,
                    device=self.device,
                    model_rootpath='gfpgan/weights')
        