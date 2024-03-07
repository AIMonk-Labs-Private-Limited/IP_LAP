import os
import cv2
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pathlib import Path
try:
    from IP_LAP.bgremoval_package.src.models.modnet import MODNet
except:
    # import sys
    # print(os.path.join(os.path.dirname(__file__),".."))
    # sys.path.append("")
    from bgremoval_package.src.models.modnet import MODNet
    


#Check if GPU available or not
GPU = True if torch.cuda.device_count() > 0 else False


#Preparing path for pretrained checkpoint
filepath = os.path.dirname(os.path.abspath(__file__))
CKPT_FOLDER = Path(filepath).parent.parent.parent
PRETRAINED_CKPT = os.path.join(CKPT_FOLDER,"checkpoints/bgremoval/modnet_webcam_portrait_matting.ckpt")
print(PRETRAINED_CKPT)
torch_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
def load_model():
    
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)

    
    if GPU:
        print('Use GPU...')
        modnet = modnet.cuda()
        modnet.load_state_dict(torch.load(PRETRAINED_CKPT))
    else:
        print('Use CPU...')
        modnet.load_state_dict(torch.load(PRETRAINED_CKPT, map_location=torch.device('cpu')))
    modnet.eval()
    return modnet

def matting_list(video,output_folder):
    
    modnet = load_model()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    frame=video[0]

    num_frame = len(video)
    h, w = frame.shape[:2]
    if w >= h:
        rh = 512
        rw = int(w / h * 512)
    else:
        rw = 512
        rh = int(h / w * 512)
    rh = rh - rh % 32
    rw = rw - rw % 32


    print('Start matting...')
    for c in tqdm(range(int(num_frame))):
        frame=video[c]
        # frame_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_np = cv2.resize(frame, (rw, rh), cv2.INTER_AREA)

        frame_PIL = Image.fromarray(frame_np)
        frame_tensor = torch_transforms(frame_PIL)
        frame_tensor = frame_tensor[None, :, :, :]
        GPU = True if torch.cuda.device_count() > 0 else False
        if GPU:
            frame_tensor = frame_tensor.cuda()

        with torch.no_grad():
            _, _, matte_tensor = modnet(frame_tensor, True)

        matte_tensor = matte_tensor.repeat(1, 3, 1, 1)
        matte_np = matte_tensor[0].data.cpu().numpy().transpose(1, 2, 0)
        view_np_alpha = matte_np * np.full(frame_np.shape, 255.0)
        view_np_alpha = cv2.cvtColor(view_np_alpha.astype(np.uint8), cv2.COLOR_RGB2BGR)
        view_np_alpha = cv2.resize(view_np_alpha, (w, h))


        view_np = matte_np * frame_np + (1 - matte_np) * np.full(frame_np.shape, 255.0)
        view_np = cv2.cvtColor(view_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
        view_np = cv2.resize(view_np, (w, h))
        view_np_alpha=view_np_alpha[:,:,1]
        four_channel_frame = np.dstack((view_np, view_np_alpha))
        output_path = os.path.join(output_folder, f'output_{c:04d}.png')
        cv2.imwrite(output_path, four_channel_frame)


def matting(video,output_folder,modnet, alpha_matte=False, fps=25,MODAL_FLAG=False):
    # video capture
    # modnet = load_model()
    modal_matt_arr=[]
    if not MODAL_FLAG:
        vc = cv2.VideoCapture(video)
        if vc.isOpened():
            rval, frame = vc.read()
        else:
            rval = False
        if not rval:
            print('Failed to read the video: {0}'.format(video))
            exit()
        
        
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # if not MODAL_FLAG:
    #     if vc.isOpened():
    #         rval, frame = vc.read()
    #     else:
    #         rval = False

    # if not rval:
    #     print('Failed to read the video: {0}'.format(video))
    #     exit()
    if not MODAL_FLAG:
        num_frame = vc.get(cv2.CAP_PROP_FRAME_COUNT)
        h, w = frame.shape[:2]
    else:
        num_frame = len(video)
        
        h, w = video[0].shape[:2]
    
    if w >= h:
        rh = 512
        rw = int(w / h * 512)
    else:
        rw = 512
        rh = int(h / w * 512)
    rh = rh - rh % 32
    rw = rw - rw % 32

    # video writer
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video_writer = cv2.VideoWriter(result, fourcc, fps, (w, h))
    # video_alpha_writer = cv2.VideoWriter(result_alpha, fourcc, fps, (w, h))


    print('Start matting...')
    with tqdm(range(int(num_frame)))as t:
        for c in t:
            if MODAL_FLAG:
                frame=video[c]
            frame_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_np = cv2.resize(frame_np, (rw, rh), cv2.INTER_AREA)

            frame_PIL = Image.fromarray(frame_np)
            frame_tensor = torch_transforms(frame_PIL)
            frame_tensor = frame_tensor[None, :, :, :]
            if GPU:
                frame_tensor = frame_tensor.cuda()

            with torch.no_grad():
                _, _, matte_tensor = modnet(frame_tensor, True)

            matte_tensor = matte_tensor.repeat(1, 3, 1, 1)
            matte_np = matte_tensor[0].data.cpu().numpy().transpose(1, 2, 0)
            view_np_alpha = matte_np * np.full(frame_np.shape, 255.0)
            view_np_alpha = cv2.cvtColor(view_np_alpha.astype(np.uint8), cv2.COLOR_RGB2BGR)
            view_np_alpha = cv2.resize(view_np_alpha, (w, h))
            # video_alpha_writer.write(view_np_alpha)

            view_np = matte_np * frame_np + (1 - matte_np) * np.full(frame_np.shape, 255.0)
            view_np = cv2.cvtColor(view_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
            view_np = cv2.resize(view_np, (w, h))
            # video_writer.write(view_np)
            view_np_alpha=view_np_alpha[:,:,1]
            four_channel_frame = np.dstack((view_np, view_np_alpha))
            output_path = os.path.join(output_folder, f'output_{c:04d}.png')
            if not MODAL_FLAG:
                cv2.imwrite(output_path, four_channel_frame)
                rval, frame = vc.read()
            else:
                modal_matt_arr.append(four_channel_frame)
                
            c += 1
    return modal_matt_arr

    # video_writer.release()
    # video_alpha_writer.release()
    # print('Save the result video to {0}'.format(result))
    # print('Save the alpha_result video to {0}'.format(result_alpha))


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='input video file')
    parser.add_argument('--result-type', type=str, default='fg', choices=['fg', 'matte'], 
                        help='matte - save the alpha matte; fg - save the foreground')
    parser.add_argument('--fps', type=int, default=25, help='fps of the result video')

    print('Get CMD Arguments...')
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print('Cannot find the input video: {0}'.format(args.video))
        exit()

    print('Load pre-trained MODNet...')
    pretrained_ckpt = './pretrained/modnet_webcam_portrait_matting.ckpt'
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)

    GPU = True if torch.cuda.device_count() > 0 else False
    if GPU:
        print('Use GPU...')
        modnet = modnet.cuda()
        modnet.load_state_dict(torch.load(pretrained_ckpt))
    else:
        print('Use CPU...')
        modnet.load_state_dict(torch.load(pretrained_ckpt, map_location=torch.device('cpu')))
    modnet.eval()

    result = os.path.splitext(args.video)[0] + '_{0}.mp4'.format('fg')
    result_alpha = os.path.splitext(args.video)[0] + '_{0}.mp4'.format('matte')
    alpha_matte = True if args.result_type == 'matte' else False
    output_folder="output"
    matting(args.video,output_folder, result,result_alpha, alpha_matte, args.fps,)

    end_time = time.time()
    print(f"time taken:{end_time - start_time}")

