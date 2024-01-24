import numpy as np
import cv2, os, argparse
import subprocess
from tqdm import tqdm
from models import Renderer
import torch
import torch.multiprocessing as multiprocessing
multiprocessing.set_start_method('spawn')
from models import Landmark_generator as Landmark_transformer
import face_alignment
from models import audio
from draw_landmark import draw_landmarks
import mediapipe as mp
from global_variable import *
from inference_utils import *

import time


# from gfpgan import GFPGANer
from pathlib import Path
import sys
# GFPGAN_CKPT_PATH = os.path.join(
#     # path to the dir that contains checkpoint dir
#     str(Path(os.path.dirname(os.path.abspath(__file__))).parent.parent.parent),
#     "sadtalker/1/checkpoints"
# )
# BG_REMOVAL_PACKAGE_PATH = os.path.join(str(Path(os.path.dirname(os.path.abspath(__file__))).parent), "bgremoval_package")
# sys.path.append(BG_REMOVAL_PACKAGE_PATH)
from bgremoval_package.demo.run import matting

from bgremoval_package.demo.run import load_model as load_model_modenet

MP_QUEUE_SIZE = 50

class IPLAP_tritoninference:
    
    def __init__(self,landmark_gen_checkpoint_path,renderer_checkpoint_path):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)
        # self.temp_dir = temp_dir
        

        self.all_landmarks_idx = summarize_landmark(FACEMESH_CONNECTION)
        self.pose_landmark_idx = \
            summarize_landmark(FACEMESH_NOSE.union(*[FACEMESH_RIGHT_EYEBROW, FACEMESH_RIGHT_EYE,
                                                    FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW, ])).union(
                [162, 127, 234, 93, 389, 356, 454, 323])
        # pose landmarks are landmarks of the upper-half face(eyes,nose,cheek) that represents the pose information

        self.content_landmark_idx = self.all_landmarks_idx - self.pose_landmark_idx
        # content_landmark include landmarks of lip and jaw which are inferred from audio
        # print(" landmark_generator_model loaded from : ", landmark_gen_checkpoint_path)
        # print(" renderer loaded from : ", renderer_checkpoint_path)
        self.landmark_generator_model = load_model(
            model=Landmark_transformer(T=T, d_model=512, nlayers=4, nhead=4, dim_feedforward=1024, dropout=0.1),
            path=landmark_gen_checkpoint_path)
        self.renderer = load_model(model=Renderer(), path=renderer_checkpoint_path)
        self.modnet=load_model_modenet()
        
        # launch daemon processes for inf and post processing
        self.launch_processes()
        
    def launch_processes(self):
        self.input_queue = multiprocessing.Queue(maxsize=MP_QUEUE_SIZE)
        self.inf_postproc_queue = multiprocessing.Queue(maxsize=MP_QUEUE_SIZE)
        self.postproc_enh_queue = multiprocessing.Queue(maxsize=MP_QUEUE_SIZE)
        self.output_queue =  multiprocessing.Queue(maxsize=MP_QUEUE_SIZE)
        self.mp_lock = multiprocessing.Lock()
        
        # Shared float Value type objects to keep track of timings
        self.model_inf_time = multiprocessing.Value('f', 0.0)
        self.post_proc_time = multiprocessing.Value('f', 0.0)
        self.face_enhc_time = multiprocessing.Value('f', 0.0)
        
        self.model_inf_process = multiprocessing.Process(
            target=model_inference_process, 
            args=(self.input_queue, self.inf_postproc_queue, self.landmark_generator_model,
                 self.renderer, self.drawing_spec, self.model_inf_time),
            daemon=True     # will get exited automatically once main process is completed/killed
        )
        
        
        self.face_enhancer_process = multiprocessing.Process(
            target=face_enhancer_process,
            args=(self.inf_postproc_queue, self.postproc_enh_queue, self.face_enhc_time),
            daemon=True
        )
        
        self.postproc_process = multiprocessing.Process(
            target=postprocessing_process,
            args=(self.postproc_enh_queue, self.output_queue, self.mp_lock, self.post_proc_time),
            daemon=True     # will get exited automatically once main process is completed/killed
        )
        
        
        self.model_inf_process.start()
        self.face_enhancer_process.start()
        self.postproc_process.start()
    
    def infer(self,input_video_path,input_audio_path,temp_dir,bgremoval,avatar_name):

        # result_out_dir = os.path.dirname(temp_dir)
        t0 = time.time()
        outfile_path = os.path.join(temp_dir,
                            '{}_N_{}_Nl_{}.mp4'.format(input_video_path.split('/')[-1][:-4] + 'result', ref_img_N, Nl))

        ori_background_frames, input_video_path, fps, input_vid_len, input_audio_path, mel_chunks, mel_chunk_idx = reading(
            input_video_path, input_audio_path, temp_dir
        )

        boxes, lip_dists, face_crop_results, all_pose_landmarks, all_content_landmarks = detection(
            self.mp_face_mesh, self.all_landmarks_idx, self.pose_landmark_idx, self.content_landmark_idx, ori_background_frames
        )

        all_pose_landmarks = get_smoothened_landmarks(all_pose_landmarks, windows_T=1)
        all_content_landmarks = get_smoothened_landmarks(all_content_landmarks, windows_T=1)

        lip_dist_idx, Nl_content, Nl_pose = chg_dt(lip_dists, input_vid_len, all_pose_landmarks, all_content_landmarks)

        ref_imgs, ref_img_sketches, ref_imgs = draw_sketches(
            self.drawing_spec, lip_dist_idx, input_vid_len, face_crop_results, all_pose_landmarks, all_content_landmarks
        )

        frame_h, frame_w, out_stream, input_mel_chunks_len, input_frame_sequence = prepare_output_stream(
            ori_background_frames, temp_dir, mel_chunks, input_vid_len, fps
        )

        output_path, render_loop_time = global_render_loop(
            self.input_queue, self.output_queue, input_mel_chunks_len, mel_chunks, 
            input_frame_sequence, face_crop_results, all_pose_landmarks, ori_background_frames,
            frame_w, frame_h, ref_imgs, ref_img_sketches, Nl_content, Nl_pose, 
            out_stream, input_audio_path, temp_dir, outfile_path,avatar_name
        )

        timing_dict = {
            'total_time': time.time() - t0,
            'model_inf_time': self.model_inf_time.value,
            'post_proc_time': self.post_proc_time.value,
            'face_enhc_time': self.face_enhc_time.value,
            'render_loop_time': render_loop_time
        }
        # print timing info
        print("\n".join([f"{key}: {value}" for key, value in timing_dict.items()]))
        if bgremoval:
            output_path=os.path.join(temp_dir,"matts")
            os.makedirs(output_path,exist_ok=True)
            matting(outfile_path,output_path,self.modnet)
            
            return output_path
        else:
            return outfile_path


















