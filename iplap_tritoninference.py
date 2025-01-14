import numpy as np
import cv2, os, argparse
import subprocess
from tqdm import tqdm
from models import Renderer
import torch
from models import Landmark_generator as Landmark_transformer
import face_alignment
from models import audio
from draw_landmark import draw_landmarks
import mediapipe as mp
from global_variable import *
from inference_utils import *



class IPLAP_tritoninference:
    
    def __init__(self,landmark_gen_checkpoint_path,renderer_checkpoint_path):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cuda')
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

    
    def infer(self,input_video_path,input_audio_path,temp_dir):

        # result_out_dir = os.path.dirname(temp_dir)
        # import pdb;pdb.set_trace()

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
        # import pdb;pdb.set_trace()
        outfile_path =render_loop(
            self.landmark_generator_model, self.renderer, self.drawing_spec, self.fa,temp_dir, input_mel_chunks_len, mel_chunks,
            input_frame_sequence, face_crop_results, all_pose_landmarks, ori_background_frames,
            frame_w, frame_h, ref_imgs, ref_img_sketches, out_stream, input_audio_path,
            outfile_path, Nl_content, Nl_pose
        )

        return outfile_path


















