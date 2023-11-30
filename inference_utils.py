import torch
import cv2
import os
import subprocess
import numpy as np
from models import audio
from draw_landmark import draw_landmarks
from tqdm import tqdm
from global_variable import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LandmarkDict(dict):# Makes a dictionary that behave like an object to represent each landmark
    def __init__(self, idx, x, y):
        self['idx'] = idx
        self['x'] = x
        self['y'] = y
    def __getattr__(self, name):
        try:
            return self[name]
        except:
            raise AttributeError(name)
    def __setattr__(self, name, value):
        self[name] = value


def summarize_landmark(edge_set):  # summarize all ficial landmarks used to construct edge
    landmarks = set()
    for a, b in edge_set:
        landmarks.add(a)
        landmarks.add(b)
    return landmarks


def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint
def load_model(model, path):
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        if k[:6] == 'module':
            new_k=k.replace('module.', '', 1)
        else:
            new_k =k
        new_s[new_k] = v
    model.load_state_dict(new_s)
    model = model.to(device)
    return model.eval()

def swap_masked_region(target_img, src_img, mask): #function used in post-process
    """From src_img crop masked region to replace corresponding masked region
    in target_img
    """  # swap_masked_region(src_frame, generated_frame, mask=mask_img)
    mask_img = cv2.GaussianBlur(mask, (21, 21), 11)
    mask1 = mask_img / 255
    mask1 = np.tile(np.expand_dims(mask1, axis=2), (1, 1, 3))
    img = src_img * mask1 + target_img * (1 - mask1)
    return img.astype(np.uint8)

def merge_face_contour_only(src_frame, generated_frame, face_region_coord, fa): #function used in post-process
    """Merge the face from generated_frame into src_frame
    """
    input_img = src_frame
    y1, y2, x1, x2 = 0, 0, 0, 0
    if face_region_coord is not None:
        y1, y2, x1, x2 = face_region_coord
        input_img = src_frame[y1:y2, x1:x2]
    ### 1) Detect the facial landmarks
    preds = fa.get_landmarks(input_img)[0]  # 68x2
    if face_region_coord is not None:
        preds += np.array([x1, y1])
    lm_pts = preds.astype(int)
    contour_idx = list(range(0, 17)) + list(range(17, 27))[::-1]
    contour_pts = lm_pts[contour_idx]
    ### 2) Make the landmark region mark image
    mask_img = np.zeros((src_frame.shape[0], src_frame.shape[1], 1), np.uint8)
    cv2.fillConvexPoly(mask_img, contour_pts, 255)
    ### 3) Do swap
    img = swap_masked_region(src_frame, generated_frame, mask=mask_img)
    return img


def reading(input_video_path,input_audio_path,temp_dir):
    print('Reading video frames ... from', input_video_path)
    if not os.path.isfile(input_video_path):
        raise ValueError('the input video file does not exist')
    elif input_video_path.split('.')[1] in ['jpg', 'png', 'jpeg']: #if input a single image for testing
        ori_background_frames = [cv2.imread(input_video_path)]
    else:
        video_stream = cv2.VideoCapture(input_video_path)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        if fps != 25:
            print(" input video fps:", fps,',converting to 25fps...')
            command = 'ffmpeg -y -i ' + input_video_path + ' -r 25 ' + '{}/temp_25fps.avi'.format(temp_dir)
            subprocess.call(command, shell=True)
            input_video_path = '{}/temp_25fps.avi'.format(temp_dir)
            video_stream.release()
            video_stream = cv2.VideoCapture(input_video_path)
            fps = video_stream.get(cv2.CAP_PROP_FPS)
        assert fps == 25

        ori_background_frames = [] #input videos frames (includes background as well as face)
        frame_idx = 0
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            ori_background_frames.append(frame)
            frame_idx = frame_idx + 1
    input_vid_len = len(ori_background_frames)

    ##(2) Extracting audio####
    if not input_audio_path.endswith('.wav'):
        command = 'ffmpeg -y -i {} -strict -2 {}'.format(input_audio_path, '{}/temp.wav'.format(temp_dir))
        subprocess.call(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        input_audio_path = '{}/temp.wav'.format(temp_dir)
    wav = audio.load_wav(input_audio_path, 16000)
    mel = audio.melspectrogram(wav)  # (H,W)   extract mel-spectrum
    ##read audio mel into list###
    mel_chunks = []  # each mel chunk correspond to 5 video frames, used to generate one video frame
    mel_idx_multiplier = 80. / fps
    mel_chunk_idx = 0
    while 1:
        start_idx = int(mel_chunk_idx * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            break
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])  # mel for generate one video frame
        mel_chunk_idx += 1
    # mel_chunks = mel_chunks[:(len(mel_chunks) // T) * T]
    return ori_background_frames,input_video_path,fps,input_vid_len,input_audio_path,mel_chunks,mel_chunk_idx



def detection(mp_face_mesh,all_landmarks_idx,pose_landmark_idx,content_landmark_idx,ori_background_frames,):
    boxes = []  #bounding boxes of human face
    lip_dists = [] #lip dists
    #we define the lip dist(openness): distance between the  midpoints of the upper lip and lower lip
    face_crop_results = []
    all_pose_landmarks, all_content_landmarks = [], []  #content landmarks include lip and jaw landmarks
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                            min_detection_confidence=0.5) as face_mesh:
        # (1) get bounding boxes and lip dist
        for frame_idx, full_frame in enumerate(ori_background_frames):
            h, w = full_frame.shape[0], full_frame.shape[1]
            results = face_mesh.process(cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                raise NotImplementedError  # not detect face
            face_landmarks = results.multi_face_landmarks[0]

            ## calculate the lip dist
            dx = face_landmarks.landmark[lip_index[0]].x - face_landmarks.landmark[lip_index[1]].x
            dy = face_landmarks.landmark[lip_index[0]].y - face_landmarks.landmark[lip_index[1]].y
            dist = np.linalg.norm((dx, dy))
            lip_dists.append((frame_idx, dist))

            # (1)get the marginal landmarks to crop face
            x_min,x_max,y_min,y_max = 999,-999,999,-999
            for idx, landmark in enumerate(face_landmarks.landmark):
                if idx in all_landmarks_idx:
                    if landmark.x < x_min:
                        x_min = landmark.x
                    if landmark.x > x_max:
                        x_max = landmark.x
                    if landmark.y < y_min:
                        y_min = landmark.y
                    if landmark.y > y_max:
                        y_max = landmark.y
            ##########plus some pixel to the marginal region##########
            #note:the landmarks coordinates returned by mediapipe range 0~1
            plus_pixel = 25
            x_min = max(x_min - plus_pixel / w, 0)
            x_max = min(x_max + plus_pixel / w, 1)

            y_min = max(y_min - plus_pixel / h, 0)
            y_max = min(y_max + plus_pixel / h, 1)
            y1, y2, x1, x2 = int(y_min * h), int(y_max * h), int(x_min * w), int(x_max * w)
            boxes.append([y1, y2, x1, x2])
        boxes = np.array(boxes)

        # (2)croppd face
        face_crop_results = [[image[y1:y2, x1:x2], (y1, y2, x1, x2)] \
                            for image, (y1, y2, x1, x2) in zip(ori_background_frames, boxes)]

        # (3)detect facial landmarks
        for frame_idx, full_frame in enumerate(ori_background_frames):
            h, w = full_frame.shape[0], full_frame.shape[1]
            results = face_mesh.process(cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                raise ValueError("not detect face in some frame!")  # not detect
            face_landmarks = results.multi_face_landmarks[0]



            pose_landmarks, content_landmarks = [], []
            for idx, landmark in enumerate(face_landmarks.landmark):
                if idx in pose_landmark_idx:
                    pose_landmarks.append((idx, w * landmark.x, h * landmark.y))
                if idx in content_landmark_idx:
                    content_landmarks.append((idx, w * landmark.x, h * landmark.y))

            # normalize landmarks to 0~1
            y_min, y_max, x_min, x_max = face_crop_results[frame_idx][1]  #bounding boxes
            pose_landmarks = [ \
                [idx, (x - x_min) / (x_max - x_min), (y - y_min) / (y_max - y_min)] for idx, x, y in pose_landmarks]
            content_landmarks = [ \
                [idx, (x - x_min) / (x_max - x_min), (y - y_min) / (y_max - y_min)] for idx, x, y in content_landmarks]
            all_pose_landmarks.append(pose_landmarks)
            all_content_landmarks.append(content_landmarks)
    return boxes, lip_dists, face_crop_results, all_pose_landmarks, all_content_landmarks

# smooth landmarks
def get_smoothened_landmarks(all_landmarks, windows_T=1):
    for i in range(len(all_landmarks)):  # frame i
        if i + windows_T > len(all_landmarks):
            window = all_landmarks[len(all_landmarks) - windows_T:]
        else:
            window = all_landmarks[i: i + windows_T]
        #####
        for j in range(len(all_landmarks[i])):  # landmark j
            all_landmarks[i][j][1] = np.mean([frame_landmarks[j][1] for frame_landmarks in window])  # x
            all_landmarks[i][j][2] = np.mean([frame_landmarks[j][2] for frame_landmarks in window])  # y
    return all_landmarks

def chg_dt(lip_dists,input_vid_len,all_pose_landmarks,all_content_landmarks):
    # print("changing_datatype")
    ##randomly select N_l reference landmarks for landmark transformer##
    dists_sorted = sorted(lip_dists, key=lambda x: x[1])
    lip_dist_idx = np.asarray([idx for idx, dist in dists_sorted])  #the frame idxs sorted by lip openness

    Nl_idxs = [lip_dist_idx[int(i)] for i in torch.linspace(0, input_vid_len - 1, steps=Nl)]
    Nl_pose_landmarks, Nl_content_landmarks = [], []  #Nl_pose + Nl_content=Nl reference landmarks
    for reference_idx in Nl_idxs:
        frame_pose_landmarks = all_pose_landmarks[reference_idx]
        frame_content_landmarks = all_content_landmarks[reference_idx]
        Nl_pose_landmarks.append(frame_pose_landmarks)
        Nl_content_landmarks.append(frame_content_landmarks)

    Nl_pose = torch.zeros((Nl, 2, 74))  # 74 landmark
    Nl_content = torch.zeros((Nl, 2, 57))  # 57 landmark
    for idx in range(Nl):
        #arrange the landmark in a certain order, since the landmark index returned by mediapipe is is chaotic
        Nl_pose_landmarks[idx] = sorted(Nl_pose_landmarks[idx],
                                        key=lambda land_tuple: ori_sequence_idx.index(land_tuple[0]))
        Nl_content_landmarks[idx] = sorted(Nl_content_landmarks[idx],
                                        key=lambda land_tuple: ori_sequence_idx.index(land_tuple[0]))

        Nl_pose[idx, 0, :] = torch.FloatTensor(
            [Nl_pose_landmarks[idx][i][1] for i in range(len(Nl_pose_landmarks[idx]))])  # x
        Nl_pose[idx, 1, :] = torch.FloatTensor(
            [Nl_pose_landmarks[idx][i][2] for i in range(len(Nl_pose_landmarks[idx]))])  # y
        Nl_content[idx, 0, :] = torch.FloatTensor(
            [Nl_content_landmarks[idx][i][1] for i in range(len(Nl_content_landmarks[idx]))])  # x
        Nl_content[idx, 1, :] = torch.FloatTensor(
            [Nl_content_landmarks[idx][i][2] for i in range(len(Nl_content_landmarks[idx]))])  # y
    Nl_content = Nl_content.unsqueeze(0)  # (1,Nl, 2, 57)
    Nl_pose = Nl_pose.unsqueeze(0)  # (1,Nl,2,74)
    return lip_dist_idx, Nl_content, Nl_pose

def draw_sketches(drawing_spec,lip_dist_idx,input_vid_len,face_crop_results,all_pose_landmarks,all_content_landmarks):
    ##select reference images and draw sketches for rendering according to lip openness##
    ref_img_idx = [int(lip_dist_idx[int(i)]) for i in torch.linspace(0, input_vid_len - 1, steps=ref_img_N)]
    ref_imgs = [face_crop_results[idx][0] for idx in ref_img_idx]
    ## (N,H,W,3)
    ref_img_pose_landmarks, ref_img_content_landmarks = [], []
    for idx in ref_img_idx:
        ref_img_pose_landmarks.append(all_pose_landmarks[idx])
        ref_img_content_landmarks.append(all_content_landmarks[idx])

    ref_img_pose = torch.zeros((ref_img_N, 2, 74))  # 74 landmark
    ref_img_content = torch.zeros((ref_img_N, 2, 57))  # 57 landmark

    for idx in range(ref_img_N):
        ref_img_pose_landmarks[idx] = sorted(ref_img_pose_landmarks[idx],
                                            key=lambda land_tuple: ori_sequence_idx.index(land_tuple[0]))
        ref_img_content_landmarks[idx] = sorted(ref_img_content_landmarks[idx],
                                                key=lambda land_tuple: ori_sequence_idx.index(land_tuple[0]))
        ref_img_pose[idx, 0, :] = torch.FloatTensor(
            [ref_img_pose_landmarks[idx][i][1] for i in range(len(ref_img_pose_landmarks[idx]))])  # x
        ref_img_pose[idx, 1, :] = torch.FloatTensor(
            [ref_img_pose_landmarks[idx][i][2] for i in range(len(ref_img_pose_landmarks[idx]))])  # y

        ref_img_content[idx, 0, :] = torch.FloatTensor(
            [ref_img_content_landmarks[idx][i][1] for i in range(len(ref_img_content_landmarks[idx]))])  # x
        ref_img_content[idx, 1, :] = torch.FloatTensor(
            [ref_img_content_landmarks[idx][i][2] for i in range(len(ref_img_content_landmarks[idx]))])  # y

    ref_img_full_face_landmarks = torch.cat([ref_img_pose, ref_img_content], dim=2).cpu().numpy()  # (N,2,131)
    ref_img_sketches = []
    for frame_idx in range(ref_img_full_face_landmarks.shape[0]):  # N
        full_landmarks = ref_img_full_face_landmarks[frame_idx]  # (2,131)
        h, w = ref_imgs[frame_idx].shape[0], ref_imgs[frame_idx].shape[1]
        drawn_sketech = np.zeros((int(h * img_size / min(h, w)), int(w * img_size / min(h, w)), 3))
        mediapipe_format_landmarks = [LandmarkDict(ori_sequence_idx[full_face_landmark_sequence[idx]], full_landmarks[0, idx],
                                                full_landmarks[1, idx]) for idx in range(full_landmarks.shape[1])]
        drawn_sketech = draw_landmarks(drawn_sketech, mediapipe_format_landmarks, connections=FACEMESH_CONNECTION,
                                    connection_drawing_spec=drawing_spec)
        drawn_sketech = cv2.resize(drawn_sketech, (img_size, img_size))  # (128, 128, 3)
        ref_img_sketches.append(drawn_sketech)
    ref_img_sketches = torch.FloatTensor(np.asarray(ref_img_sketches) / 255.0).cuda().unsqueeze(0).permute(0, 1, 4, 2, 3)
    # (1,N, 3, 128, 128)
    ref_imgs = [cv2.resize(face.copy(), (img_size, img_size)) for face in ref_imgs]
    ref_imgs = torch.FloatTensor(np.asarray(ref_imgs) / 255.0).unsqueeze(0).permute(0, 1, 4, 2, 3).cuda()
    # (1,N,3,H,W)

    return ref_imgs,ref_img_sketches,ref_imgs


def prepare_output_stream(ori_background_frames,temp_dir,mel_chunks,input_vid_len,fps):
    frame_h, frame_w = ori_background_frames[0].shape[:-1]
    out_stream = cv2.VideoWriter('{}/result.mp4'.format(temp_dir), cv2.VideoWriter_fourcc(*'avc1'), fps,
                                (frame_w, frame_h))  # +frame_h*3


    ##generate final face image and output video##
    input_mel_chunks_len = len(mel_chunks)
    input_frame_sequence = torch.arange(input_vid_len).tolist()
    #the input template video may be shorter than audio
    #in this case we repeat the input template video as following
    num_of_repeat=input_mel_chunks_len//input_vid_len+1
    input_frame_sequence = input_frame_sequence + list(reversed(input_frame_sequence))
    input_frame_sequence=input_frame_sequence*((num_of_repeat+1)//2)

    return frame_h,frame_w,out_stream,input_mel_chunks_len,input_frame_sequence

def render_loop(landmark_generator_model, renderer, drawing_spec,fa,temp_dir, input_mel_chunks_len, mel_chunks,
                 input_frame_sequence, face_crop_results, all_pose_landmarks, ori_background_frames,
                 frame_w, frame_h, ref_imgs, ref_img_sketches, out_stream, input_audio_path,
                 outfile_path, Nl_content, Nl_pose):
    for batch_idx, batch_start_idx in tqdm(enumerate(range(0, input_mel_chunks_len - 2, 1)),total=len(range(0, input_mel_chunks_len - 2, 1))):
        # preprocessing_st = time.time()
        T_input_frame, T_ori_face_coordinates = [], []
        #note: input_frame include background as well as face
        T_mel_batch, T_crop_face,T_pose_landmarks = [], [],[]

        # (1) for each batch of T frame, generate corresponding landmarks using landmark generator
        for mel_chunk_idx in range(batch_start_idx, batch_start_idx + T):  # for each T frame
            # 1 input audio
            T_mel_batch.append(mel_chunks[max(0, mel_chunk_idx - 2)])

            # 2.input face
            input_frame_idx = int(input_frame_sequence[mel_chunk_idx])
            face, coords = face_crop_results[input_frame_idx]
            T_crop_face.append(face)
            T_ori_face_coordinates.append((face, coords))  ##input face
            # 3.pose landmarks
            T_pose_landmarks.append(all_pose_landmarks[input_frame_idx])
            # 3.background
            T_input_frame.append(ori_background_frames[input_frame_idx].copy())
        T_mels = torch.FloatTensor(np.asarray(T_mel_batch)).unsqueeze(1).unsqueeze(0)  # 1,T,1,h,w
        #prepare pose landmarks
        T_pose = torch.zeros((T, 2, 74))  # 74 landmark
        for idx in range(T):
            T_pose_landmarks[idx] = sorted(T_pose_landmarks[idx],
                                        key=lambda land_tuple: ori_sequence_idx.index(land_tuple[0]))
            T_pose[idx, 0, :] = torch.FloatTensor(
                [T_pose_landmarks[idx][i][1] for i in range(len(T_pose_landmarks[idx]))])  # x
            T_pose[idx, 1, :] = torch.FloatTensor(
                [T_pose_landmarks[idx][i][2] for i in range(len(T_pose_landmarks[idx]))])  # y
        T_pose = T_pose.unsqueeze(0)  # (1,T, 2,74)
        # preprocessing_et = time.time()
        # preprocessing_time.append(preprocessing_et-preprocessing_st)




        # landmark  generator inference
        # landmark_gen_st = time.time()
        Nl_pose, Nl_content = Nl_pose.cuda(), Nl_content.cuda() # (Nl,2,74)  (Nl,2,57)
        T_mels, T_pose = T_mels.cuda(), T_pose.cuda()
        with torch.no_grad():  # require    (1,T,1,hv,wv)(1,T,2,74)(1,T,2,57)
            predict_content = landmark_generator_model(T_mels, T_pose, Nl_pose, Nl_content)  # (1*T,2,57)
        T_pose = torch.cat([T_pose[i] for i in range(T_pose.size(0))], dim=0)  # (1*T,2,74)
        T_predict_full_landmarks = torch.cat([T_pose, predict_content], dim=2).cpu().numpy()  # (1*T,2,131)
        # landmark_gen_et = time.time()
        # landmark_gen_time.append(landmark_gen_et - landmark_gen_st)





        #1.draw target sketch
        # target_sketches_st = time.time()
        T_target_sketches = []
        for frame_idx in range(T):
            full_landmarks = T_predict_full_landmarks[frame_idx]  # (2,131)
            h, w = T_crop_face[frame_idx].shape[0], T_crop_face[frame_idx].shape[1]
            drawn_sketech = np.zeros((int(h * img_size / min(h, w)), int(w * img_size / min(h, w)), 3))
            mediapipe_format_landmarks = [LandmarkDict(ori_sequence_idx[full_face_landmark_sequence[idx]]
                                                    , full_landmarks[0, idx], full_landmarks[1, idx]) for idx in
                                        range(full_landmarks.shape[1])]
            drawn_sketech = draw_landmarks(drawn_sketech, mediapipe_format_landmarks, connections=FACEMESH_CONNECTION,
                                        connection_drawing_spec=drawing_spec)
            drawn_sketech = cv2.resize(drawn_sketech, (img_size, img_size))  # (128, 128, 3)
            if frame_idx == 2:
                show_sketch = cv2.resize(drawn_sketech, (frame_w, frame_h)).astype(np.uint8)
            T_target_sketches.append(torch.FloatTensor(drawn_sketech) / 255)
        T_target_sketches = torch.stack(T_target_sketches, dim=0).permute(0, 3, 1, 2)  # (T,3,128, 128)
        target_sketches = T_target_sketches.unsqueeze(0).cuda()  # (1,T,3,128, 128)

        # 2.lower-half masked face
        ori_face_img = torch.FloatTensor(cv2.resize(T_crop_face[2], (img_size, img_size)) / 255).permute(2, 0, 1).unsqueeze(
            0).unsqueeze(0).cuda()  #(1,1,3,H, W)
        
        # target_sketches_et = time.time()
        # target_sketches_time.append(target_sketches_et - target_sketches_st)


        # 3. render the full face
        # require (1,1,3,H,W)   (1,T,3,H,W)  (1,N,3,H,W)   (1,N,3,H,W)  (1,1,1,h,w)
        # return  (1,3,H,W)
        # renderer_st = time.time()
        with torch.no_grad():
            generated_face, _, _, _ = renderer(ori_face_img, target_sketches, ref_imgs, ref_img_sketches,
                                                        T_mels[:, 2].unsqueeze(0))  # T=1
        gen_face = (generated_face.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # (H,W,3)
        # renderer_et = time.time()
        # renderer_time.append(renderer_et - renderer_st)




        # 4. paste each generated face
        # postprocessing_st = time.time()
        y1, y2, x1, x2 = T_ori_face_coordinates[2][1]  # coordinates of face bounding box
        original_background = T_input_frame[2].copy()
        T_input_frame[2][y1:y2, x1:x2] = cv2.resize(gen_face,(x2 - x1, y2 - y1))  #resize and paste generated face
        # 5. post-process
        full = merge_face_contour_only(original_background, T_input_frame[2], T_ori_face_coordinates[2][1],fa)   #(H,W,3)
        # 6.output
        out_stream.write(full)
        if batch_idx == 0:
            out_stream.write(full)
            out_stream.write(full)
        # postprocessing_et = time.time()
        # postprocessing_time.append(postprocessing_et - postprocessing_st)
        # complete_time.append(postprocessing_et - preprocessing_st)




    out_stream.release()
    command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(input_audio_path, '{}/result.mp4'.format(temp_dir), outfile_path)
    subprocess.call(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print("succeed output results to:", outfile_path)
    print('{}/result.avi'.format(temp_dir))
    print(input_audio_path)
    return outfile_path

