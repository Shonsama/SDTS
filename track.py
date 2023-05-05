import argparse
from Small_obejct.face import face_reid 
import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import cv2
import sys
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from numpy import random

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov7') not in sys.path:
    sys.path.append(str(ROOT / 'yolov7'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadImages, LoadStreams
from yolov7.utils.general import (check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, check_file)
from yolov7.utils.torch_utils import select_device, time_synchronized
from yolov7.utils.plots import plot_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT
import glob


VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes


@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov7.pt',  # model.pt path(s),
        strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=0,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        visualize=False,  # visualize features
        half=False,  # use FP16 half-precision inference
):
    # Load model
    device = select_device(device)
    save_dir = 'tracks'
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    model = attempt_load(Path(yolo_weights), map_location=device)  # load FP32 model
    names, = model.names,
    stride = model.stride.max().cpu().numpy()  # model stride
    imgsz = check_img_size(imgsz[0], s=stride)  # check image size

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    nr_sources = 1

    # initialize StrongSORT
    cfg = get_config()
    cfg.merge_from_file(opt.config_strongsort)

    # Create as many strong sort instances as there are video sources
    strongsort_list = []
    for i in range(nr_sources):
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                half,
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA,
            )
        )
        strongsort_list[i].model.warmup()
    outputs = [None] * nr_sources
    

    # Run tracking
    seen = 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    bodies = {}
    boxes = {}
    for frame_idx, (path, im, im0s, vid_cap) in enumerate(dataset):
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        pred = model(im)
        # Apply NMS
        pred = non_max_suppression(pred[0], conf_thres, iou_thres, classes, agnostic_nms)

        box = {}
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            # video file
            curr_frames[i] = im0

            if cfg.STRONGSORT.ECC:  # camera motion compensation
                strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]
                
                # pass detections to strongsort
                outputs[i] = strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    for j, (output, conf) in enumerate(zip(outputs[i], confs)):
                        id = int(output[4])
                        box[id] = output[:4]
                        body = im0[int(output[1]):int(output[3]), int(output[0]):int(output[2])]
                        faces = face_cascade.detectMultiScale(body, scaleFactor=1.1, minNeighbors=5)
                        temp = {}
                        temp['frame'] = body
                        temp['detect'] = False if id != 0 else True
                        if id in bodies:
                            if len(faces) > 0:
                                bodies[id] = temp
                        else:
                            bodies[id] = temp

            else:
                strongsort_list[i].increment_ages()
                print('No detections')

            prev_frames[i] = curr_frames[i]
        boxes[frame_idx] = box
    return bodies, boxes


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str, default='./track', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, default=0, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    return opt

def get_video_urls(source):
    return glob.glob(os.path.join(source, '*.mp4'), recursive=True)

def recgonize_Face(videos):
    people = []
    suspect_frame = cv2.imread('./suspect_frame.jpg')
    for key,video in videos.items():
        for key_body,body in video['bodies'].items():
            if face_reid(body['frame'], suspect_frame):
                body['detect'] = True
                people.append(body['frame'])
                video['detect'] = True

    return videos, people

def recgonize_Body(videos, people):
    for key,video in videos.items():
        for key_body,body in video['bodies'].items():
            if not body['detect']:
                for person in people:
                    if body_reid(person, body['frame']):
                        body['detect'] = True
                        break
    return videos

def draw_boxes(video, video_url):
    # Load model
    device = select_device('')
    save_dir = './tracks'
    model = attempt_load(Path(WEIGHTS / 'yolov7.pt'), map_location=device)  # load FP32 model
    names, = model.names,
    stride = model.stride.max().cpu().numpy()  # model stride
    imgsz = check_img_size(640, s=stride)  # check image size
    # Dataloader
    dataset = LoadImages(video_url, img_size=imgsz, stride=stride)

    # Define output video writer
    vid_name = Path(video_url).stem + '_tracked.mp4'
    save_path = Path(save_dir) / vid_name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = None
    for frame_idx, (path, im, im0s, vid_cap) in enumerate(dataset):
        if out == None:
            out = cv2.VideoWriter(str(save_path), fourcc, 25, (im0s.shape[1], im0s.shape[0]))

        for key_body, body in video['boxes'][frame_idx].items():
            if video['bodies'][key_body]['detect']:
                x1, y1, x2, y2 = body #body[989.0, 313.0, 1124.0, 672.0]
                im0s = cv2.rectangle(im0s, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(im0s, str(key_body), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(im0s)  # Write frame to output video

    out.release()  # Release output video writer

    
    
def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    # opt = parse_opt()
    video_urls = get_video_urls(opt.source)
    print(video_urls)
    videos = {}
    for video_url in video_urls:
        opt.source = video_url
        bodies, boxes = run(**vars(opt))
        temp = {
            'detect': False,
            'bodies': bodies,
            'boxes': boxes
        }
        videos[video_url] = temp
    
    videos, people = recgonize_Face(videos)
    # videos = recgonize_Body(videos, people)
    for video_url in video_urls:
        if videos[video_url]['detect']:
            draw_boxes(videos[video_url], video_url)
    
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)