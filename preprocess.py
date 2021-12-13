# the contents of this file are based on interpolate.py
import argparse
from tqdm import tqdm
from torchvision.io import read_video, write_video

parsare = argparse.ArgumentParser()

parser.add_argument("--input_video", type=str, required=True, help="path to input vid")
args = parser.parse_args()
input_video = args.input_video


videoTensor, _, metadata = read_video(input_video)
print("input FPS is: ", metadata['video_fps'])

videoTensor = videoTensor.float().permute(3,0,1,3) / 255.0
print("video Tensor shape:", videoTensor.shape)


#idxs =torch.Tensor(range(len(videoTensor))).type(torch.long).view(1,-1).unfold(1,size=nbr_frame,step=1).squeeze(0)

frame_indices = [[i,i+1,i+2,i+3] for i in range(len(videoTensor)-3)]

frames = torch.unbind(videoTensor, 1)
print(frames.shape)
n_inputs = len(frames)
width = n_outputs + 1

outputs = [frames[indices[0][0]], frames[indices[0][1]]] # store the images to the left of first inferred frames


for indices in tqdm(frame_indices):
    inputs = [frames[i].cuda().unsqueeze(0) for i in indices]
    with torch.no_grad():
        out_frames = [out_frame.squeeze(0).cpu().data for out_frame in model(inputs)]  # call FLAVR_arch model
    outputs.extend(out_frames).append(inputs[2].squeeze(0).cpu().data)


def make_img(tensor):
    data = tensor.data.mul(255.0).clamp(0,255).round().permute(1,2,0).cpu().numpy().astype(np.uint8)
    return cv2.cvtColor(data, cv2.COLOR_RGBR2BGR)

out_vid = [make_image(out_frame) for out_frame in outputs]

