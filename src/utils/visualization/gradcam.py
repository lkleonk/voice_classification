import io
from typing import Any, List, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import src.utils.toolkit.cuda_handling as cuda_handling

device = cuda_handling.set_cuda_to_gpu_nr()


class GradCAMWrapper(nn.Module):
    def __init__(self, original_model, demo_tensor):
        super().__init__()
        self.original_model = original_model
        
        self.demo_tensor = demo_tensor # demo_tensor_for_specific_sample

    def forward(self, x):
        
        return self.original_model(x, self.demo_tensor) #, grad_cam_mode=False)


def get_gradcam_img_w_sample(args, config, model, sample, demo_tensor = None, label_encoder = None):

    pil_img, pred = get_gradcam_img(args, config, model, sample.spec, sample.label, demo_tensor, label_encoder, sample.index) # input params: args, config, model, spec, label, demo_tensor = None, label_encoder

    sample.pil_gradcam_img_by_label = pil_img
    sample.pred = pred

    return sample


GRADCAM_SUITED_MODEL_TYPES = ['cnn', 'test_cnn', 'cnnlstm']


def get_gradcam_img(args, config, model, spec, label, demo_tensor, label_encoder = None, spec_index = None):
    # --- 1. INITIALIZE UNBOUND VARIABLES ---
    target_layer: List[nn.Module] = []
    gradcam_pil_img = None
    pred = None
    gradcam_model = None


    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL library not found. Install it with `pip install pillow`.")
    

    if model is None:
        print("Grad-CAM skipped: model is None")
        return None, None

 

    if args.model_type == 'cnn':
        target_layer = [model.conv_layers[8]]  # 3rd conv layer
    if args.model_type == 'cnnlstm':
        conv_layers = [layer for layer in model.conv_layers if isinstance(layer, nn.Conv2d)]
        target_layer = [conv_layers[-1]]  # last conv layer
    elif args.model_type in {'test_cnn', 'test_cnn2'}:
        target_layer = [model.conv_layers[3]]  # 3rd conv layer (only cnn layer)
    elif args.model_type == 'vggish':
        # Unfreeze the entire VGGish features
        for param in model.base_model.features.parameters():
            param.requires_grad = True
        target_layer = [model.base_model.features[13]]  # -3rd conv layer (cnn layer)
    elif args.model_type == 'pann':
        # Unfreeze the entire PANN features
        for param in model.base_model.model.conv_block6.parameters():
            param.requires_grad = True
        target_layer = [model.base_model.model.conv_block6]  # last conv layer (cnn layer)
   

    gradcam_logit = label # maybe the predicted class should be used instead of the true class
    target_class = [ClassifierOutputTarget(gradcam_logit)] # not necessary

    


    # 3 subsequent if-clauses without elif-clauses are used intentiona
    # lly
    # Convert to tensor and add channel dimension if needed
    if isinstance(spec, np.ndarray):
        spec_tensor = torch.from_numpy(spec).float()
    else:
        spec_tensor=spec
        print()

    #if args.model_type == 'pann':
        #print(f'pann spec in gradcam before processing: {spec_tensor.shape}')
        #spec_tensor = wav_to_spec_w_pann(spec_tensor, w_augment = False) 
        #spec_tensor = spec_tensor.unsqueeze(0)
        #print(f'pann spec in gradcam after processing: {spec_tensor.shape}')

    if len(spec_tensor.shape) == 2:  # If shape is (height, width)
        spec_tensor = spec_tensor.unsqueeze(0)  # Add channel dimension -> (1, height, width)
    
    if len(spec_tensor.shape) == 3:  # (C, H, W)
        spec_tensor = spec_tensor.unsqueeze(0)  # Add batch dimension
    spec_tensor = spec_tensor.to(device)

    # --------------------------------
    # Initialize and generate GradCAM
    # --------------------------------

    if args.add_demo_data:
        demo_tensor = demo_tensor.unsqueeze(0).to(device)
        assert demo_tensor is not None

        gradcam_model = GradCAMWrapper(model, demo_tensor)
        gradcam = GradCAM(model=gradcam_model, target_layers=target_layer)

        grayscale_cam = gradcam(input_tensor=spec_tensor, targets=cast(Any, target_class))

        with torch.no_grad():
            pred_tensor = gradcam_model(spec_tensor)
            _, pred = torch.max(pred_tensor.data, 1)

    else:
        gradcam = GradCAM(model=model, target_layers=target_layer)

        grayscale_cam = gradcam(input_tensor=spec_tensor, targets=cast(Any, target_class))

        with torch.no_grad():
            pred_tensor = model(spec_tensor)
            _, pred = torch.max(pred_tensor.data, 1)

    #spec = spec_tensor[0, 0].cpu().numpy()  # assuming (B, C, H, W)
    heatmap = grayscale_cam[0]
    # Flip the spectrogram and heatmap upside down
    spec = np.flipud(spec)
    heatmap = np.flipud(heatmap)


    # get labels:
    if label_encoder:
        # ensure CPU and convert to scalar
        label_cpu = label.cpu().item() if isinstance(label, torch.Tensor) else label
        pred_cpu  = pred.cpu().item()  if isinstance(pred, torch.Tensor)  else pred
        gradcam_logit_cpu = gradcam_logit.cpu().item() if isinstance(gradcam_logit, torch.Tensor) else gradcam_logit

        label = label_encoder.inverse_transform([label_cpu])[0]
        pred  = label_encoder.inverse_transform([pred_cpu])[0]
        gradcam_logit = label_encoder.inverse_transform([gradcam_logit_cpu])[0]
    # -------------------------
    # Plot spec above + heatmap below
    # -------------------------
    fig, ax = plt.subplots(2, 1, figsize=(6, 8))

    # Original spectrogram
    index_str = f'({spec_index})' if spec_index is not None else ''
    ax[0].imshow(spec, cmap='magma')#, origin='lower')
    ax[0].set_title(f'Original Spectrogram {index_str} - true: {label}; predicted: {pred}')
    ax[0].axis('off')

    # Grad-CAM heatmap with spectrogram contours
    ax[1].imshow(heatmap, cmap='jet')#, origin='lower')
    ax[1].contour(spec, colors='white', linewidths=0.5)#, origin='lower')
    ax[1].set_title(f'Grad-CAM with Spectrogram Contours - for {gradcam_logit} logit')
    ax[1].axis('off')

    # Reduce vertical spacing
    # Reduce vertical spacing explicitly
    fig.subplots_adjust(hspace=0.02, top=0.95, bottom=0.05)  # tweak as needed
    plt.tight_layout(pad=0.5)

    # Save to buffer and convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    gradcam_pil_img = Image.open(buf)

    if args.model_type in ['vggish', 'pann']:
        for param in model.base_model.features.parameters():
                param.requires_grad = False

    return gradcam_pil_img, pred

