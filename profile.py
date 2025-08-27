import os
import logging
import torch
from thop import profile, clever_format
from models import create_model
from config.config import get_parser

def measure_flops_and_params(model_cfg, device=None):
    # set device
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # create model
    model = create_model(model_cfg).to(device)
    model.eval()

    # specify input size
    input_size = (1, 3, model_cfg.MODEL_KWARGS['IMG_SIZE'], model_cfg.MODEL_KWARGS['IMG_SIZE'])

    # measure
    input_tensor = torch.randn(input_size, device=device)
    input_dict = {'in_img': input_tensor}
    macs, params = profile(model, inputs=(input_dict,),verbose=True)
    macs, params = clever_format([macs, params], "%.3f")

    # print
    logging.warning("MACs:" + macs + ", Params:" + params)
    logging.warning("From model.flops(): " + str(model.flops()))
    

if __name__ == "__main__":
    args = get_parser()
    measure_flops_and_params(args)
