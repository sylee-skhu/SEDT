from .vggperceptual import MultiScaleVGGPerceptualLoss

def create_loss(args):
    loss_name = args.LOSS_NAME.lower()
    if loss_name == 'multiscalevggperceptualloss':
        loss = MultiScaleVGGPerceptualLoss(
            num_scales=args.NUM_SCALES,
            lam=args.LAM,
            lam_p=args.LAM_P
        )
    else:
        raise NotImplementedError(f"Unknown loss: {loss_name}")
    return loss
