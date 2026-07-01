from .config import _C as config


def update_mamba_config(model_size):
    config_dict = dict(
        patch_size=config.PATCH_SIZE,
        in_chans=config.IN_CHANS,
        num_classes=config.NUM_CLASSES,
        depths=config.DEPTHS,
        dims=config.EMBED_DIM,
        ssm_d_state=config.SSM_D_STATE,
        ssm_ratio=config.SSM_RATIO,
        ssm_rank_ratio=config.SSM_RANK_RATIO,
        ssm_dt_rank=("auto" if config.SSM_DT_RANK == "auto" else int(config.SSM_DT_RANK)),
        ssm_act_layer=config.SSM_ACT_LAYER,
        ssm_conv=config.SSM_CONV,
        ssm_conv_bias=config.SSM_CONV_BIAS,
        ssm_drop_rate=config.SSM_DROP_RATE,
        ssm_init=config.SSM_INIT,
        forward_type=config.SSM_FORWARDTYPE,
        mlp_ratio=config.MLP_RATIO,
        mlp_act_layer=config.MLP_ACT_LAYER,
        mlp_drop_rate=config.MLP_DROP_RATE,
        drop_path_rate=config.DROP_PATH_RATE,
        patch_norm=config.PATCH_NORM,
        norm_layer=config.NORM_LAYER,
        downsample_version=config.DOWNSAMPLE,
        patchembed_version=config.PATCHEMBED,
        gmlp=config.GMLP,
        use_checkpoint=config.USE_CHECKPOINT,
    )

    if model_size == "base":
        config_dict.update(
            dict(
                patch_size=4,
                depths=[2, 2, 15, 2],
                dims=[128, 128 * 2, 128 * 4, 128 * 8],
                ssm_d_state=1,
                ssm_conv_bias=False,
                mlp_ratio=4.0,
                downsample_version="v3",
                patchembed_version="v2",
                forward_type="v04",
            )
        )
    return config_dict
