_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py',
    '../_base_/datasets/loveda.py'
]

checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_t_20230227-119e8c9f.pth'

ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

crop_size = (512, 512)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],  
    std=[58.395, 57.12, 57.375],     
    bgr_to_rgb=True,                  
    pad_val=0,                        
    seg_pad_val=255,                  
    size=(512, 512),                  
    test_cfg=dict(size_divisor=32)    
)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='MSCAN',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=[32, 64, 160, 256],           
        mlp_ratios=[8, 8, 4, 4],                
        drop_rate=0.0,                           
        drop_path_rate=0.1,                      
        depths=[3, 3, 5, 2],                     
        attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],  
        attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]], 
        act_cfg=dict(type='GELU'),               
        norm_cfg=dict(type='BN', requires_grad=True)  
    ),
    neck=dict(
        type='CBAMNeck',  
        in_channels=[32, 64, 160, 256],  
        out_channels=[32, 64, 160, 256],  
        reduction=16  
    ),  
    decode_head=dict(
        type='EnhancedSegformerHead',               
        in_channels=[32, 64, 160, 256],           
        in_index=[0, 1, 2, 3],                      
        channels=256,                               
        dropout_ratio=0.1,                          
        num_classes=7,                              
        norm_cfg=dict(type='SyncBN', requires_grad=True),  
        align_corners=False,                        
        ppm_out_channels=256,
        pool_scales=(1, 2, 3, 6)
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

train_dataloader = dict(batch_size=4)

optim_wrapper = dict(
    _delete_=True,  
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.00006,
        betas=(0.9, 0.999),
        weight_decay=0.01
    ),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }
    )
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=1500
    ),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=80000,
        eta_min=0.0,
        by_epoch=False,
    )
]
