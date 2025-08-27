import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# 스타일 및 폰트 설정 (Times New Roman)
sns.set(style='whitegrid', context='talk')
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 25,
    "axes.labelsize": 25,
    "xtick.labelsize": 25,
    "ytick.labelsize": 25,
})

# 방향별 오프셋, ha, va 매핑
direction_cfg = {
    'N':  {'offset': (  0,  10), 'ha':'center','va':'bottom'},
    'NE': {'offset': (  30,  2), 'ha':'center','va':'bottom'},
    'E':  {'offset': ( 10,   0), 'ha':'left',  'va':'center'},
    'SE': {'offset': ( -10, -10), 'ha':'left','va':'top'},
    'S':  {'offset': (  0, -10), 'ha':'center','va':'top'},
    'SW': {'offset': ( 15, -10), 'ha':'right','va':'top'},
    'W':  {'offset': (-10,   0), 'ha':'right', 'va':'center'},
    'NW': {'offset': (10,  10), 'ha':'right','va':'bottom'},
}


# 데이터 (마지막 원소로 방향 지정)
data = {
    'STD-Net':       [-1,    27.96, 31.12, 0.903, 'NE'],
    # 'PMTNet':        [4.8,    0.31, 30.839,0.9006,'S'],
    'DnCNN':         [72.2,   0.60, 24.54, 0.834, 'W'], # from MBCNN paper
    'DMCNN':         [20.8,   0.79, 26.77, 0.871, 'W'],
    # 'MopNet':        [293.4, 58.57, 27.75, 0.895, 'NW'],
    'MopNet':        [396.1, 12.4, 27.75, 0.895, 'SE'], # from MBCNN paper
    'FHDe$^{2}$Net': [353.4, 13.59, 27.785,0.8960,'NW'],
    'U-Net':         [32.8,   8.6,  26.49, 0.864, 'S'], # from MBCNN paper
    'VDSR':          [87.5,   0.7,  24.68, 0.837, 'E'], # from MBCNN paper
    'EDSR':          [160.3, 12.2,  26.82, 0.853, 'S'], # from MBCNN paper
    'HRDN':          [15.05,  7.79, 28.47, 0.860, 'S'],
    'WDNet':         [42.9,    5.7,    28.08, 0.904, 'S'], # from MBCNN paper
    'MBCNN-conf':    [125.3, 13.50, 30.03, 0.8930,'S'], # from MBCNN paper
    'MBCNN':         [148.6, 14.90, 30.41, 0.9000,'NE'], # from MBCNN paper
    'ESDNet':   [17.6*2,   5.934, 29.81, 0.916, 'SW'],
    'ESDNet-L': [-1,  10.623, 30.11, 0.920,'NE'],
    'DMSFN':   [21.9*2,   5.3,   29.9,  0.916, 'SE'],
    'DMSFN-L': [87.6*2,  21.3,   30.3,  0.923, 'SE'],
    'Uformer-B':     [-1,    50.88,    29.28, 0.917, 'N'],
    'DDFNet':        [-1,   15.4,  30.939,0.914,'SE'],
    # 'CBDN':          [139.3,13.1,  31.02, 0.922,'NE'],
    'TransRes':      [186.2,13.4,  27.55, 0.890,'W'],
    'Original Image':[-1,   -1,    20.30, 0.738,'SW'],
    'AMSDM':         [14.97,2.39,  30.27, 0.897,'S'],
    'P-BiC':         [-1, 4.922, 30.56, 0.9250, 'N'],
    'SEDT': {
        'SEDT-T': [14.8834877440*2, 5.872, 30.1332, 0.9190, 'N'],
        'SEDT-S': [49.6311992320*2, 21.275, 30.5782, 0.9244, 'N'],
        'SEDT-B': [97.131692032*2, 51.066, 30.9696, 0.9287, 'N'],
    },
}

# 색상 사이클 (원형 마커용)
color_cycle = itertools.cycle(plt.cm.tab20.colors)

plt.figure(figsize=(10, 7))

# 1) SEDT variants만 빨간 별표 + 빨간 실선 연결
sedt_items = data['SEDT']
sedt_sorted = sorted(sedt_items.items(), key=lambda x: x[1][0])
sedt_flops = [v[0] for _, v in sedt_sorted]
sedt_psnrs = [v[2] for _, v in sedt_sorted]
plt.plot(sedt_flops, sedt_psnrs, '-', color='red', linewidth=3, label='SEDT family')
for name, vals in sedt_sorted:
    flop, _, psnr, _, direction = vals
    plt.scatter(flop, psnr, marker='*', s=450, color='red', edgecolor='k', zorder=3)
    cfg = direction_cfg[direction]
    plt.annotate(
        name,
        xy=(flop, psnr),
        xytext=cfg['offset'],
        textcoords='offset points',
        ha=cfg['ha'], va=cfg['va'],
        fontsize=25
    )

# 2) 나머지 모델 및 모델군
for model_name, model_info in data.items():
    if model_name == 'SEDT':
        continue

    if isinstance(model_info, dict):
        # 모델군: 통일 색상 + 실선 연결
        fam_color = next(color_cycle)
        sorted_models = sorted(model_info.items(), key=lambda x: x[1][0])
        flops = [v[0] for _, v in sorted_models]
        psnrs = [v[2] for _, v in sorted_models]
        plt.plot(flops, psnrs, '-', color=fam_color, linewidth=2, alpha=0.6, label=model_name)
        for name, vals in sorted_models:
            flop, _, psnr, _, direction = vals
            plt.scatter(flop, psnr, marker='o', s=200, color=fam_color, edgecolor='k', zorder=3)
            cfg = direction_cfg[direction]
            plt.annotate(
                name,
                xy=(flop, psnr),
                xytext=cfg['offset'],
                textcoords='offset points',
                ha=cfg['ha'], va=cfg['va'],
                fontsize=22
            )
    else:
        # 단독 모델
        flop, _, psnr, _, direction = model_info
        plt.scatter(
            flop, psnr,
            marker='o',
            s=200,
            color=next(color_cycle),
            edgecolor='k',
            zorder=3,
            label=model_name
        )
        cfg = direction_cfg[direction]
        plt.annotate(
            model_name,
            xy=(flop, psnr),
            xytext=cfg['offset'],
            textcoords='offset points',
            ha=cfg['ha'], va=cfg['va'],
            fontsize=15
        )

plt.xscale('log')
plt.xlim(10, 1000)    # 예: GFLOPs 를 1~400 사이로
plt.ylim(24, 32)    # 예: PSNR 을 24~32 dB 사이로
plt.xlabel(r'GFLOPs', fontsize=25)
plt.ylabel(r'PSNR [dB]', fontsize=25)
# plt.legend(loc='lower left', fontsize=12)

plt.tight_layout()
plt.savefig(
    'image1.eps',
    format='eps',
    dpi=600,
    bbox_inches='tight'
)
plt.show()
