SEED = 7
TMP_DIR = "./_dump/"
CLIP_FEATURE_DIM = 768 # imension of the feature space of CLIP
DINO_FEATURE_DIM = 768
REAL_LABEL = 1
FAKE_LABEL = 0
PATH_TO_DATA = "/data3/AID/"
REAL_IMG_GEN = "null"
# int -> generator name for model trained on the 18 classes from data3
INT_TO_GEN_DATA3 = {0: 'null',
                    1: 'miniDallEUP',
                    2: 'GlideUP',
                    3: 'LCM_Dreamshaper-v7-vRC',
                    4: 'Kandinsky-2.2-vRC',
                    5: 'DF-XL-vRC',
                    6: 'pixartSigma-vRC',
                    7: 'animagineXL3-1-vRC',
                    8: 'megaDallEUP',
                    9: 'stable-xl-vRC',
                    10: 'stable-2-1-vRC',
                    11: 'dreamlike-vRC',
                    12: 'playground-v2-vRC',
                    13: 'gigaGAN',
                    14: 'LafitteUP',
                    15: 'playground-v2-5-vRC',
                    16: 'stable-1-5-vRC',
                    17: 'Kandinsky-2.1-vRC'}
# generator name -> int for model trained on the 18 classes from data3
GEN_TO_INT_DATA3 = {'null': 0,
                    'miniDallEUP': 1,
                    'GlideUP': 2,
                    'LCM_Dreamshaper-v7-vRC': 3,
                    'Kandinsky-2.2-vRC': 4,
                    'DF-XL-vRC': 5,
                    'pixartSigma-vRC': 6,
                    'animagineXL3-1-vRC': 7,
                    'megaDallEUP': 8,
                    'stable-xl-vRC': 9,
                    'stable-2-1-vRC': 10,
                    'dreamlike-vRC': 11,
                    'playground-v2-vRC': 12,
                    'gigaGAN': 13,
                    'LafitteUP': 14,
                    'playground-v2-5-vRC': 15,
                    'stable-1-5-vRC': 16,
                    'Kandinsky-2.1-vRC': 17}

GEN_TO_INT_OOD = {'null': 0,
                  'Lexica': 1,
                  'Ideogram': 2,
                  'Leonardo': 3,
                  'Copilot': 4,
                  'img2img_SD1.5': 5,
                  'Photoshop_generativemagnification': 6,
                  'Photoshop_generativefill': 7}

INT_TO_GEN_OOD = {0: 'null',
                  1: 'Lexica',
                  2: 'Ideogram',
                  3: 'Leonardo',
                  4: 'Copilot',
                  5: 'img2img_SD1.5',
                  6: 'Photoshop_generativemagnification',
                  7: 'Photoshop_generativefill'}