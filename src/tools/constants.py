def gen2int(gen: str) -> int:
    """Maps generator names to integers using intermediate mapping to a family of generators

    Args:
        gen (str): values in the lists of constants.GEN_TO_GEN

    Returns:
        int: integer that can be used as a key with constants.INT_TO_GEN
    """
    for generator in GEN_TO_GEN:
        if gen in GEN_TO_GEN[generator]:
            return GEN_TO_INT[generator]
    raise Exception(f"{gen} is not in GEN_TO_GEN map (see tools/constants.py)")


SEED = 7
TMP_DIR = "./_dump/"
CLIP_FEATURE_DIM = 768 # imension of the feature space of CLIP
DINO_FEATURE_DIM = 768
REAL_LABEL = 1
FAKE_LABEL = 0
PATH_TO_DATA = "/data3/AID/"
REAL_IMG_GEN = "null"
# int -> generator name for model trained on the 18 classes from data3/AID
INT_TO_GEN_DATA3 = {0: REAL_IMG_GEN,
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
# generator name -> int for model trained on the 18 classes from data3/AID
GEN_TO_INT_DATA3 = {REAL_IMG_GEN: 0,
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

GEN_TO_INT_OOD = {REAL_IMG_GEN: 0,
                  'Lexica': 1,
                  'Ideogram': 2,
                  'Leonardo': 3,
                  'Copilot': 4,
                  'img2img_SD1.5': 5,
                  'Photoshop_generativemagnification': 6,
                  'Photoshop_generativefill': 7}

INT_TO_GEN_OOD = {0: REAL_IMG_GEN,
                  1: 'Lexica',
                  2: 'Ideogram',
                  3: 'Leonardo',
                  4: 'Copilot',
                  5: 'img2img_SD1.5',
                  6: 'Photoshop_generativemagnification',
                  7: 'Photoshop_generativefill'}

INT_TO_GEN_DATA3_TEST = {0: REAL_IMG_GEN,
                         1: 'Source_30_styleGAN2_512',
                         2: 'Source_10_Kandinsky-2.2-vRC_512',
                         3: 'Source_23_ShiftedDiffusion_512',
                         4: 'Source_27_stable-2-1-vRC_512',
                         5: 'Source_29_stable-xl-vRC_512',
                         6: 'Source_31_styleGAN3_512',
                         7: 'Source_5_dreamlike_512',
                         8: 'Source_2_animagineXL3-1-vRC_512',
                         9: 'Source_14_megaDallEUP_512',
                         10: 'Source_20_playground-v2-5_512',
                         11: 'Source_22_playground-v2-vRC_512',
                         12: 'Source_15_miniDallEUP_512',
                         13: 'Source_25_stable-1-5-vRC_512',
                         14: 'Source_19_playground-v2_512',
                         15: 'Source_13_LDM_512',
                         16: 'Source_26_stable-2.1_512',
                         17: 'Source_12_LafitteUP_512',
                         18: 'Source_3_DF-XL_512',
                         19: 'Source_1_animagineXL3-1_512',
                         20: 'Source_32_LCM_Dreamshaper-v7-vRC_512',
                         21: 'Source_17_pixartAlpha-vRC_512',
                         22: 'Source_28_stable-xl_512',
                         23: 'Source_4_DF-XL-vRC_512',
                         24: 'Source_6_dreamlike-vRC_512',
                         25: 'Source_9_Kandinsky-2.1-vRC_512',
                         26: 'Source_16_pixart_512',
                         27: 'Source_7_gigaGAN_512',
                         28: 'Source_11_kandinsky_512',
                         29: 'Source_8_glideUP_512',
                         30: 'Source_24_stable-1.5_512',
                         31: 'Source_18_pixartSigma-vRC_512',
                         32: 'Source_21_playground-v2-5-vRC_512'}


GEN_TO_INT_DATA3_TEST = {"Source_00_RealPhoto": 0,
                         'Source_30_styleGAN2_512': 1,
                         'Source_10_Kandinsky-2.2-vRC_512': 2,
                         'Source_23_ShiftedDiffusion_512': 3,
                         'Source_27_stable-2-1-vRC_512': 4,
                         'Source_29_stable-xl-vRC_512': 5,
                         'Source_31_styleGAN3_512': 6,
                         'Source_5_dreamlike_512': 7,
                         'Source_2_animagineXL3-1-vRC_512': 8,
                         'Source_14_megaDallEUP_512': 9,
                         'Source_20_playground-v2-5_512': 10,
                         'Source_22_playground-v2-vRC_512': 11,
                         'Source_15_miniDallEUP_512': 12,
                         'Source_25_stable-1-5-vRC_512': 13,
                         'Source_19_playground-v2_512': 14,
                         'Source_13_LDM_512': 15,
                         'Source_26_stable-2.1_512': 16,
                         'Source_12_LafitteUP_512': 17,
                         'Source_3_DF-XL_512': 18,
                         'Source_1_animagineXL3-1_512': 19,
                         'Source_32_LCM_Dreamshaper-v7-vRC_512': 20,
                         'Source_17_pixartAlpha-vRC_512': 21,
                         'Source_28_stable-xl_512': 22,
                         'Source_4_DF-XL-vRC_512': 23,
                         'Source_6_dreamlike-vRC_512': 24,
                         'Source_9_Kandinsky-2.1-vRC_512': 25,
                         'Source_16_pixart_512': 26,
                         'Source_7_gigaGAN_512': 27,
                         'Source_11_kandinsky_512': 28,
                         'Source_8_glideUP_512': 29,
                         'Source_24_stable-1.5_512': 30,
                         'Source_18_pixartSigma-vRC_512': 31,
                         'Source_21_playground-v2-5-vRC_512': 32}

GEN_TO_GEN = {REAL_IMG_GEN: [REAL_IMG_GEN, "Source_00_RealPhoto"],
              "Stable diffusion" : ["stable-1-5-vRC",
                                    "stable-2-1-vRC",
                                    "stable-xl-vRC",
                                    "Source_24_stable-1.5_512",
                                    "Source_25_stable-1-5-vRC_512",
                                    "Source_26_stable-2.1_512",
                                    "Source_27_stable-2-1-vRC_512",
                                    "Source_28_stable-xl_512",
                                    "Source_29_stable-xl-vRC_512",
                                    "Source_25_SD1_5",
                                    "Source_27_SD2_1",
                                    "Source_29_SDXL",
                                    "LongCaptions"],
            
            "Kandisky":["Kandinsky-2.1-vRC",
                        "Kandinsky-2.2-vRC",
                        "Source_10_Kandinsky-2.2-vRC_512",
                        "Source_11_kandinsky_512",
                        "Source_9_Kandinsky-2.1-vRC_512",
                        "Source_9_10_kandinsky"],
            
            "DF_XL": ["DF-XL-vRC",
                      "Source_3_DF-XL_512",
                      "Source_4_DF-XL-vRC_512",
                      "Source_4_DF-XL"],
            
            "dreamlike": ["dreamlike-vRC",
                          "Source_5_dreamlike_512",
                          "Source_6_dreamlike-vRC_512",
                          "Source_6_dreamlike"],


            "gigaGan": ["gigaGAN","Source_7_gigaGAN_512","Source_7_gigaGan"],

            "GlideUP": ["GlideUP", "Source_8_glideUP_512"],

            "LafitteUP": ["LafitteUP", "Source_12_LafitteUP_512"],

            "LCM_Dreamshaper": ["LCM_Dreamshaper-v7-vRC", 
                                "Source_32_LCM_Dreamshaper-v7-vRC_512",
                                "Source_32_Dreamshaper"],

            "megaDallEUP": ["megaDallEUP", "Source_14_megaDallEUP_512","Copilot_images"],

            "miniDallEUP": ["miniDallEUP", "Source_15_miniDallEUP_512"],

            "pixart": ["pixartSigma-vRC",
                       "Source_16_pixart_512",
                       "Source_17_pixartAlpha-vRC_512",
                       "Source_18_pixartSigma-vRC_512",
                       "Source_18_pixartSigma",
                       "Source_17_pixartAlpha"],

            "playground": ["playground-v2-5-vRC",
                           "playground-v2-vRC",
                           "Source_19_playground-v2_512",
                           "Source_20_playground-v2-5_512",
                           "Source_21_playground-v2-5-vRC_512",
                           "Source_22_playground-v2-vRC_512",
                           "Source_21_playground"],
            
            "styleGan2": ["styleGan2","Source_30_styleGAN2_512"],
            
            "styleGan3": ["styleGan3", 
                          "Source_31_styleGAN3_512",
                          "Source_30_31_stylegan2_3",
                          "Ideogram_images",
                          "Leonardo_images",
                          "Lexica_images",
                          "MidJourneyV6"], # put generators that I didn't know where to put

            "animagineXL": ["animagineXL3-1-vRC",
                            "Source_1_animagineXL3-1_512",
                            "Source_2_animagineXL3-1-vRC_512",
                            "Source_2_animagineXL3"]
            }

GEN_TO_INT = {REAL_IMG_GEN: 0,
              'Stable diffusion': 1,
              'Kandisky': 2,
              'DF_XL': 3,
              'dreamlike': 4,
              'gigaGan': 5,
              'GlideUP': 6,
              'LafitteUP': 7,
              'LCM_Dreamshaper': 8,
              'megaDallEUP': 9,
              'miniDallEUP': 10,
              'pixart': 11,
              'playground': 12,
              'styleGan2': 13,
              'styleGan3': 14,
              'animagineXL': 15}

INT_TO_GEN = {0: REAL_IMG_GEN,
              1: 'Stable diffusion',
              2: 'Kandisky',
              3: 'DF_XL',
              4: 'dreamlike',
              5: 'gigaGan',
              6: 'GlideUP',
              7: 'LafitteUP',
              8: 'LCM_Dreamshaper',
              9: 'megaDallEUP',
              10: 'miniDallEUP',
              11: 'pixart',
              12: 'playground',
              13: 'styleGan2',
              14: 'styleGan3',
              15: 'animagineXL'}

INT_TO_LABEL = {FAKE_LABEL: "fake", REAL_LABEL: "real"} 

LABEL_TO_INT = {"fake":FAKE_LABEL,"real":REAL_LABEL} 

PATH_TO_DATA4 = "/data4/saland/data/"

BLACKLIST = ["Source_13_LDM_512","Source_23_ShiftedDiffusion_512"]

# the images from this generators contain content that is repetitive and very similar (semantic bias)
AID_BLACKLIST = [GEN_TO_INT['GlideUP'],
                 GEN_TO_INT['LafitteUP'],
                 GEN_TO_INT['megaDallEUP'],
                 GEN_TO_INT['miniDallEUP']]


# the images from this generators contain content that is repetitive and very similar (semantic bias)
AID_TEST_BLACKLIST = [gen2int("Source_1_animagineXL3-1_512"),
                      gen2int("Source_3_DF-XL_512"),
                      gen2int("Source_5_dreamlike_512"),
                      gen2int('Source_8_glideUP_512'),
                      gen2int('Source_11_kandinsky_512'),
                      gen2int('Source_12_LafitteUP_512'),
                      gen2int('Source_14_megaDallEUP_512'),
                      gen2int('Source_15_miniDallEUP_512'),
                      gen2int('Source_16_pixart_512'),
                      gen2int("Source_19_playground-v2_512"),
                      gen2int('Source_20_playground-v2-5_512'),
                      gen2int('Source_24_stable-1.5_512'),
                      gen2int('Source_26_stable-2.1_512'),
                      gen2int('Source_28_stable-xl_512'),
                      gen2int('Source_30_styleGAN2_512'),
                      gen2int('Source_31_styleGAN3_512')]

OOD_BLACKLIST_OLD = [GEN_TO_INT_OOD['img2img_SD1.5'],
                     GEN_TO_INT_OOD['Photoshop_generativefill'],
                     GEN_TO_INT_OOD['Photoshop_generativemagnification']]