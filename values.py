VEGAS_CLASSES_INDEXES = [
    ["c001",0,"Baby cry"],
    ["c002",1,"Dog"],
    ["c003",2,"Rail transport"],
    ["c004",3,"Fireworks"],
    ["c005",4,"Water flowing"]]

AUDIO_EMBEDDING_DIMENSION = 512
EMBEDDING_NOISE_RATIO = 0.
NOISE_VECTOR_DIM = int(AUDIO_EMBEDDING_DIMENSION*EMBEDDING_NOISE_RATIO) # generator noise vector dimension
NOISE_MU = 0 # mean for generating noise vector
NOISE_SIGMA = .035 # standard deviation for generating noise vector
EXPERIMENT 'Second' = # Options: 'First' / 'Second'

#=======================================================
# Attention params

#ACTIVE_ATTENTION = (1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0) # Replaced for variable NUM_ATTENTION_BLOCKS
NUM_ATTENTION_BLOCKS = (0,18,0,0,0,0,0) # 191_R3/276: (0,0,0,0,0,0,0) / 271_R: (1,1,1,1,1,1,0) / 273: (3,3,3,1,1,1,0) / 274: (3,3,3,3,3,1,0) / 279_R2: (3,3,3,3,3,3,0) / 280_R: (9,9,9,0,0,0,0) / 282: (9,9,9,9,9,0,0) / 283_R: (9,9,9,9,9,9,0) / 284: (6,6,6,6,6,0,0) / 285: (18,0,0,0,0,0,0) / 286_R: (0,18,0,0,0,0,0) / 287: (0,0,18,0,0,0,0) / 288: (3,0,0,0,0,0,0) / 289: (9,0,0,0,0,0,0)
ATTENTIONAL_NET = False
for i in NUM_ATTENTION_BLOCKS:
    if i > 0:
        ATTENTIONAL_NET = True
SHARE_ATTENTION_BLOCKS = False
DROPOUT_ATT = [0.0] * 25 # (0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0) # (0.0,0.0,0.0,0.0,0.12,0.0,0.0,0.0,0.1,0.0,0.0,0.0,0.8,0.0,0.0,0.0,0.6,0.0,0.0,0.0,0.4) 0.05 0.1
HEADS = (8,8,8,8,8,8,1) # (8,8,8,8,8,8,1) # (6,6,6,6,6,6,1) # (8,8,8,4,4,2,1) # (1,2,2,4,8,8,8)
D_MODEL = (320,320,320,320,320,320,4) # (320,320,320,320,320,320,4) # (240,240,240,240,240,240,4) # (192,192,192,192,192,192,4) # (256,256,176,112,80,56,4) # (4,16,32,128,256,512,1024) (8,32,64,128,512,1024) # D_MODEL if FEATURE_EMBEDDING_FROM_CONVOLUTION is true: 4,16,36,144,576,2304
AUDIO_ENCODER_LAYER = 13 # Audio encoder layer used as source embedding. Highest layer that produces feature maps is the 12
AUDIO_EMBEDDING_FROM_LINEAR_PROJECTION = False # Implemented in the audio autoencoder
FEATURE_EMBEDDING_FROM_CONVOLUTION = False
INTERPOLATION_MODE = 'bicubic' # 'nearest' 'bilinear' 'bicubic'
ADD_RESIDUAL = True
ADD_POS_ENCOD = True # teste 237 False
SHARE_AUDIO_EMB = True # teste 237 False
TRANSPOSE_EMB = (1,1) # Channel attention: (0,0) / Spatial attention: (1,1)
PATCHSIZE = (1,1,1,1,2,2,8) # (1,1,1,1,2,4,8) (1,1,1,1,1,1,1) # Patches are only applicable for target embeddings
APPLY_SRC_SELF_ATTENTION = False # If AUDIO_ENCODER_LAYER is set to 13 (last encoder layer) and if spatial attention is being used, then it doesn't make sense to apply source self attention, because the the source embedding sequence length is one.
APPLY_TGT_SELF_ATTENTION = True # teste 237, 246 False
APPLY_FF_AFTER_ATTENTION = True

#=======================================================

SPECS_DIR = 'spectrograms/' # add local path to spectrograms directory

SPECS_SEGMENTS_DIR = 'segments/' # add local path to spectrogram segments directory
FRAMES_DIR = 'frames/' # add local path to frames directory

SPECS_DIR_TRAIN = SPECS_DIR + 'train/'
SPECS_SEGMENTS_DIR_TRAIN = SPECS_SEGMENTS_DIR + 'train/'
SPECS_SEGMENTS_CSV_TRAIN = SPECS_SEGMENTS_DIR + 'spectrograms_segments_train.csv'
FRAMES_DIR_TRAIN = FRAMES_DIR + 'train/'
BATCH_FILE_NAME_TRAIN = 'data_batch_'

SPECS_DIR_VAL = SPECS_DIR + 'val/'
SPECS_SEGMENTS_DIR_VAL = SPECS_SEGMENTS_DIR + 'val/'
SPECS_SEGMENTS_CSV_VAL = SPECS_SEGMENTS_DIR + 'spectrograms_segments_val.csv'
FRAMES_DIR_VAL = FRAMES_DIR + 'val/'
BATCH_FILE_NAME_VAL = 'val_batch_'

SPECS_DIR_TEST = SPECS_DIR + 'test/'
SPECS_SEGMENTS_DIR_TEST = SPECS_SEGMENTS_DIR + 'test/'
SPECS_SEGMENTS_CSV_TEST = SPECS_SEGMENTS_DIR + 'spectrograms_segments_test.csv'
FRAMES_DIR_TEST = FRAMES_DIR + 'test/'
BATCH_FILE_NAME_TEST = 'test_batch_'


BATCHES_DIR = 'batches/' # add local path to batches directory
BATCHES_AUDIO_DIR = BATCHES_DIR + 'audio/spectrograms_128x100/'
BATCHES_VISUAL_DIR = BATCHES_DIR + 'visual/images_96x96_color/'
BATCHES_AUDIO_VISUAL_SPECTROGRAM_TO_IMAGE_DIR = BATCHES_DIR + 'audio_visual/spectrograms_to_images/'
if AUDIO_EMBEDDING_DIMENSION == 128:
    BATCHES_AUDIO_VISUAL_EMBEDDING_TO_IMAGE_DIR = BATCHES_DIR + 'audio_visual/embeddings_128_to_images_audioautoencoder_72_1-2/'
elif AUDIO_EMBEDDING_DIMENSION == 256:
    BATCHES_AUDIO_VISUAL_EMBEDDING_TO_IMAGE_DIR = BATCHES_DIR + 'audio_visual/embeddings_256_to_images_audioautoencoder_71_1-2/'
elif AUDIO_EMBEDDING_DIMENSION == 512:
    BATCHES_AUDIO_VISUAL_EMBEDDING_TO_IMAGE_DIR = BATCHES_DIR + 'audio_visual/embeddings_512_to_images_audioautoencoder_66_2/'
elif AUDIO_EMBEDDING_DIMENSION == 1024:
    BATCHES_AUDIO_VISUAL_EMBEDDING_TO_IMAGE_DIR = BATCHES_DIR + 'audio_visual/embeddings_1024_to_images_audioautoencoder_70_1-2/'
elif AUDIO_EMBEDDING_DIMENSION == 2048:
    BATCHES_AUDIO_VISUAL_EMBEDDING_TO_IMAGE_DIR = BATCHES_DIR + 'audio_visual/embeddings_2048_to_images_audioautoencoder_73_1-2/'

TRAINED_MODELS_AUTO_SAVE_DIR = 'trained_models_auto_save/' # add local path to trained models auto-save directory
TRAINED_MODELS_DIR = 'trained_models/' # add local path to trained models directory

AUDIO_NN_MODEL_FILE = TRAINED_MODELS_DIR + 'net_audio_model_(test_91_epoch_73).pth'
if AUDIO_EMBEDDING_DIMENSION == 128:
    AUDIO_AUTOENCODER_NN_MODEL_FILE = TRAINED_MODELS_DIR + 'net_audio_autoencoder_model_(test_72_1-2_epoch_1999).pth'
elif AUDIO_EMBEDDING_DIMENSION == 256:
    AUDIO_AUTOENCODER_NN_MODEL_FILE = TRAINED_MODELS_DIR + 'net_audio_autoencoder_model_(test_71_1-2_epoch_1999).pth'
elif AUDIO_EMBEDDING_DIMENSION == 512:
    AUDIO_AUTOENCODER_NN_MODEL_FILE = TRAINED_MODELS_DIR + 'net_audio_autoencoder_model_(test_66_2_epoch_995).pth'
elif AUDIO_EMBEDDING_DIMENSION == 1024:
    AUDIO_AUTOENCODER_NN_MODEL_FILE = TRAINED_MODELS_DIR + 'net_audio_autoencoder_model_(test_70_1-2_epoch_1999).pth'
elif AUDIO_EMBEDDING_DIMENSION == 2048:
    AUDIO_AUTOENCODER_NN_MODEL_FILE = TRAINED_MODELS_DIR + 'net_audio_autoencoder_model_(test_73_1-2_epoch_1995).pth'

VISUAL_NN_MODEL_FILE = TRAINED_MODELS_DIR + 'net_visual_model_(test_43_epoch_142).pth'
if EXPERIMENT = 'First':
    VISUAL_INTERPRET_NN_MODEL_FILE = [
        TRAINED_MODELS_DIR + 'net_visual_interpret_model_C1_(test_2_R2_epoch_165).pth',
        TRAINED_MODELS_DIR + 'net_visual_interpret_model_C2_(test_2_R2_epoch_146).pth',
        TRAINED_MODELS_DIR + 'net_visual_interpret_model_C3_(test_2_R_epoch_220).pth',
        TRAINED_MODELS_DIR + 'net_visual_interpret_model_C4_(test_2_R5_epoch_205).pth',
        TRAINED_MODELS_DIR + 'net_visual_interpret_model_C5_(test_2_R2_epoch_224).pth']
elif EXPERIMENT = 'Second':
    VISUAL_INTERPRET_NN_MODEL_FILE = [
        TRAINED_MODELS_DIR + 'net_visual_interpret_model_C1_(test_A2_epoch_203).pth',
        TRAINED_MODELS_DIR + 'net_visual_interpret_model_C2_(test_A2_epoch_307).pth',
        TRAINED_MODELS_DIR + 'net_visual_interpret_model_C3_(test_A2_epoch_240).pth',
        TRAINED_MODELS_DIR + 'net_visual_interpret_model_C4_(test_A2_epoch_124).pth',
        TRAINED_MODELS_DIR + 'net_visual_interpret_model_C5_(test_A2_epoch_125).pth']

VISUAL_GENERATOR_NN_MODEL_FILE = TRAINED_MODELS_DIR + 'net_visual_generator_model_(test_191_R3_1-4_TS_epoch_3679).pth'
DISCRIMINATOR_NN_MODEL_FILE = TRAINED_MODELS_DIR + 'net_discriminator_model_(test_191_R3_1-4_TS_epoch_3679).pth'

#=======================================================

NUM_IMAGES_LIMIT_A = 1800
NUM_IMAGES_LIMIT_AV_EMBEDDING_TO_IMAGE = 700 if (AUDIO_EMBEDDING_DIMENSION < 2048) else 500
MAX_FILES_PER_CLASS = -1 # -1 to get min-max number of all classes
