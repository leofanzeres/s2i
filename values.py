VEGAS_CLASSES_INDEXES = [
    ["c001",0,"Baby cry"],
    ["c002",1,"Dog"],
    ["c003",2,"Rail transport"],
    ["c004",3,"Fireworks"],
    ["c005",4,"Water flowing"]]

AUDIO_EMBEDDING_DIMENSION = 512

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
VISUAL_INTERPRET_NN_MODEL_FILE = [
    TRAINED_MODELS_DIR + 'net_visual_interpret_model_C1_(test_2_R2_epoch_165).pth',
    TRAINED_MODELS_DIR + 'net_visual_interpret_model_C2_(test_2_R2_epoch_146).pth',
    TRAINED_MODELS_DIR + 'net_visual_interpret_model_C3_(test_2_R_epoch_220).pth',
    TRAINED_MODELS_DIR + 'net_visual_interpret_model_C4_(test_2_R5_epoch_205).pth',
    TRAINED_MODELS_DIR + 'net_visual_interpret_model_C5_(test_2_R2_epoch_224).pth']

VISUAL_GENERATOR_NN_MODEL_FILE = TRAINED_MODELS_DIR + 'net_visual_generator_model_(test_191_R3_1-4_TS_epoch_3679).pth'
DISCRIMINATOR_NN_MODEL_FILE = TRAINED_MODELS_DIR + 'net_discriminator_model_(test_191_R3_1-4_TS_epoch_3679).pth'

#=======================================================

NUM_IMAGES_LIMIT_A = 1800
NUM_IMAGES_LIMIT_AV_EMBEDDING_TO_IMAGE = 700 if (AUDIO_EMBEDDING_DIMENSION < 2048) else 500
MAX_FILES_PER_CLASS = -1 # -1 to get min-max number of all classes
