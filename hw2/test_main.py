import sys
import json
import numpy as np
import model
import util
import input
import eval
import time

VOCAB_SIZE = 3000
FRAME_STEP = 20
FRAME_DIM = 4096
BATCH_SIZE = 100
CAPTION_STEP = 45
EPOCH = 1000
SCHEDULED_SAMPLING_CONVERGE = 5000
MODEL_FILE_NAME = 'result_schedule'


def test(model, test_data, dict_rev):
    answers = []
    score = 0
    for x, video_ids in test_data:
        result = model.predict(x)
        sentences = [' '.join([dict_rev[str(word)]
                               for word in trim(sen.tolist())]) for sen in result[0]]
        answers.extend(list(zip(video_ids, sentences)))
    json.dump([{'caption': cap, 'id:': vid} for vid, cap in answers],
              open('output.json', 'w'))

def main():

    d_idx2word = json.load(open('data/dict_rev.json', 'r'))


    feature_path = sys.argv[2]
    id_file = sys.argv[1]
    print(feature_path, id_file)
    test_data_loader = input.TestPrivateDataLoader(id_path=id_file,
                                        data_path=feature_path,
                                        frame_step=FRAME_STEP,
                                        frame_dim=FRAME_DIM,
                                        caption_step=CAPTION_STEP,
                                        vocab_size=VOCAB_SIZE,
                                        shuffle=False
                                        )
    S2VT = model.Effective_attention_model(caption_steps=CAPTION_STEP)
    S2VT.loadModel('result_schedule_para/model_30000.ckpt')
    test_batch = test_data_loader.get_data(BATCH_SIZE)
    test(S2VT, test_batch, d_idx2word)


if __name__ == '__main__':
    main()
