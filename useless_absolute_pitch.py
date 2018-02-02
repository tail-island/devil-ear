import librosa
import numpy      as np
import pyaudio
import tensorflow as tf

from funcy        import first, second, take
from keras.models import load_model
from time         import sleep
from utility      import ZeroPadding


labels = ['M・A・O:若狭悠里',
          '井口裕香:三宅日向',
          '佐藤聡美:田井中律',
          '原紗友里:大垣千明',
          '子安武人:ロズワール・L・メイザース',
          '寿美菜子:琴吹紬',
          '小林裕介:ナツキ・スバル',
          '小澤亜李:恵飛須沢胡',
          '山寺宏一:スパイク・スピーゲル',
          '新井里美:ベアトリス',
          '日笠陽子:秋山澪',
          '日野由利加:カテリーナ',
          '村川梨衣:ラム',
          '東山奈央:志摩リン',
          '水瀬いのり:レム',
          '水瀬いのり:丈槍由紀',
          '水瀬いのり:玉木マリ',
          '石塚運昇:ジェット・ブラック',
          '福島潤:佐藤和真',
          '竹達彩奈:中野梓',
          '花守ゆみり:各務原なでしこ',
          '花澤香菜:小淵沢報瀬',
          '茅野愛衣:佐倉慈',
          '西凛太朗:アシモフ・ソーレンサン',
          '豊崎愛生:ゆんゆん',
          '豊崎愛生:平沢唯',
          '豊崎愛生:犬山あおい',
          '雨宮天:アクア',
          '高橋李衣:エミリア',
          '高橋李衣:めぐみん',
          '高橋李衣:斉藤恵那',
          '高橋李衣:直樹美紀']


def main():
    def stream_callback(data, frame_count, time_info, status):
        with graph.as_default():
            mfcc = librosa.feature.mfcc(np.frombuffer(data, dtype=np.float32), sr=44100, n_mfcc=44)

            x = ((mfcc - -14.631151332833856) / 92.12358373202312).reshape((1,) + mfcc.shape + (1,))
            y = model.predict(x)

            for p, i in take(1, (reversed(sorted(zip(y[0], range(len(y[0]))), key=first)))):
                print('{0:.3}\t{1}'.format(p, labels[i]))
            print()

            return data, pyaudio.paContinue

    model = load_model('./results/model.h5', custom_objects={'ZeroPadding': ZeroPadding})
    graph = tf.get_default_graph()

    audio = pyaudio.PyAudio()

    stream = audio.open(format=pyaudio.paFloat32, channels=1, rate=44100, input=True, frames_per_buffer=22050, stream_callback=stream_callback)
    stream.start_stream()

    while stream.is_active():
        sleep(1)

    stream.stop_stream()
    stream.close()

    audio.terminate()


if __name__ == '__main__':
    main()
