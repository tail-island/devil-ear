import librosa
import numpy      as np
import os.path    as path
import pyaudio
import tensorflow as tf

from funcy                        import first, second, take
from keras.models                 import load_model
from useless_absolute_pitch_frame import UselessAbsolutePitchFrame
from utility                      import ZeroPadding, child_paths


def character_paths():
    for actor_path in filter(path.isdir, child_paths('./data/validate')):
        for character_path in filter(path.isdir, child_paths(actor_path)):
            yield character_path


def main():
    def stream_callback(data, frame_count, time_info, status):
        with graph.as_default():
            wave = np.frombuffer(data, dtype=np.float32)

            mfcc = librosa.feature.mfcc(wave, sr=44100, n_mfcc=44)

            x = ((mfcc - -14.631151332833856) / 92.12358373202312).reshape((1,) + mfcc.shape + (1,))
            y = model.predict(x)

            gui.draw_predict_result(wave, tuple(map(second, reversed(sorted(zip(y[0], range(len(y[0]))), key=first)))))

            return data, pyaudio.paContinue

    audio = pyaudio.PyAudio()

    model = load_model('./results/model.h5', custom_objects={'ZeroPadding': ZeroPadding})
    graph = tf.get_default_graph()

    stream = audio.open(format=pyaudio.paFloat32, channels=1, rate=44100, input=True, frames_per_buffer=22050, stream_callback=stream_callback)
    stream.start_stream()

    gui = UselessAbsolutePitchFrame(tuple(character_paths()))
    gui.mainloop()

    stream.stop_stream()
    stream.close()

    audio.terminate()


if __name__ == '__main__':
    main()
