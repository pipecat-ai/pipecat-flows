#
# This demo will join a Daily meeting and send the audio from a WAV file into
# the meeting.
#
# Usage: python3 wav_audio_send.py -m MEETING_URL -i FILE.wav
#

import argparse
import threading
import wave

from daily import *

SAMPLE_RATE = 16000
NUM_CHANNELS = 1


class SendWavApp:
    def __init__(self, input_file_name, sample_rate, num_channels):
        self.__mic_device = Daily.create_microphone_device(
            "my-mic", sample_rate=sample_rate, channels=num_channels
        )

        self.__client = CallClient()

        self.__client.update_subscription_profiles(
            {"base": {"camera": "unsubscribed", "microphone": "unsubscribed"}}
        )

        self.__app_quit = False
        self.__app_error = None

        self.__start_event = threading.Event()
        self.__thread = threading.Thread(target=self.send_wav_file, args=[input_file_name])
        self.__thread.start()

    def on_joined(self, data, error):
        if error:
            print(f"Unable to join meeting: {error}")
            self.__app_error = error
        self.__start_event.set()

    def run(self, meeting_url, meeting_token):
        self.__client.join(
            meeting_url,
            meeting_token,
            client_settings={
                "inputs": {
                    "camera": False,
                    "microphone": {"isEnabled": True, "settings": {"deviceId": "my-mic"}},
                }
            },
            completion=self.on_joined,
        )
        self.__thread.join()

    def leave(self):
        self.__app_quit = True
        self.__thread.join()
        self.__client.leave()
        self.__client.release()

    def send_wav_file(self, file_name):
        self.__start_event.wait()

        if self.__app_error:
            print(f"Unable to send WAV file!")
            return

        wav = wave.open(file_name, "rb")

        sent_frames = 0
        total_frames = wav.getnframes()
        sample_rate = wav.getframerate()
        while not self.__app_quit and sent_frames < total_frames:
            # Read 100ms worth of audio frames.
            frames = wav.readframes(int(sample_rate / 10))
            if len(frames) > 0:
                self.__mic_device.write_frames(frames)
                sent_frames += sample_rate / 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--meeting", required=True, help="Meeting URL")
    parser.add_argument("-t", "--token", required=True, help="Meeting token")
    parser.add_argument("-i", "--input", required=True, help="WAV input file")
    parser.add_argument(
        "-c", "--channels", type=int, default=NUM_CHANNELS, help="Number of channels"
    )
    parser.add_argument("-r", "--rate", type=int, default=SAMPLE_RATE, help="Sample rate")

    args = parser.parse_args()

    Daily.init()

    app = SendWavApp(args.input, args.rate, args.channels)

    try:
        app.run(args.meeting, args.token)
    except KeyboardInterrupt:
        print("Ctrl-C detected. Exiting!")
    finally:
        app.leave()


if __name__ == "__main__":
    main()
