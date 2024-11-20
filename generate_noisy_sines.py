"""
Generates noisy sines for use in the Zooniverse
"""
from csv import writer
from typing import Tuple, List
from pathlib import Path
import subprocess

import numpy as np
from plotly.graph_objects import Figure, Scatter, Layout
from tqdm import tqdm
from numpy.typing import NDArray
from numpy import float32
from sonounolib.tracks import Track
from sonounolib.notes import get_note_frequencies

from ffmpeg import FFmpeg

# This is from SonoUno Desktop...
from data_import.data_import import DataImport
from data_transform.predef_math_functions import PredefMathFunctions
from data_export.data_export import DataExport
from sound_module.simple_sound import simpleSound, reproductorRaw


def generate_times(frequency: int, duration: int) -> NDArray[float32]:
    """
    Generates a series of times, in seconds.

    :param frequency: Frequency in seconds
    :param duration: Duration in seconds
    :return: The time range
    """
    return np.linspace(0, duration, duration * frequency, endpoint=True, dtype=float32)


def generate_sine_wave(time: NDArray[float32], period: int, magnitude: float = 1.0) -> NDArray[float32]:
    """
    Generates a sine wave, with an amplitude of 1.

    :param time: The times to generate the sine for
    :param period: period in seconds

    :return: The base sine wave
    """
    return np.sin(2 * np.pi * time / period) * magnitude


def apply_noise_to_signal(
        signal: NDArray[float32], sigma: float, seed: int,
        rescale_to: float|None = None,
        clip_to: Tuple[float, float]|None = None,
) -> NDArray[float32]:
    """
    Applies a normal distribution of noise to the signal.

    :param signal: The signal
    :param sigma: The magnitude of the noise
    :param seed: The random seed
    :param rescale_to: The output is rescaled to have a maximum of this value, if provided
    :param clip_to: The range to clip the signal between, if provided
    :return: The signal, with gaussian noise applied
    """
    signal_modified: NDArray[float32] = signal + np.random.normal(0, sigma, signal.shape)
    if rescale_to:
        signal_modified *= rescale_to / signal_modified.max()
    if clip_to:
        signal_modified = np.clip(signal_modified, clip_to[0], clip_to[1])

    return signal_modified


def save_signal_to_csv(
        path: str|Path,
        time: NDArray[float32],
        signal: NDArray[float32]
):
    """
    Saves the signal data to a text file.

    :param filename:
    :param time: The array of times the signal is recorded
    :param signal: The signal
    """
    np.savetxt(path, np.column_stack((time, signal)), fmt='%.6f', header='Time, Signal', comments='')


def plot_signals(
        time: NDArray[float32], signals: List[NDArray[float32]],
        path: str|Path, title: str, names: List[str],
        x_label: str = 'Time (seconds)', y_label: str = 'Amplitude'
) -> None:
    """
    Plots the signal to file

    :param time: The array of times the signal is recorded
    :param path: The root path to save as (html suffix added)
    :param title: The title of the plot
    :param signals: The list of signals
    :param names: The names of the signals
    :param x_label: The label of the x-axis, defaults to 'Time (seconds)'
    :param y_label: The label of the y-axis, defaults to 'Amplitude'
    """
    fig: Figure = Figure(
        data=[
            Scatter(
                {'x': time, 'y': signal, 'name':name}
            ) for name, signal in zip(names[::-1], signals[::-1])
       ],
        layout=Layout(
            title={'text': title},
            xaxis_title={'text': x_label},
            yaxis_title={'text': y_label},
        )
    )
    # fig.show()
    fig.write_image(
        file=str(path.with_suffix(path.suffix + '.jpg')),
        width=800,
    )



def plot_signal(
        time: NDArray[float32], signal: NDArray[float32],
        path: str|Path,
        x_label: str = 'Time (seconds)',
        y_label: str = 'Amplitude'
):
    """
    Plots one signal to file

    :param time: The array of times the signal is recorded
    :param filename: The filename to save as
    :param signals: The signal to plot
    :param x_label: The label of the x-axis, defaults to 'Time (seconds)'
    :param y_label: The label of the y-axis, defaults to 'Amplitude'
    """
    fig: Figure = Figure(
        data=[
            Scatter(
                {'x': time, 'y': signal}
            )
       ],
        layout=Layout(
            xaxis_title={'text': x_label},
            yaxis_title={'text': y_label},
        )
    )
    # fig.show()
    fig.write_image(
        file=path,
        width=640,
    )


def save_signal_to_wav(
        path: str|Path,
        signal: NDArray[float32],
        frequency: int,
):
    """
    Saves a signal as wav file.

    :param path: The path to save the file to
    :param signal: A numpy array containing the signal values
    :param frequency: The frequency of the signal
    """
    path: Path = Path(path)

    track: Track = Track(rate=frequency)
    track.add_raw_data(signal)
    track.to_wav(path)


def convert_wav_to_video(
        sound_path: str|Path,
        image_path: str|Path,
        video_path: str|Path,
):
    """
    Converts a wav file and image into a video

    :param sound_path: Path to the input sound file, deleted after use
    :param image_path: Path to the input image file, deleted after use
    :param video_path: Path to the output video file, overwritten
    :returns: None
    """
    video_path.unlink(missing_ok=True)
    ffmpeg: FFmpeg = (
        FFmpeg()
        .input(image_path)
        .input(sound_path)
        .output(
            video_path,
            {
                'codec:v': 'libx264',
                'codec:a': 'mp3',
                'preset:v': 'veryslow',
            },
        )
    )
    ffmpeg.execute()


def generate_sonouno_wav(
        path: str|Path,
        time: NDArray[float32],
        signal: NDArray[float32],
        frequency: float,
):
    """
    """
    simple_sound: simpleSound = simpleSound()
    reproductor: reproductorRaw = simple_sound.reproductor
    reproductor.set_waveform('celesta')
    reproductor.set_time_base(1/frequency)
    reproductor.set_min_freq(300)
    reproductor.set_max_freq(1200)
    simple_sound.save_sound(str(path), time, signal)


def generate_sonouno_video_for_signal(
        path_root: str|Path,
        time: NDArray[float32],
        signal: NDArray[float32],
        frequency: int,
):
    """
    """
    path_root: Path = Path(path_root)
    image_path: Path = path_root.with_suffix(path_root.suffix+".jpg")
    sound_path: Path = path_root.with_suffix(path_root.suffix+".wav")
    video_path: Path = path_root.with_suffix(path_root.suffix+".mp4")

    plot_signal(
        path=image_path,
        time=time,
        signal=signal,
    )
    generate_sonouno_wav(
        path=sound_path,
        time=time,
        signal=signal,
        frequency=frequency,
    )
    convert_wav_to_video(
        sound_path=sound_path,
        image_path=image_path,
        video_path=video_path,
    )
    image_path.unlink()
    sound_path.unlink()


def generate_basic_video_for_signal(
        path_root: str|Path,
        time: NDArray[float32],
        signal: NDArray[float32],
        frequency: int,
):
    """
    """
    path_root: Path = Path(path_root)
    image_path: Path = path_root.with_suffix(path_root.suffix+".jpg")
    sound_path: Path = path_root.with_suffix(path_root.suffix+".wav")
    video_path: Path = path_root.with_suffix(path_root.suffix+".mp4")

    plot_signal(
        path=image_path,
        time=time,
        signal=signal,
    )
    save_signal_to_wav(
        path=sound_path,
        signal=signal,
        frequency=frequency,
    )
    convert_wav_to_video(
        sound_path=sound_path,
        image_path=image_path,
        video_path=video_path,
    )
    image_path.unlink()
    sound_path.unlink()



OUTPUT_PATH_BASIC: Path = Path('output/basic/')
OUTPUT_PATH_SONOUNO: Path = Path('output/sonouno/')

RANDOM_SEED: int = 0  # Seed for reproducibility
# FREQUENCY: int = 22050  # Frequency, in Hz
FREQUENCY: int = 44100  # Frequency, in Hz
FREQUENCY_SONOUNO: int = 20
DURATION: int = 12  # Duration, in seconds
PERIOD: int = 6  # Period, in seconds


def main() -> None:
    """
    Main function to generate and plot noisy sine waves with varying noise levels.
    """
    noise_levels: List[float] = [0.1, 0.2, 0.5]

    time_basic: NDArray[float32] = generate_times(
        duration=DURATION,
        frequency=FREQUENCY,
    )
    signal_clean_basic: NDArray[float32] = generate_sine_wave(
       time=time_basic,
       period=PERIOD,
       magnitude=1.0,
    )
    signal_rows: List[Path] = [
        (
            "sine_basic.mp4",
            "Clean signal",
            0,
            None,
            None,
        ),
    ]
    generate_basic_video_for_signal(
        path_root=OUTPUT_PATH_BASIC / "sine_basic",
        time=time_basic,
        signal=signal_clean_basic,
        frequency=FREQUENCY,
    )

    for noise_level in tqdm(noise_levels, desc="Generating basic sounds"):
        path_rescaled: Path = OUTPUT_PATH_BASIC / f"sine_basic_rescaled_{noise_level}"
        path_clipped: Path = OUTPUT_PATH_BASIC / f"sine_basic_clipped_{noise_level}"

        signal_rows.append(
            (
                f"sine_basic_rescaled_{noise_level}.mp4",
                "Noisy signal",
                noise_level,
                1,
                None,
            )
        )
        signal_rows.append(
            (
                f"sine_basic_clipped_{noise_level}.mp4",
                "Noisy signal",
                noise_level,
                None,
                1,
            )

        )
        generate_basic_video_for_signal(
            path_root=path_rescaled,
            time=time_basic,
            signal=apply_noise_to_signal(
                signal=signal_clean_basic,
                sigma=noise_level,
                seed=RANDOM_SEED,
                rescale_to=1.0
            ),
            frequency=FREQUENCY,
        )
        generate_basic_video_for_signal(
            path_root=path_clipped,
            time=time_basic,
            signal=apply_noise_to_signal(
                signal=signal_clean_basic,
                sigma=noise_level,
                seed=RANDOM_SEED,
                rescale_to=1.0
            ),
            frequency=FREQUENCY,
        )

    with open(OUTPUT_PATH_BASIC / "manifest_basic.csv", 'w', encoding="UTF-8") as manifest_file:
        manifest_writer: writer = writer(manifest_file)
        manifest_writer.writerow(
            (
                "subject_id",
                "image_name_1",
                "description",
                "sigma",
                "rescaled_to",
                "clipped_to"
            )
        )

        for idx, signal_row in enumerate(signal_rows):
            manifest_writer.writerow(
                (
                    idx,
                    *signal_row,
                )
            )


    time_sonouno: NDArray[float32] = generate_times(
        duration=DURATION,
        frequency=FREQUENCY_SONOUNO,
    )
    signal_clean_sonouno: NDArray[float32] = generate_sine_wave(
       time=time_sonouno,
       period=PERIOD,
       magnitude=1.0,
    )
    signal_rows: List[Path] = [
        (
            "siine_sonouno.mp4",
            "Clean signal",
            0,
            None,
            None,
        )
    ]
    generate_sonouno_video_for_signal(
        path_root=OUTPUT_PATH_SONOUNO / "sine_sonouno",
        time=time_sonouno,
        signal=signal_clean_basic,
        frequency=FREQUENCY_SONOUNO,
    )

    for noise_level in tqdm(noise_levels, desc="Generating SonoUno format"):
        path_rescaled: Path = OUTPUT_PATH_SONOUNO / f"sine_sonouno_rescaled_{noise_level}"
        path_clipped: Path = OUTPUT_PATH_SONOUNO / f"sine_sonouno_clipped_{noise_level}"

        signal_rows.append(
            (
                f"sine_sonouno_rescaled_{noise_level}.mp4",
                "Noisy signal",
                noise_level,
                1,
                None,
            )
        )
        signal_rows.append(
            (
                f"sine_sonouno_clipped_{noise_level}.mp4",
                "Noisy signal",
                noise_level,
                None,
                1,
            )
        )

        generate_sonouno_video_for_signal(
            path_root=path_rescaled,
            time=time_sonouno,
            signal=apply_noise_to_signal(
                signal=signal_clean_sonouno,
                sigma=noise_level,
                seed=RANDOM_SEED,
                rescale_to=1,
            ),
            frequency=FREQUENCY_SONOUNO,
        )
        generate_sonouno_video_for_signal(
            path_root=path_clipped,
            time=time_sonouno,
            signal=apply_noise_to_signal(
                signal=signal_clean_sonouno,
                sigma=noise_level,
                seed=RANDOM_SEED,
                clip_to=[-1., 1.],
            ),
            frequency=FREQUENCY_SONOUNO,
        )

    with open(OUTPUT_PATH_SONOUNO / "manifest_sonouno.csv", 'w', encoding="UTF-8") as manifest_file:
        manifest_writer: writer = writer(manifest_file)
        manifest_writer.writerow(
            (
                "subject_id",
                "image_name_1",
                "description",
                "sigma",
                "rescaled_to",
                "clipped_to"
            )
        )
        for idx, signal_row in enumerate(signal_rows):
            manifest_writer.writerow(
                (
                    idx,
                    *signal_row,
                )
            )



if __name__ == '__main__':
    main()
