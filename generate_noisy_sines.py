import numpy as np
from plotly.graph_objects import Figure, Scatter, Layout
import random
from csv import writer
from typing import Tuple, List
from numpy.typing import NDArray
from numpy import float32
from sonounolib.tracks import Track
from pathlib import Path
from pydub import AudioSegment


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


def save_signal_to_csv(filename: str|Path, time: NDArray[float32], signal: NDArray[float32]):
    """
    Saves the signal data to a text file.

    :param filename:
    :param time: The array of times the signal is recorded
    :param signal: The signal
    """
    np.savetxt(filename, np.column_stack((time, signal)), fmt='%.6f', header='Time, Signal', comments='')


def plot_signals(
        time: NDArray[float32], signals: List[NDArray[float32]] ,
        filename: str|Path, title: str, names: List[str],
        x_label: str = 'Time (seconds)', y_label: str = 'Amplitude'
) -> None:
    """
    Plots the signal to file

    :param time: The array of times the signal is recorded
    :param filename: The root filename to save as (html suffix added)
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
    # fig.write_image(f'{filename}.svg')
    fig.write_html(f'{filename}.html')


def save_signal_to_mp3(
        filename: str|Path, signal: NDArray[float32], frequency: int
):
    """
    Saves a signal as an MP3 file, going via Track and Wav
    """
    filename: Path = Path(filename)
    path_mp3: Path = filename.with_suffix(filename.suffix + '.mp4')
    path_wav: Path = filename.with_suffix(filename.suffix + '.wav')

    track: Track = Track(rate=FREQUENCY)
    track.add_raw_data(signal)
    track.to_wav(path_wav)

    sound: AudioSegment = AudioSegment.from_wav(path_wav)
    sound.export(path_mp3, format='mp4')
    path_wav.unlink()


PLACEHOLDER_PATH: Path = Path("placeholder.jpg")
PLACEHOLDER_BYTES: bytes = PLACEHOLDER_PATH.read_bytes()

OUTPUT_PATH: Path = Path('output/')

RANDOM_SEED: int = 0  # Seed for reproducibility
FREQUENCY: int = 22050  # Frequency, in Hz
# FREQUENCY: int = 44100  # Frequency, in Hz
DURATION: int = 12  # Duration, in seconds
PERIOD: int = 6  # Period, in seconds


def main() -> None:
    """
    Main function to generate and plot noisy sine waves with varying noise levels.
    """
    time: NDArray[float32] = generate_times(FREQUENCY, DURATION)
    signal_small_clean: NDArray[float32] = generate_sine_wave(time, PERIOD, magnitude=0.5)
    signal_large_clean: NDArray[float32] = generate_sine_wave(time, PERIOD, magnitude=1.0)

    noise_levels: List[float] = [0.1, 0.2, 0.5]

    signals_rescaled: List[NDArray[float32]] = [
        apply_noise_to_signal(
            signal_large_clean, sigma=noise_level, seed=RANDOM_SEED, rescale_to=1.0
        ) for noise_level in noise_levels
    ]
    signals_clipped: List[NDArray[float32]] = [
        apply_noise_to_signal(
            signal_large_clean, sigma=noise_level, seed=RANDOM_SEED, clip_to=(-1., 1.)
        ) for noise_level in noise_levels
    ]

    plot_signals(
        time=time,
        signals=[signal_large_clean]+signals_rescaled,
        title="Signals for a range of noise levels, rescaled to 1",
        filename=OUTPUT_PATH / 'signals-rescaled',
        names=[f"σ={sigma}" for sigma in [0]+noise_levels],
    )

    plot_signals(
        time=time,
        signals=[signal_large_clean]+signals_clipped,
        title="Signals for a range of noise levels, clipped to 1",
        filename=OUTPUT_PATH / 'signals-clipped',
        names=[f"σ={sigma}" for sigma in [0]+noise_levels],
    )
    save_signal_to_mp3(
        filename=OUTPUT_PATH / "sine_large",
        signal=signal_large_clean,
        frequency=FREQUENCY,
    )
    save_signal_to_csv(
        filename=OUTPUT_PATH / "sine_large.csv",
        time=time,
        signal=signal_large_clean
    )

    Path(
        OUTPUT_PATH / "sine_large.jpg"
    ).write_bytes(
        PLACEHOLDER_BYTES
    )

    for noise_level, signal in zip(noise_levels, signals_rescaled):
        save_signal_to_mp3(
            filename=OUTPUT_PATH / f"sine_large_rescaled_{noise_level}",
            signal=signal, 
            frequency=FREQUENCY,
        )
        save_signal_to_csv(
            filename=OUTPUT_PATH / f"sine_large_rescaled_{noise_level}.csv",
            time=time,
            signal=signal,
        )
        Path(
            OUTPUT_PATH / f"sine_large_rescaled_{noise_level}.jpg"
        ).write_bytes(
            PLACEHOLDER_BYTES
        )


    for noise_level, signal in zip(noise_levels, signals_clipped):
        save_signal_to_mp3(
            filename=OUTPUT_PATH / f"sine_large_clipped_{noise_level}", 
            signal=signal, 
            frequency=FREQUENCY,
        )
        save_signal_to_csv(
            filename=OUTPUT_PATH / f"sine_large_clipped_{noise_level}.csv",
            time=time,
            signal=signal,
        )
        Path(
            OUTPUT_PATH / f"sine_large_clipped_{noise_level}.jpg"
        ).write_bytes(
            PLACEHOLDER_BYTES
        )


    with open(OUTPUT_PATH / "manifest.csv", 'w', encoding="UTF-8") as manifest_file:
        manifest_writer: writer = writer(manifest_file)
        manifest_writer.writerow(
            ("subject_id", "image_name_1", "image_name_2", "description", "sigma", "rescaled_to", "clipped_to")
        )

        idx: int = 0
        manifest_writer.writerow(
            (
                0,
                "sine_large.mp3", 
                "sine_large.jpg", 
                "Clean signal", 0, None, None
            )
        )

        for idx, noise_level in enumerate(noise_levels, start=idx+1):
            manifest_writer.writerow(
                (
                    idx,
                    f"sine_large_rescaled_{noise_level}.mp3",
                    f"sine_large_rescaled_{noise_level}.jpg",
                    "Noisy signal", noise_level, 1.0, None
                )
            )

        for idx, noise_level in enumerate(noise_levels, start=idx+1):
            manifest_writer.writerow(
                (
                    idx,
                    f"sine_large_clipped_{noise_level}.mp3",
                    f"sine_large_clipped_{noise_level}.jpg",
                    "Noisy signal", noise_level, None, 1.0
                )
            )


if __name__ == '__main__':
    main()
