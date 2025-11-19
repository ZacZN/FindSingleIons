import numpy as np
import matplotlib.pyplot as plt
from iqtools import plotters, tools
import os
import toml
import logging
import time


# Load the config file
with open("config.toml", "r") as f:
    config = toml.load(f)

print(config["settings"])

# Load settings from config
file_path = config["settings"]["file_path"]
spectra_output_path = config["settings"]["spectra_output_path"]
found_peaks_output_path = config["settings"]["found_peaks_output_path"]
lframes = config["settings"]["lframes"]
nframes = config["settings"]["nframes"]
ref_start = config["settings"]["ref_start"]
ref_end = config["settings"]["ref_end"]
peak_start = config["settings"]["peak_start"]
peak_end = config["settings"]["peak_end"]
sframes = config["settings"]["sframes"]
avg_every = config["settings"]["avg_every"]
whole_region_start = config["settings"]["whole_region_start"]
whole_region_end = config["settings"]["whole_region_end"]
peak_criterion = config["settings"]["peak_criterion"]
set_logging = config["settings"]["set_logging"]


# Returns a list of files in the filepath
def get_files() -> list:
    datafiles = os.listdir(path=file_path)
    sorted_files = sorted(datafiles)

    return sorted_files


# Return x and y slices for zooming into the spectrogram
def get_2d_slices() -> tuple:
    sly = slice(0, nframes-(sframes))
    slx = slice(int(lframes*whole_region_start), int(lframes*whole_region_end))

    return sly, slx


# Read in the data and get spectrogram based on parameters specified in the config file
# Returns xx, yy, zz and filename of spectrogram
def load_file(file_list: list, file_index: int) -> tuple:
    file = file_path + file_list[file_index]
    iq = tools.get_iq_object(file)

    print(f"Total Samples: {iq.nsamples_total}")
    
    iq.read(
        nframes=nframes-sframes,
        lframes=lframes,
        sframes=sframes
    )

    iq.window = "hamming"
    iq.method = "fftw"

    xx, yy, zz = iq.get_power_spectrogram(
        nframes=nframes-sframes,
        lframes=lframes,
        sparse=False
    )

    filename = file_list[file_index]

    return xx, yy, zz, filename


# Takes the original xx, yy, zz from the loaded file then cools, averages, and
# cuts the frequency range to that specified in the config file
def process_spectrograms(xx: np.ndarray, yy: np.ndarray, zz: np.ndarray, sly: slice, slx: slice ) -> tuple: 
    xx_cooled, yy_cooled, zz_cooled = tools.get_cooled_spectrogram(
        xx=xx,
        yy=yy,
        zz=zz,
        yy_idx=nframes-(sframes+1)
    )

    xx_cooled_cut, yy_cooled_cut, zz_cooled_cut = xx_cooled[sly,slx], yy_cooled[sly,slx], zz_cooled[sly,slx]

    xx_avg, yy_avg, zz_avg = tools.get_averaged_spectrogram(
        xx=xx_cooled_cut,
        yy=yy_cooled_cut,
        zz=zz_cooled_cut,
        every=avg_every
    )

    return xx_avg, yy_avg, zz_avg


# Adds all of the time frames of the averaged spectrogram, producing a 2d spectrum
# Returns a 1d array of x-values and a 1d array of intensities
def get_spectrum(xx: np.ndarray, zz: np.ndarray):
    zz_sum = np.zeros(np.shape(xx[0]))
    for i in range(len(zz)):
        zz_sum += zz[i]

    zz_log = np.log10(zz_sum)
    xvals = xx[0]

    return xvals, zz_log


# Checks the peak and reference areas specified in the config to look for a peak
def find_ion(zz: np.ndarray, peak_start: int, peak_end: int, ref_start: int, ref_end: int, peak_criterion: float) -> bool:
    print(f"Bins in peak area: {peak_end-peak_start}")
    peak_integral = sum(zz[peak_start:peak_end])
    peak_avg = peak_integral / (peak_end - peak_start)
    print(f"Integral of peak area {peak_integral}")
    print(f"Average of peak area: {peak_avg}")

    print(f"Bins in reference area: {ref_end-ref_start}")
    ref_integral = sum(zz[ref_start:ref_end])
    ref_avg = ref_integral / (ref_end - ref_start)
    print(f"Integral of reference area {ref_integral}")
    print(f"Average of reference area: {ref_avg}")

    diff = peak_avg - ref_avg
    print(f"Difference between peak and reference: {diff}")

    if diff >= peak_criterion:
        outcome = True
    else:
        outcome = False

    return outcome


def save_info(xx: np.ndarray, zz: np.ndarray, filename: str, found_peaks_output_path: str, spectra_output_path: str) -> None:
    with open(found_peaks_output_path + "single-ions.txt", "a") as f:
        # write the name of the file where the single ion was found to a new line
        # in a text file 
        f.write(filename + "\n")

    fig, ax = plt.subplots(figsize=(16,10))
    plt.stairs(
        values=zz[:-1],
        edges=xx
    )
    plt.vlines(
    x=xx[ref_start],
    ymin=min(zz),
    ymax=max(zz),
    color="r",
    linestyles="dashed"
    )
    plt.vlines(
    x=xx[ref_end],
    ymin=min(zz),
    ymax=max(zz),
    color="r",
    linestyles="dashed"
    )
    plt.vlines(
    x=xx[peak_start],
    ymin=min(zz),
    ymax=max(zz),
    color="g",
    linestyles="dashed"
    )
    plt.vlines(
    x=xx[peak_end],
    ymin=min(zz),
    ymax=max(zz),
    color="g",
    linestyles="dashed"
    )    
    plt.savefig(spectra_output_path + filename + ".png")
    plt.close()


def process_file(file_list: list, file_index: int) -> None:
    sly, slx = get_2d_slices()

    xx_orig, yy_orig, zz_orig, filename = load_file(file_list, file_index)

    xx_avg, yy_avg, zz_avg = process_spectrograms(
        xx=xx_orig,
        yy=yy_orig,
        zz=zz_orig,
        sly=sly,
        slx=slx
    )

    xvals, zz_log = get_spectrum(
        xx=xx_avg,
        zz=zz_avg
    )

    ion_present = find_ion(
        zz=zz_log,
        peak_start=peak_start,
        peak_end=peak_end,
        ref_start=ref_start,
        ref_end=ref_end,
        peak_criterion=peak_criterion
    )

    print(f"Ion detected: {ion_present}")

    if ion_present:
        save_info(
            xx=xvals,
            zz=zz_log,
            filename=filename,
            found_peaks_output_path=found_peaks_output_path,
            spectra_output_path=spectra_output_path
        )


def main() -> None:
    files = get_files()

    for i in range(len(files)):
        process_file(files, i)
        time.sleep(0.5)


if __name__=="__main__":
    main()