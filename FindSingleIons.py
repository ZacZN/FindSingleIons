import numpy as np
import matplotlib.pyplot as plt
from iqtools import plotters, tools
import os
import toml
import logging


# Load the config file
with open("config.toml", "r") as f:
    config = toml.load(f)

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

if set_logging.lower() == "true":
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)



# Returns a list of files in the filepath
def get_files():
    datafiles = os.listdir(path=file_path)
    sorted_files = sorted(datafiles)

    return sorted_files


# Return x and y slices for zooming into the spectrogram
def get_slices():
    sly = slice(0, nframes-(sframes+1))
    slx = slice(int(lframes*whole_region_start), int(lframes*whole_region_end))

    return sly, slx


# Read in the data and get spectrogram based on parameters specified in the config file
# Returns xx, yy, zz of spectrogram sliced to zoom in on the region of interest according 
# to the config
def load_file(file_list: list, file_index: int, sly: slice, slx: slice):
    file = file_path + file_list[file_index]
    iq = tools.get_iq_object(file)

    logger.info(f"File: {file_list[file_index]}")
    logger.info(f"Total samples: {iq.nsamples_total}")
    logger.info(f"LFRAMES: {lframes}")
    logger.info(f"NFRAMES: {nframes}")
    logger.info(f"Total requested samples: {lframes*nframes}")
    logger.info(f"SFRAMES {sframes}")

    #nframes = int((iq.nsamples_total/iq.fs)/(lframes/iq.fs))
    
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

    return xx[sly,slx], yy[sly,slx], zz[sly,slx], file_list[file_index]





def main():
    # run all the functions


    file_index = 0
    
    file_list = get_files()
    sly, slx = get_slices()

    for file in range(len(file_list)):
        xx_orig, yy_orig, zz_orig, filename = load_file(
            file_list=file_list,
            file_index=file_index,
            sly=sly,
            slx=slx
        )

        xx_cooled, yy_cooled, zz_cooled = tools.get_cooled_spectrogram(
            xx=xx_orig,
            yy=yy_orig,
            zz=zz_orig,
            yy_idx=nframes-(sframes+2)
        )

        xx_avg, yy_avg, zz_avg = tools.get_averaged_spectrogram(
            xx=xx_cooled,
            yy=yy_cooled,
            zz=zz_cooled,
            every=avg_every
        )

        zz_sum = np.zeros(np.shape(xx_avg)[1])
        for i in range(len(zz_avg)):
            zz_sum += zz_avg[i]

        log_zz_sum = np.log10(zz_sum)

        ref_integral = sum(log_zz_sum[ref_start:ref_end])
        peak_integral = sum(log_zz_sum[peak_start:peak_end])

        ref_avg = ref_integral / (ref_end - ref_start)
        peak_avg = peak_integral / (peak_end - peak_start)

        if peak_avg - ref_avg > peak_criterion:
            # record filename to textfile and save spectra
            
            with open(found_peaks_output_path + "single-ions.txt", "a") as f:
                f.write(filename)

            fig, ax = plt.subplots(figsize=(16,10))
            plt.stairs(log_zz_sum[:-1], xx_avg[:])
            plt.savefig(spectra_output_path + filename + ".png")
            plt.close()




if __name__ == "__main__":
    main()