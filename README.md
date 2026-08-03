# CSV Extraction

Extract a colored plot curve from a local image and save the traced line as a CSV file.

This is intended for screenshots or exported figures where the target curve color is reasonably consistent against the background. The line does not need to stay in the same value range across images.

## Install

```powershell
python -m pip install -r requirements.txt
```

## Basic Usage

Edit the paths near the top of `config.py`:

```python
INPUT_PATH = Path(r"input.png")
OUTPUT_PATH = Path(r"curve.csv")
```

Then run:

```powershell
python main.py
```

After each CSV is written, the script also writes a line plot PNG next to it using the
same stem, for example `curve.csv` and `curve.png`.

You can still override those paths from the command line:

```powershell
python main.py --input "path\to\image.png" --output "curve.csv"
```

The CSV contains normalized coordinates and original image pixel coordinates:

```csv
x_norm,y_norm,x_px,y_px
0,0.527,514,412
0.001,0.531,515,411
```

`x_norm` runs from left to right. `y_norm` is normalized bottom-to-top, which matches normal plot coordinates.

## Line Plot

By default, the line plot is written as a PNG with the same stem as the CSV output.
If calibrated columns are available, the plot uses `x_value` and `y_value`; otherwise
it uses `x_norm` and `y_norm`. Plots are rendered with Matplotlib.

Specify a plot path for one image:

```powershell
python main.py --input "image.png" --output "curve.csv" --plot-output "curve_plot.png"
```

For batch processing, `--plot-output` is treated as a directory:

```powershell
python main.py --input "images" --output "csv" --plot-output "plots"
```

Skip plot generation and keep CSV-only output:

```powershell
python main.py --input "image.png" --output "curve.csv" --no-plot
```

## WAV Doppler Envelope

The program can also read standard uncompressed signed 16-bit PCM WAV files directly.
For WAV input, it does not use image recognition. It runs an STFT on each channel and
estimates a per-frequency background noise floor over time. It subtracts that noise
floor from each time window, keeps continuous frequency regions above threshold, and
uses spectral roll-off to estimate the Doppler frequency boundary. This avoids treating
isolated high-frequency noise near Nyquist as the envelope.

```powershell
python main.py --input "Wave.wav" --output "Wave.csv"
```

The WAV CSV uses long format so channels are kept separate before their physical
meaning is confirmed:

```csv
time_s,channel,peak_frequency_hz,envelope_normalized,raw_peak_frequency_hz,confidence,snr_db,is_interpolated
0.02321995465,channel_1,1234.56,0.1436,1241.02,0.82,18.2,0
0.02321995465,channel_0,1320.67,0.1545,1318.44,0.61,15.4,0
```

`channel_1` is processed first because it is usually the stronger Doppler channel, but
both channels are retained. `envelope_normalized` maps `peak_frequency_hz` into the
analysis band, so the default 100-8000 Hz band maps to 0-1. Low-confidence detections
are set to blank and only short gaps are interpolated. This is a frequency boundary
estimate, not `abs(samples)`, Hilbert envelope, or sliding RMS.

The WAV plot defaults to the first 1.1 seconds and uses `envelope_normalized`, which
is the right scale for comparing against a 1.1-second exported image trace.

Useful WAV tuning options:

```powershell
python main.py --input "Wave.wav" --output "Wave.csv" `
  --wav-stft-window 2048 --wav-stft-hop 256 `
  --wav-threshold-db 8 --wav-noise-percentile 20 `
  --wav-min-frequency 100 --wav-max-frequency 8000 `
  --wav-rolloff-percentile 95
```

## Experimental Vevo Stitching

`vevo_stitcher.py` is a standalone helper for stitching rolling-window Vevo PNG
exports. It is not connected to `main.py` yet.

```powershell
python vevo_stitcher.py --input "frames" --output "stitched_waveform.png"
```

For files listed in reverse time order, the default `--order auto` can usually detect
and reverse them. If auto ROI is not reliable, provide the waveform crop manually:

```powershell
python vevo_stitcher.py --input "frames" --output "stitched_waveform.png" `
  --roi "267,151,1024,257"
```

For blue Image Peak traces, seam checking is enabled automatically. A frame is appended
only when the previous tail and current head are close at the stitch seam; otherwise the
candidate frame is skipped and reported. Accepted seams are also repaired with a small
overlap max-blend and a short bridge between detected curve endpoints, which reduces
hard-cut breaks at steep peaks. Disable this conservative behavior if you want every
frame appended:

```powershell
python vevo_stitcher.py --input "frames" --output "stitched_waveform.png" `
  --seam-check off
```

If valid seams are being skipped, relax the vertical seam tolerance:

```powershell
python vevo_stitcher.py --input "frames" --output "stitched_waveform.png" `
  --max-seam-y-gap 20
```

## Specify Curve Color

Automatic color detection works when the target curve is the dominant bright saturated color. If the image has several colored traces or labels, specify the curve color:

```powershell
python main.py --input "image.png" --output "curve.csv" --target-color "#b448ff"
```

RGB form also works:

```powershell
python main.py --input "image.png" --output "curve.csv" --target-color "180,72,255"
```

Increase tolerance if the curve has anti-aliased edges or compression artifacts:

```powershell
python main.py --input "image.png" --output "curve.csv" --target-color "#b448ff" --tolerance 70
```

## Crop To Plot Area

Use ROI when text, legends, or other same-colored objects interfere with extraction.

Pixel ROI:

```powershell
python main.py --input "image.png" --output "curve.csv" --roi "500,250,1050,600"
```

Normalized ROI:

```powershell
python main.py --input "image.png" --output "curve.csv" --roi "0.25,0.20,0.55,0.65"
```

The ROI format is `x,y,width,height`.

## Real Axis Values

If you know the plot axis limits, provide them to add calibrated columns:

```powershell
python main.py --input "image.png" --output "curve.csv" `
  --x-min 4.2 --x-max 14.4 --y-min -1355 --y-max 581
```

The CSV will include `x_value` and `y_value` columns in addition to normalized and pixel coordinates.

## Batch Processing

```powershell
python main.py --input "images" --output "csv"
```

Each supported image in the input directory is written to a same-stem CSV in the output directory.

## Debug Overlay

Create an image showing the selected ROI and detected curve pixels:

```powershell
python main.py --input "image.png" --output "curve.csv" --debug-image "debug.png"
```

The cyan box is the ROI, the yellow box is the selected curve component, and red pixels are the extracted component.

## Practical Notes

- If the output jumps to labels, legends, or horizontal marker lines, use `--roi` first.
- If auto mode picks the wrong line, use `--target-color`.
- If the detected line is fragmented, increase `--tolerance`.
- If too much noise is included, reduce `--tolerance` or increase `--min-area`.
