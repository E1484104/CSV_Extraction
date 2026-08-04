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

After each CSV is written, the script opens a Matplotlib plot window. Use the window
toolbar to save the plot manually, or close the window without saving.

You can still override those paths from the command line:

```powershell
python main.py --input "path\to\image.png" --output "curve.csv"
```

The CSV contains original image pixel coordinates. It does not normalize each image on
its own, so multiple CSV files can be normalized together later:

```csv
x_px,y_px
514,412
515,411
```

Use `normalize_csv_folder.py` when you want shared normalization across a folder of CSV
files.

## Line Plot

By default, the line plot is shown in a Matplotlib window and is not saved
automatically. If calibrated columns are available, the plot uses `x_value` and
`y_value`; otherwise it uses `x_px` and `y_px`.

Automatically save a PNG next to the CSV while still showing the window:

```powershell
python main.py --input "image.png" --output "curve.csv" --save-plot
```

Specify a saved plot path for one image:

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

## Batch Normalize CSV Files

`normalize_csv_folder.py` is a standalone helper. It is not connected to `main.py`.
It reads all CSV files in one folder, maps each CSV's x range to the same time span,
uses one shared y range across the folder, and writes the same number of normalized CSV
files to a new folder.

```powershell
python normalize_csv_folder.py --input "csv" --output "csv_normalized"
```

Each output CSV preserves the original columns and adds:

```csv
x_norm,y_norm
```

By default, each file's `x_px` range is mapped to `x_norm=0..30`, because each stitched
long image records 30 seconds. The y values are normalized together across all files.
`y_norm` is inverted so image pixel coordinates become normal plot coordinates, with
larger values higher on the graph.

Useful options:

```powershell
python normalize_csv_folder.py --input "csv" --output "csv_normalized" `
  --x-column x_px --y-column y_px

python normalize_csv_folder.py --input "csv" --output "csv_normalized" `
  --duration-s 30

python normalize_csv_folder.py --input "csv" --output "csv_normalized" `
  --recursive
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
and reverse them. Full-screen `1412x932` Vevo exports use a built-in fixed waveform ROI
of `267,155,1024,616`, so separate stitched outputs keep the same height and vertical
pixel coordinate system. If the layout is different, provide the waveform crop manually:

```powershell
python vevo_stitcher.py --input "frames" --output "stitched_waveform.png" `
  --roi "267,155,1024,616"
```

Input images may have different widths as long as their heights and y-axis pixel scale
match. The matcher compares only the actual same-width overlap region. By default it
searches shifts up to the narrower image width minus `--min-overlap`. Use `--max-shift`
only when you want to manually cap that search range.

The stitcher detects only blue and green Image Peak traces. The default `--feature auto`
chooses between those two colors; use `--feature blue` or `--feature green` if the
automatic choice is not what you want. The default output keeps only the detected curve
pixels, including horizontal curve segments near the axis, which avoids broken white
axis fragments in the stitched image.

```powershell
python vevo_stitcher.py --input "frames" --output "stitched_waveform.png" `
  --feature green
```

If you want a continuous horizontal axis, add `--draw-axis`:

```powershell
python vevo_stitcher.py --input "frames" --output "stitched_waveform.png" `
  --draw-axis
```

For non-standard layouts, use `--auto-roi` to fall back to curve-based ROI detection.
If you need to keep more empty graph area below the trace in that mode, increase only
the bottom padding:

```powershell
python vevo_stitcher.py --input "frames" --output "stitched_waveform.png" `
  --auto-roi --y-padding-bottom 430 --draw-axis
```

Seam checking is enabled automatically for blue/green traces. The main alignment still
comes from the whole overlapping waveform region. If an adjacent pair has a very high
overlap score, the stitcher accepts it even when the exact seam endpoint is missing and
prints a `seam warning`. If one or more frames are skipped, the default recovery mode
accepts a later frame by overlap score only; it does not assume a fixed `102/103px`
step. Use `--skip-recovery-mode expected-shift` if you want the stricter raw-frame
behavior that also checks recent shift trends. Use `--seam-check on` if you want the
seam endpoint test to be strict. Disable seam checking if you want every frame appended:

```powershell
python vevo_stitcher.py --input "frames" --output "stitched_waveform.png" `
  --seam-check off
```

If valid adjacent seams are still being skipped, relax the seam tolerances or lower the
high-score bypass threshold. If you use expected-shift recovery and it is too strict,
relax the recovery shift tolerance:

```powershell
python vevo_stitcher.py --input "frames" --output "stitched_waveform.png" `
  --max-seam-y-gap 20

python vevo_stitcher.py --input "frames" --output "stitched_waveform.png" `
  --seam-score-bypass 0.90

python vevo_stitcher.py --input "frames" --output "stitched_waveform.png" `
  --skip-recovery-mode expected-shift --skip-recovery-tolerance 35
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

The CSV will include `x_value` and `y_value` columns in addition to pixel coordinates.

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
