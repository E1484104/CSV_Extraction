# CSV Extraction

Extract a colored plot curve from a local image and save the traced line as a CSV file.
The repo also includes standalone helpers for image stitching, CSV normalization,
wearable BPI CSV plotting, and waveform interval matching.

This is intended for screenshots or exported figures where the target curve color is reasonably consistent against the background. The line does not need to stay in the same value range across images.

## Install

```powershell
python -m pip install -r requirements.txt
```

## Basic Usage

For each experiment, either edit `ROOT_PATH` near the top of `config.py`:

```python
ROOT_PATH = Path(r"../TesterX")
```

Or pass the root folder from the command line:

```powershell
python vevo_stitcher.py --root "../TesterX"
python main.py --root "../TesterX" --no-plot
python ultrasound_normalizer.py --root "../TesterX"
```

The default workflow expects this folder layout:

```text
TesterX/
  Raw_Images/
    series_1/
    series_2/
    series_3/
  Stitched_Images/
  raw_data/
  norm_data/
```

Then run the three stages:

```powershell
python vevo_stitcher.py

python main.py --no-plot

python ultrasound_normalizer.py
```

By default, `vevo_stitcher.py` reads `TesterX/Raw_Images/series_*` and writes
`TesterX/Stitched_Images/TesterX-1.png`, `TesterX-2.png`, and so on. `main.py` reads
`TesterX/Stitched_Images` and writes CSV files to `TesterX/raw_data`. `ultrasound_normalizer.py`
reads `TesterX/raw_data` and writes normalized CSV files to `TesterX/norm_data`.

You can still override `--input` and `--output` from the command line. Explicit
`--input` and `--output` values take priority over `--root`:

```powershell
python main.py --input "path\to\image.png" --output "curve.csv"
```

After each CSV is written, the script opens a Matplotlib plot window. Use the window
toolbar to save the plot manually, or close the window without saving.

The CSV contains original image pixel coordinates. It does not normalize each image on
its own, so multiple CSV files can be normalized together later:

```csv
x_px,y_px
514,412
515,411
```

Use `ultrasound_normalizer.py` when you want shared normalization across CSV files.

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

## Wearable BPI CSV Plot

`wearable_normalizer.py` is a standalone helper. It can read a wearable CSV
such as `Wearable_2.csv` with columns like:

```csv
Timestamp,BPI,PPG
1741065063.3578727,597.286865234375,54.693328857421875
```

It writes a new processed CSV preserving the original columns and adding:

```csv
time_s,bpi_normalized
```

`time_s` is each timestamp minus the first timestamp in the file. `bpi_normalized`
is min-max normalized from the raw BPI column and then smoothed with a 15-point
centered moving average by default. The old post-normalization noise-floor step is
disabled by default because it mostly rescales the signal and randomly removes rows.
Pass `--bottom-envelop` to estimate the lower envelope of the normalized wearable
curve and subtract it from every point so the curve bottom is shifted to zero.

```powershell
python wearable_normalizer.py --input "../20260805/Test2/Wearable_2.csv"
```

By default, the processed CSV is saved next to the input CSV as
`Wearable_2_bpi_processed.csv`. A Matplotlib window opens for inspection and the
script does not save a PNG unless you request it. If you prefer to choose the output
location and save the figure automatically:

```powershell
python wearable_normalizer.py --input "../20260805/Test2/Wearable_2.csv" `
  --output "Wearable_2_bpi_processed.csv" `
  --plot-output "Wearable_2_bpi_normalized.png"
```

If the wearable CSV uses different headers, specify them:

```powershell
python wearable_normalizer.py --input "wearable.csv" --timestamp-column Timestamp --bpi-column BPI
```

For the old noise-floor workflow, pass `--denoise`. It randomly samples normalized
points, averages them as the noise floor, removes those sampled rows, and divides the
remaining normalized values by that floor; because of that division,
`bpi_normalized` can become greater than `1`. Noise-floor sampling is deterministic
by default (`--noise-seed 0`) so repeated runs are comparable.

```powershell
python wearable_normalizer.py --input "wearable.csv" `
  --denoise `
  --noise-sample-count 2000 `
  --noise-seed 0
```

For raw min-max without smoothing, use:

```powershell
python wearable_normalizer.py --input "wearable.csv" `
  --bpi-smooth-window-points 1
```

To bottom-zero the normalized wearable curve after smoothing, use:

```powershell
python wearable_normalizer.py --input "wearable.csv" `
  --bottom-envelop
```

## CSV Interval Matching

`csv_interval_matcher.py` finds the long-CSV interval whose waveform best matches a
short CSV. It slides the short CSV's relative time axis over the long CSV, samples the
long CSV at the shifted short timestamps, scores each candidate window, and prints the
top start times.

By default, it reads short CSV files from `ROOT/Norm_Data` and picks the only CSV in
`ROOT/BPI_Processed` as the long CSV:

```powershell
python csv_interval_matcher.py --root "../20260805/Test22"
```

```powershell
python csv_interval_matcher.py `
  --short "../20260805/Test2/Norm_Data/Test2-1.csv" `
  --long "../20260805/Test2/Wearable_2_bpi_processed.csv"
```

By default it auto-detects time columns from `time_s`, `x_norm`, `x_value`, `x_px`,
or `Timestamp`, and signal columns from `bpi_normalized`, `y_norm`, `y_value`, `y_px`,
or `BPI`. The default `fusion` metric is tuned on the Test11/Test22-style data and
combines ordinary Pearson correlation, smoothed derivative correlation, smoothed
Pearson correlation, and a small morphology feature-correlation term. Absolute-value
diagnostics such as raw normalized MAE/RMSE and Bland-Altman width remain available as
standalone metrics, but they are not part of the default fusion score.

For multiple short CSVs known to be in sorted order, pass the folder. The tool then
selects one ordered, non-overlapping interval for each short CSV:

```powershell
python csv_interval_matcher.py `
  --short-dir "../20260805/Test2/Norm_Data" `
  --long "../20260805/Test2/Wearable_2_bpi_processed.csv"
```

If `ROOT/BPI_Processed` contains more than one CSV, pass `--long` directly or narrow the
folder scan with `--long-dir` and `--long-pattern`.

Compare all available metrics:

```powershell
python csv_interval_matcher.py `
  --short-dir "../20260805/Test2/Norm_Data" `
  --long "../20260805/Test2/Wearable_2_bpi_processed.csv" `
  --metric all --start-step-rows 10
```

Useful options:

```powershell
python csv_interval_matcher.py --short "short.csv" --long "long.csv" `
  --short-x-column x_norm --short-y-column y_norm `
  --long-x-column time_s --long-y-column bpi_normalized `
  --metric fusion `
  --min-start-separation-s 5 `
  --output "top_matches.csv"
```

Show the full long CSV with matched short curves aligned onto the same time axis:

```powershell
python csv_interval_matcher.py --short "short.csv" --long "long.csv" `
  --top 1
```

Save the overlay only when explicitly needed:

```powershell
python csv_interval_matcher.py --short "short.csv" --long "long.csv" `
  --top 1 `
  --plot-output "matched_overlay.png"
```

The overlay uses the matched long time as the bottom x-axis and maps short time with
`long_time = match_start + (short_time - short_start)`. The Matplotlib window shows a
full-long overlay on top and up to four matched-window detail panes below it. The left
y-axis shows the full long signal, the right y-axis shows the aligned short signals,
and each lower detail pane rescales both long and short y-axes to the visible local
window. Short lines are drawn semi-transparently so the underlying long trace remains
visible. For multiple selected matches, pass a folder to `--plot-output` if you also
want PNG files for more than one metric.

If a CSV timestamp column has lost sub-second precision, synthesize the time axis from
row index and the known sampling rate:

```powershell
python csv_interval_matcher.py --short "short.csv" --long "long.csv" `
  --long-x-column row_index --long-sample-rate 250
```

## Batch Normalize CSV Files

`ultrasound_normalizer.py` is a standalone helper. It is not connected to `main.py`.
It accepts either one CSV file or a folder of CSV files. It maps each CSV's x range to
the same time span, uses one shared y range across the selected CSV files, and writes
one normalized CSV for each input CSV.

```powershell
python ultrasound_normalizer.py

python ultrasound_normalizer.py --input "curve.csv" --output "curve_normalized.csv"

python ultrasound_normalizer.py --input "csv" --output "csv_normalized"
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
python ultrasound_normalizer.py --input "csv" --output "csv_normalized" `
  --x-column x_px --y-column y_px

python ultrasound_normalizer.py --input "csv" --output "csv_normalized" `
  --duration-s 30

python ultrasound_normalizer.py --input "csv" --output "csv_normalized" `
  --recursive
```

## Experimental Vevo Stitching

`vevo_stitcher.py` is a standalone helper for stitching rolling-window Vevo PNG
exports. It is not connected to `main.py` yet.

```powershell
python vevo_stitcher.py

python vevo_stitcher.py --input "frames" --output "stitched_waveform.png"
```

For batch stitching, pass multiple frame folders or a parent folder whose children are
frame folders. In that mode `--output` must be a directory:

```powershell
python vevo_stitcher.py --input "run_01" "run_02" --output "stitched"

python vevo_stitcher.py --input "all_runs" --output "stitched"
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
automatic choice is not what you want. Blue/green trace detection also uses an RGB
distance tolerance, defaulting to `150`, so anti-aliased or shifted trace colors are less
likely to be dropped. Increase `--tolerance` if raw frames still produce broken curve
segments, or reduce it if same-colored noise is included. The default output keeps only
the detected curve pixels, including horizontal curve segments near the axis, which
avoids broken white axis fragments in the stitched image.

```powershell
python vevo_stitcher.py --input "frames" --output "stitched_waveform.png" `
  --feature green

python vevo_stitcher.py --input "frames" --output "stitched_waveform.png" `
  --tolerance 180
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

Seam checking defaults to `auto`, and seam bridging is disabled by default. The main
alignment still comes from the whole overlapping waveform region. If one or more frames
are skipped, the default recovery mode accepts a later frame by overlap score only; it
does not assume a fixed `102/103px` step. Use `--seam-check on` if you want the seam
endpoint test to be strict, or `--seam-check off` if you want to disable seam checking.
Use `--bridge-seams` only when you want the program to draw short connecting lines at
seams:

```powershell
python vevo_stitcher.py --input "frames" --output "stitched_waveform.png" `
  --seam-check on

python vevo_stitcher.py --input "frames" --output "stitched_waveform.png" `
  --seam-check off

python vevo_stitcher.py --input "frames" --output "stitched_waveform.png" `
  --bridge-seams
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

Each supported image in the input directory is written to a same-stem CSV in the output
directory. For input directories, `--output` must be a directory.

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
