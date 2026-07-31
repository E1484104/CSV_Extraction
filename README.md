# CSV Extraction

Extract a colored plot curve from a local image and save the traced line as a CSV file.

This is intended for screenshots or exported figures where the target curve color is reasonably consistent against the background. The line does not need to stay in the same value range across images.

## Install

```powershell
python -m pip install -r requirements.txt
```

## Basic Usage

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
