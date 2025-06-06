# Video Watermarking System

A robust video watermarking system that embeds and extracts binary messages (0 or 1) from videos using DCT (Discrete Cosine Transform) domain techniques. The system is designed to be imperceptible to human vision while being robust against compression and noise attacks.

## Features

- **Invisible Watermarking**: Embeds binary messages without visible artifacts
- **DCT Domain**: Uses frequency domain embedding for robustness
- **Redundant Embedding**: Multiple embedding locations per frame for reliability
- **Comprehensive Testing**: Includes PSNR/SSIM quality metrics and robustness testing
- **Attack Simulation**: Tests against compression and Gaussian noise attacks

## Files

1. **`video_encoder.py`** - Embeds binary messages into videos
2. **`video_decoder.py`** - Extracts binary messages from watermarked videos
3. **`video_tester.py`** - Comprehensive testing and evaluation framework

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Encoding a Message

Embed a binary message (0 or 1) into a video:

```bash
python video_encoder.py input_video.mp4 watermarked_video.mp4 1
```

**Arguments:**
- `input_video`: Path to the original video file
- `output_video`: Path where the watermarked video will be saved
- `message`: Binary message to embed (0 or 1)
- `--alpha`: (Optional) Watermark strength (default: 0.1, lower = more imperceptible)

**Example:**
```bash
python video_encoder.py 8A7IWeZulKs.mp4 watermarked_video.mp4 1 --alpha 0.1
```

### 2. Decoding a Message

Extract the embedded message from a watermarked video:

```bash
python video_decoder.py watermarked_video.mp4
```

**Arguments:**
- `input_video`: Path to the watermarked video file
- `--frames`: (Optional) Number of frames to analyze (default: all frames)
- `--detailed`: (Optional) Perform detailed frame-by-frame analysis
- `--sample-rate`: (Optional) Sample rate for detailed analysis (default: 10)

**Examples:**
```bash
# Basic decoding
python video_decoder.py watermarked_video.mp4

# Analyze only first 100 frames
python video_decoder.py watermarked_video.mp4 --frames 100

# Detailed analysis with frame-by-frame results
python video_decoder.py watermarked_video.mp4 --detailed --sample-rate 5
```

### 3. Testing and Evaluation

Compare original vs watermarked videos and test robustness:

```bash
python video_tester.py original_video.mp4 watermarked_video.mp4 1
```

**Arguments:**
- `original_video`: Path to original video file
- `watermarked_video`: Path to watermarked video file
- `original_message`: The message that was embedded (0 or 1)
- `--frames`: (Optional) Number of frames to compare
- `--plot`: (Optional) Generate quality metrics plot
- `--plot-save`: (Optional) Save plot to specified path
- `--skip-robustness`: (Optional) Skip robustness testing

**Examples:**
```bash
# Full testing with robustness evaluation
python video_tester.py 8A7IWeZulKs.mp4 watermarked_video.mp4 1

# Quality comparison only (faster)
python video_tester.py 8A7IWeZulKs.mp4 watermarked_video.mp4 1 --skip-robustness

# Generate and save quality plots
python video_tester.py 8A7IWeZulKs.mp4 watermarked_video.mp4 1 --plot-save quality_metrics.png
```

## Complete Workflow Example

Here's a complete example using your test video:

```bash
# 1. Encode message "1" into the video
python video_encoder.py 8A7IWeZulKs.mp4 watermarked_test.mp4 1

# 2. Decode the message from the watermarked video
python video_decoder.py watermarked_test.mp4

# 3. Test quality and robustness
python video_tester.py 8A7IWeZulKs.mp4 watermarked_test.mp4 1
```

## Technical Details

### Watermarking Algorithm

- **Domain**: DCT (Discrete Cosine Transform) frequency domain
- **Embedding Strategy**: Modifies relationship between mid-frequency coefficients
- **Redundancy**: Embeds the same bit in multiple 8x8 blocks per frame
- **Color Space**: Works in YUV color space (embeds in luminance channel)

### Quality Metrics

- **PSNR (Peak Signal-to-Noise Ratio)**: Measures pixel-level distortion
  - > 40 dB: Excellent (imperceptible)
  - 30-40 dB: Good (barely perceptible)
  - 20-30 dB: Fair (perceptible)
  - < 20 dB: Poor (very noticeable)

- **SSIM (Structural Similarity Index)**: Measures perceptual similarity
  - Range: 0 to 1 (1 = identical)
  - > 0.95: Excellent quality
  - 0.9-0.95: Good quality
  - < 0.9: Noticeable differences

### Robustness Testing

The system tests robustness against:
- **Compression attacks**: JPEG compression at various quality levels (90%, 70%, 50%, 30%, 10%)
- **Noise attacks**: Gaussian noise at different strength levels (5, 10, 15, 20, 25)

## Expected Performance

With the default settings (alpha=0.1):
- **Quality**: PSNR > 35 dB, SSIM > 0.95
- **Robustness**: Should survive compression down to 50% quality
- **Noise tolerance**: Should handle Gaussian noise up to strength 15

## Troubleshooting

1. **Low PSNR/SSIM**: Reduce alpha value (e.g., --alpha 0.05)
2. **Poor robustness**: Increase alpha value (e.g., --alpha 0.2)
3. **Extraction failures**: Try analyzing more frames or use detailed mode
4. **Memory issues**: Process fewer frames at a time using --frames parameter

## System Requirements

- Python 3.7+
- OpenCV 4.5+
- NumPy, SciPy, scikit-image, matplotlib
- Sufficient RAM for video processing (depends on video size)

The system has been optimized for a balance between imperceptibility and robustness. Adjust the `alpha` parameter based on your specific requirements. #   v i d e o - w a t e r m a r k i n g  
 