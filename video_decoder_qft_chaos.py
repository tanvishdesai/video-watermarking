import cv2
import numpy as np
import argparse
import os
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from collections import Counter
import matplotlib.pyplot as plt

class QFTChaosVideoWatermarkDecoder:
    def __init__(self, chaos_param=3.8, arnold_iterations=0, multi_scale=False):
        """
        Initialize the QFT-Chaos video watermark decoder (Ultra-Imperceptible Version)
        
        Args:
            chaos_param (float): Chaos parameter for logistic map (must match encoder)
            arnold_iterations (int): Arnold transform iterations (disabled for imperceptibility)
            multi_scale (bool): Multi-scale extraction (disabled for imperceptibility)
        """
        self.chaos_param = chaos_param
        self.arnold_iterations = arnold_iterations
        self.multi_scale = multi_scale
        
        # Initialize chaos sequence (same as encoder)
        self.chaos_seed = 0.123456789
        
        # Single scale only
        self.scales = [1.0]
        
        # Detection parameters for ultra-weak watermarks
        self.confidence_threshold = 0.1  # Very low for ultra-weak signals
    
    def generate_chaos_sequence(self, length):
        """Generate chaotic sequence using logistic map (same as encoder)"""
        sequence = []
        x = self.chaos_seed
        
        for _ in range(length):
            x = self.chaos_param * x * (1 - x)
            sequence.append(x)
        
        # Convert to binary sequence
        binary_sequence = [1 if x > 0.5 else -1 for x in sequence]
        return np.array(binary_sequence)
    
    def seamless_extract_qft_chaos(self, frame):
        """
        Seamless extraction from entire frame to match encoder
        
        Args:
            frame: Input frame
        
        Returns:
            Extracted bit and confidence
        """
        height, width = frame.shape[:2]
        
        # Convert to YUV color space (same as encoder)
        frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        
        # Work only on Y (luminance) channel
        y_channel = frame_yuv[:, :, 0].astype(np.float64)
        
        # Apply FFT to entire frame
        fft_y = fft2(y_channel)
        fft_shifted = fftshift(fft_y)
        
        # Generate chaos sequence for position selection
        total_positions = height * width
        chaos_seq = self.generate_chaos_sequence(total_positions)
        
        # Define extraction positions (same as encoder)
        center_h, center_w = height // 2, width // 2
        
        # Ultra-conservative frequency selection (same as encoder)
        embed_positions = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                if abs(i) == 1 and abs(j) == 1:  # Only 4 positions
                    pos_h = center_h + i
                    pos_w = center_w + j
                    if 0 <= pos_h < height and 0 <= pos_w < width:
                        embed_positions.append((pos_h, pos_w))
        
        # Extract from embedding positions
        bit_estimates = []
        confidences = []
        
        for idx, (pos_h, pos_w) in enumerate(embed_positions):
            if idx < len(chaos_seq):
                # Get coefficient
                coeff = fft_shifted[pos_h, pos_w]
                magnitude = abs(coeff)
                
                # Get reference magnitude from nearby frequencies
                ref_magnitudes = []
                for di in [-3, -2, 2, 3]:
                    for dj in [-3, -2, 2, 3]:
                        ref_h = pos_h + di
                        ref_w = pos_w + dj
                        if 0 <= ref_h < height and 0 <= ref_w < width:
                            ref_magnitudes.append(abs(fft_shifted[ref_h, ref_w]))
                
                if ref_magnitudes:
                    avg_ref_magnitude = np.mean(ref_magnitudes)
                    
                    # Simplified extraction: just compare magnitude to reference
                    # If magnitude > reference, bit is 1; if magnitude < reference, bit is 0
                    magnitude_diff = magnitude - avg_ref_magnitude
                    
                    bit_estimate = 1 if magnitude_diff > 0 else 0
                    confidence = abs(magnitude_diff) / (avg_ref_magnitude + 1e-10) * 100
                    confidence = min(confidence, 1.0)
                    
                    bit_estimates.append(bit_estimate)
                    confidences.append(confidence)
        
        if not bit_estimates:
            return 0, 0.0
        
        # Simple majority voting
        final_bit = 1 if np.mean(bit_estimates) > 0.5 else 0
        final_confidence = np.mean(confidences) if confidences else 0.0
        
        return final_bit, final_confidence
    
    def adaptive_region_extract(self, frame):
        """
        Extract from adaptive regions (alternative method)
        
        Args:
            frame: Input frame
        
        Returns:
            Extracted bit and confidence
        """
        height, width = frame.shape[:2]
        
        # Calculate texture map (same as encoder)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate local variance using a larger window
        kernel = np.ones((15, 15), np.float32) / (15 * 15)
        mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        sqr_mean = cv2.filter2D((gray.astype(np.float32) ** 2), -1, kernel)
        variance = sqr_mean - mean ** 2
        
        # Only extract from medium-high texture regions
        texture_threshold = np.percentile(variance, 70)
        embedding_mask = variance > texture_threshold
        
        # Apply morphological operations
        kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        embedding_mask = cv2.morphologyEx(embedding_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel_morph)
        embedding_mask = cv2.morphologyEx(embedding_mask, cv2.MORPH_CLOSE, kernel_morph)
        
        # Find suitable regions
        embedding_positions = np.where(embedding_mask > 0)
        num_positions = len(embedding_positions[0])
        
        if num_positions == 0:
            return 0, 0.0
        
        # Generate chaos sequence
        chaos_seq = self.generate_chaos_sequence(num_positions)
        
        # Extract from positions (limit to match encoder)
        max_extractions = min(num_positions, 50)
        
        bit_estimates = []
        for i in range(max_extractions):
            pos_y = embedding_positions[0][i]
            pos_x = embedding_positions[1][i]
            chaos_val = chaos_seq[i]
            
            # Get pixel value
            pixel = frame[pos_y, pos_x].astype(np.float64)
            
            # Get reference from neighbors
            ref_pixels = []
            for dy in [-2, -1, 1, 2]:
                for dx in [-2, -1, 1, 2]:
                    ref_y = pos_y + dy
                    ref_x = pos_x + dx
                    if 0 <= ref_y < height and 0 <= ref_x < width:
                        ref_pixels.append(frame[ref_y, ref_x].astype(np.float64))
            
            if ref_pixels:
                avg_ref_pixel = np.mean(ref_pixels, axis=0)
                
                # Calculate difference
                pixel_diff = np.mean(pixel - avg_ref_pixel)
                
                # Account for chaos sequence
                if chaos_val > 0:
                    bit_estimate = 1 if pixel_diff > 0 else 0
                else:
                    bit_estimate = 0 if pixel_diff > 0 else 1
                
                bit_estimates.append(bit_estimate)
        
        if bit_estimates:
            final_bit = 1 if np.mean(bit_estimates) > 0.5 else 0
            confidence = 0.3  # Conservative confidence for this method
        else:
            final_bit = 0
            confidence = 0.0
        
        return final_bit, confidence
    
    def minimal_qft_extract(self, frame):
        """
        Minimal QFT extraction (alternative method)
        
        Args:
            frame: Input frame
        
        Returns:
            Extracted bit and confidence
        """
        height, width = frame.shape[:2]
        
        # Use same region as encoder
        crop_size = min(height, width) // 4
        center_y, center_x = height // 2, width // 2
        start_y = center_y - crop_size // 2
        end_y = start_y + crop_size
        start_x = center_x - crop_size // 2
        end_x = start_x + crop_size
        
        # Extract central region
        central_region = frame[start_y:end_y, start_x:end_x]
        
        # Process green channel
        if len(central_region.shape) == 3:
            g_channel = central_region[:, :, 1].astype(np.float64)
        else:
            g_channel = central_region.astype(np.float64)
        
        # Apply FFT
        fft_g = fft2(g_channel)
        fft_shifted = fftshift(fft_g)
        
        # Check the same position as encoder
        h, w = fft_shifted.shape
        center_h, center_w = h // 2, w // 2
        pos_h = center_h + 1
        pos_w = center_w + 1
        
        if 0 <= pos_h < h and 0 <= pos_w < w:
            # Generate chaos value
            chaos_seq = self.generate_chaos_sequence(1)
            chaos_val = chaos_seq[0]
            
            # Get coefficient
            coeff = fft_shifted[pos_h, pos_w]
            magnitude = abs(coeff)
            
            # Get reference magnitude
            ref_magnitudes = []
            for di in [-2, 2]:
                for dj in [-2, 2]:
                    ref_h = pos_h + di
                    ref_w = pos_w + dj
                    if 0 <= ref_h < h and 0 <= ref_w < w:
                        ref_magnitudes.append(abs(fft_shifted[ref_h, ref_w]))
            
            if ref_magnitudes:
                avg_ref_magnitude = np.mean(ref_magnitudes)
                magnitude_diff = magnitude - avg_ref_magnitude
                normalized_diff = magnitude_diff / (avg_ref_magnitude + 1e-10)
                
                # Account for chaos sequence
                if chaos_val > 0:
                    bit_estimate = 1 if normalized_diff > 0 else 0
                else:
                    bit_estimate = 0 if normalized_diff > 0 else 1
                
                confidence = min(abs(normalized_diff) * 50, 1.0)
                
                return bit_estimate, confidence
        
        return 0, 0.0
    
    def extract_bit_from_frame(self, frame):
        """
        Extract a bit from a frame using the most appropriate method
        
        Args:
            frame: Input frame
        
        Returns:
            Extracted bit and confidence
        """
        # Use seamless extraction for best results
        return self.seamless_extract_qft_chaos(frame)
    
    def decode_video(self, input_path, max_frames=None, detailed=False, sample_rate=10, extraction_method='seamless'):
        """
        Decode binary message from a watermarked video
        
        Args:
            input_path: Path to watermarked video
            max_frames: Maximum number of frames to analyze
            detailed: Whether to perform detailed analysis
            sample_rate: Sample rate for detailed analysis
            extraction_method: Method to use for extraction
        
        Returns:
            Decoded message and analysis results
        """
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            frames_to_analyze = min(max_frames, total_frames)
        else:
            frames_to_analyze = total_frames
        
        print(f"QFT-Chaos Video Watermark Decoder (Ultra-Imperceptible Mode)")
        print(f"Video: {frames_to_analyze} frames, {fps} FPS, {width}x{height}")
        print(f"Chaos parameter: {self.chaos_param}")
        print(f"Extraction method: {extraction_method}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print(f"Ultra-weak signal detection: Enabled")
        
        # Override extraction method if specified
        if extraction_method == 'adaptive':
            extract_func = self.adaptive_region_extract
        elif extraction_method == 'minimal':
            extract_func = self.minimal_qft_extract
        else:
            extract_func = self.seamless_extract_qft_chaos
        
        extracted_bits = []
        confidences = []
        frame_results = []
        
        frame_count = 0
        while frame_count < frames_to_analyze:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract bit from frame
            bit, confidence = extract_func(frame)
            extracted_bits.append(bit)
            confidences.append(confidence)
            
            if detailed and frame_count % sample_rate == 0:
                frame_results.append({
                    'frame': frame_count,
                    'bit': bit,
                    'confidence': confidence
                })
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Analyzed {frame_count}/{frames_to_analyze} frames")
        
        cap.release()
        
        # Analyze results with ultra-weak signal handling
        if extracted_bits:
            # Count occurrences
            bit_counts = Counter(extracted_bits)
            total_extractions = len(extracted_bits)
            
            # Enhanced confidence weighting for ultra-weak signals
            weighted_0 = sum((1-conf) * (1 + conf) for bit, conf in zip(extracted_bits, confidences) if bit == 0)
            weighted_1 = sum(conf * (1 + conf) for bit, conf in zip(extracted_bits, confidences) if bit == 1)
            
            if weighted_1 > weighted_0:
                final_message = 1
                success_rate = weighted_1 / (weighted_0 + weighted_1)
            else:
                final_message = 0
                success_rate = weighted_0 / (weighted_0 + weighted_1)
            
            # Calculate statistics
            avg_confidence = np.mean(confidences)
            confidence_std = np.std(confidences)
            high_confidence_frames = sum(1 for c in confidences if c > self.confidence_threshold)
            high_confidence_rate = high_confidence_frames / total_extractions
            
            print(f"\n=== DECODING RESULTS (Ultra-Imperceptible Mode) ===")
            print(f"Extracted message: {final_message}")
            print(f"Enhanced success rate: {success_rate:.2%}")
            print(f"Average confidence: {avg_confidence:.4f}")
            print(f"Confidence std: {confidence_std:.4f}")
            print(f"Detection rate: {high_confidence_rate:.2%}")
            print(f"Bit distribution: 0={bit_counts[0]}, 1={bit_counts[1]}")
            
            if detailed:
                print(f"\n=== DETAILED ANALYSIS ===")
                for result in frame_results:
                    print(f"Frame {result['frame']:4d}: bit={result['bit']}, confidence={result['confidence']:.4f}")
            
            return {
                'message': final_message,
                'success_rate': success_rate,
                'confidence': avg_confidence,
                'confidence_std': confidence_std,
                'high_confidence_rate': high_confidence_rate,
                'bit_distribution': bit_counts,
                'frame_results': frame_results if detailed else None,
                'total_frames': frame_count
            }
        else:
            print("No bits could be extracted!")
            return None

def main():
    parser = argparse.ArgumentParser(description='QFT-Chaos Video Watermark Decoder (Ultra-Imperceptible)')
    parser.add_argument('input_video', help='Path to watermarked video file')
    parser.add_argument('--chaos', type=float, default=3.8, help='Chaos parameter (must match encoder)')
    parser.add_argument('--frames', type=int, help='Maximum number of frames to analyze')
    parser.add_argument('--detailed', action='store_true', help='Perform detailed frame-by-frame analysis')
    parser.add_argument('--sample-rate', type=int, default=10, help='Sample rate for detailed analysis')
    parser.add_argument('--method', choices=['seamless', 'adaptive', 'minimal'], default='seamless',
                       help='Extraction method (default: seamless)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input_video):
        print(f"Error: Input video file '{args.input_video}' does not exist")
        return
    
    # Create decoder
    decoder = QFTChaosVideoWatermarkDecoder(
        chaos_param=args.chaos,
        arnold_iterations=0,  # Disabled for imperceptibility
        multi_scale=False     # Disabled for imperceptibility
    )
    
    # Decode the video
    try:
        result = decoder.decode_video(
            args.input_video,
            max_frames=args.frames,
            detailed=args.detailed,
            sample_rate=args.sample_rate,
            extraction_method=args.method
        )
        
        if result:
            print(f"\nExtracted message: {result['message']}")
            print(f"Confidence: {result['confidence']:.4f}")
        else:
            print("Decoding failed!")
            
    except Exception as e:
        print(f"Error during decoding: {str(e)}")
        return

if __name__ == "__main__":
    main() 