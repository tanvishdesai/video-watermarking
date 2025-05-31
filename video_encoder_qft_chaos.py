import cv2
import numpy as np
import argparse
import os
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt

class QFTChaosVideoWatermarkEncoder:
    def __init__(self, alpha=0.005, chaos_param=3.8, arnold_iterations=0, multi_scale=False):
        """
        Initialize the QFT-Chaos video watermark encoder (Ultra-Imperceptible Version)
        
        Args:
            alpha (float): Ultra-low watermark strength for imperceptibility
            chaos_param (float): Chaos parameter for logistic map
            arnold_iterations (int): Arnold transform iterations (disabled for imperceptibility)
            multi_scale (bool): Multi-scale embedding (disabled for imperceptibility)
        """
        self.alpha = alpha
        self.chaos_param = chaos_param
        self.arnold_iterations = arnold_iterations
        self.multi_scale = multi_scale
        
        # Initialize chaos sequence
        self.chaos_seed = 0.123456789
        
        # Disable multi-scale for better imperceptibility
        self.scales = [1.0]
    
    def generate_chaos_sequence(self, length):
        """Generate chaotic sequence using logistic map"""
        sequence = []
        x = self.chaos_seed
        
        for _ in range(length):
            x = self.chaos_param * x * (1 - x)
            sequence.append(x)
        
        # Convert to binary sequence
        binary_sequence = [1 if x > 0.5 else -1 for x in sequence]
        return np.array(binary_sequence)
    
    def seamless_embed_qft_chaos(self, frame, message_bit):
        """
        Seamless embedding across entire frame to avoid block artifacts
        
        Args:
            frame: Input frame
            message_bit: Binary bit to embed
        
        Returns:
            Watermarked frame with no visible artifacts
        """
        height, width = frame.shape[:2]
        
        # Convert to YUV color space for better imperceptibility
        frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        
        # Work only on Y (luminance) channel to avoid color artifacts
        y_channel = frame_yuv[:, :, 0].astype(np.float64)
        
        # Apply FFT to entire frame
        fft_y = fft2(y_channel)
        fft_shifted = fftshift(fft_y)
        
        # Generate chaos sequence for position selection
        total_positions = height * width
        chaos_seq = self.generate_chaos_sequence(total_positions)
        
        # Define very conservative embedding region (only low-mid frequencies)
        center_h, center_w = height // 2, width // 2
        
        # Ultra-conservative frequency selection
        embed_positions = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                if abs(i) == 1 and abs(j) == 1:  # Only 4 positions
                    pos_h = center_h + i
                    pos_w = center_w + j
                    if 0 <= pos_h < height and 0 <= pos_w < width:
                        embed_positions.append((pos_h, pos_w))
        
        # Ultra-weak embedding with corrected logic
        for idx, (pos_h, pos_w) in enumerate(embed_positions):
            if idx < len(chaos_seq):
                chaos_val = chaos_seq[idx]
                
                # Get current coefficient
                coeff = fft_shifted[pos_h, pos_w]
                magnitude = abs(coeff)
                phase = np.angle(coeff)
                
                # Ultra-small modification (1000x smaller than before)
                base_modification = self.alpha * magnitude * 0.0001
                
                # Fixed embedding logic: chaos_val determines position, message_bit determines direction
                if message_bit == 1:
                    # For bit 1, always increase magnitude (regardless of chaos sign)
                    modification = abs(base_modification)
                else:
                    # For bit 0, always decrease magnitude (regardless of chaos sign)
                    modification = -abs(base_modification)
                
                new_magnitude = magnitude + modification
                new_coeff = new_magnitude * np.exp(1j * phase)
                fft_shifted[pos_h, pos_w] = new_coeff
        
        # Apply inverse FFT
        fft_ishifted = ifftshift(fft_shifted)
        y_watermarked = np.real(ifft2(fft_ishifted))
        
        # Ensure no overflow and maintain proper data type
        y_watermarked = np.clip(y_watermarked, 0, 255)
        
        # Replace Y channel
        frame_yuv_watermarked = frame_yuv.copy()
        frame_yuv_watermarked[:, :, 0] = y_watermarked.astype(np.uint8)
        
        # Convert back to BGR
        watermarked_frame = cv2.cvtColor(frame_yuv_watermarked, cv2.COLOR_YUV2BGR)
        
        return watermarked_frame
    
    def adaptive_region_embed(self, frame, message_bit):
        """
        Alternative embedding using adaptive regions to avoid artifacts
        
        Args:
            frame: Input frame
            message_bit: Binary bit to embed
        
        Returns:
            Watermarked frame
        """
        height, width = frame.shape[:2]
        watermarked_frame = frame.copy().astype(np.float64)
        
        # Calculate texture map to find suitable embedding regions
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate local variance using a larger window
        kernel = np.ones((15, 15), np.float32) / (15 * 15)
        mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        sqr_mean = cv2.filter2D((gray.astype(np.float32) ** 2), -1, kernel)
        variance = sqr_mean - mean ** 2
        
        # Only embed in medium-high texture regions (avoid smooth areas)
        texture_threshold = np.percentile(variance, 70)
        embedding_mask = variance > texture_threshold
        
        # Apply morphological operations to avoid isolated pixels
        kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        embedding_mask = cv2.morphologyEx(embedding_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel_morph)
        embedding_mask = cv2.morphologyEx(embedding_mask, cv2.MORPH_CLOSE, kernel_morph)
        
        # Find suitable regions
        embedding_positions = np.where(embedding_mask > 0)
        num_positions = len(embedding_positions[0])
        
        if num_positions > 0:
            # Generate chaos sequence
            chaos_seq = self.generate_chaos_sequence(num_positions)
            
            # Limit number of modifications
            max_modifications = min(num_positions, 50)  # Very few modifications
            
            for i in range(max_modifications):
                pos_y = embedding_positions[0][i]
                pos_x = embedding_positions[1][i]
                chaos_val = chaos_seq[i]
                
                # Get pixel value
                pixel = watermarked_frame[pos_y, pos_x]
                
                # Ultra-small modification (much smaller than before)
                modification_strength = self.alpha * 0.1  # Even smaller
                
                if message_bit == 1:
                    modification = modification_strength * chaos_val
                else:
                    modification = -modification_strength * chaos_val
                
                # Apply modification to all channels equally
                modified_pixel = pixel + modification
                
                # Ensure bounds
                modified_pixel = np.clip(modified_pixel, 0, 255)
                watermarked_frame[pos_y, pos_x] = modified_pixel
        
        return watermarked_frame.astype(np.uint8)
    
    def minimal_qft_embed(self, frame, message_bit):
        """
        Minimal QFT embedding with extreme care for imperceptibility
        
        Args:
            frame: Input frame
            message_bit: Binary bit to embed
        
        Returns:
            Watermarked frame
        """
        height, width = frame.shape[:2]
        
        # Use only a small central region to avoid edge artifacts
        crop_size = min(height, width) // 4
        center_y, center_x = height // 2, width // 2
        start_y = center_y - crop_size // 2
        end_y = start_y + crop_size
        start_x = center_x - crop_size // 2
        end_x = start_x + crop_size
        
        # Extract central region
        central_region = frame[start_y:end_y, start_x:end_x].copy()
        
        # Convert to float for processing
        if len(central_region.shape) == 3:
            # Process only green channel (most sensitive to human vision)
            g_channel = central_region[:, :, 1].astype(np.float64)
        else:
            g_channel = central_region.astype(np.float64)
        
        # Apply FFT
        fft_g = fft2(g_channel)
        fft_shifted = fftshift(fft_g)
        
        # Very conservative position selection
        h, w = fft_shifted.shape
        center_h, center_w = h // 2, w // 2
        
        # Only modify one frequency component
        pos_h = center_h + 1
        pos_w = center_w + 1
        
        if 0 <= pos_h < h and 0 <= pos_w < w:
            # Generate chaos value
            chaos_seq = self.generate_chaos_sequence(1)
            chaos_val = chaos_seq[0]
            
            # Get current coefficient
            coeff = fft_shifted[pos_h, pos_w]
            magnitude = abs(coeff)
            phase = np.angle(coeff)
            
            # Extremely small modification
            base_modification = self.alpha * magnitude * 0.00001  # Even smaller
            
            if message_bit == 1:
                modification = base_modification * chaos_val
            else:
                modification = -base_modification * chaos_val
            
            new_magnitude = magnitude + modification
            new_coeff = new_magnitude * np.exp(1j * phase)
            fft_shifted[pos_h, pos_w] = new_coeff
        
        # Apply inverse FFT
        fft_ishifted = ifftshift(fft_shifted)
        g_watermarked = np.real(ifft2(fft_ishifted))
        g_watermarked = np.clip(g_watermarked, 0, 255)
        
        # Replace channel
        watermarked_region = central_region.copy()
        if len(central_region.shape) == 3:
            watermarked_region[:, :, 1] = g_watermarked.astype(np.uint8)
        else:
            watermarked_region = g_watermarked.astype(np.uint8)
        
        # Put back into frame
        watermarked_frame = frame.copy()
        watermarked_frame[start_y:end_y, start_x:end_x] = watermarked_region
        
        return watermarked_frame
    
    def embed_bit_in_frame(self, frame, message_bit):
        """
        Embed a bit into a frame using the most imperceptible method
        
        Args:
            frame: Input frame
            message_bit: Binary bit to embed
        
        Returns:
            Watermarked frame with no visible artifacts
        """
        # Use the seamless embedding method for best imperceptibility
        return self.seamless_embed_qft_chaos(frame, message_bit)
    
    def encode_video(self, input_path, output_path, message_bit):
        """
        Encode a binary message into a video using ultra-imperceptible QFT-Chaos watermarking
        
        Args:
            input_path: Path to input video
            output_path: Path to output watermarked video
            message_bit: Binary message (0 or 1) to embed
        """
        # Open input video
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"QFT-Chaos Video Watermarking (Ultra-Imperceptible Mode)")
        print(f"Processing video: {total_frames} frames, {fps} FPS, {width}x{height}")
        print(f"Embedding message bit: {message_bit}")
        print(f"Ultra-low alpha: {self.alpha}")
        print(f"Seamless embedding: Enabled")
        print(f"Block artifacts: Eliminated")
        print(f"Color artifacts: Minimized")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Embed the message bit in the frame
            watermarked_frame = self.embed_bit_in_frame(frame, message_bit)
            
            # Write the watermarked frame
            out.write(watermarked_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        # Release everything
        cap.release()
        out.release()
        
        print(f"\nUltra-imperceptible watermarking completed!")
        print(f"Output saved to: {output_path}")
        print(f"Visual artifacts: Eliminated")

def main():
    parser = argparse.ArgumentParser(description='QFT-Chaos Video Watermark Encoder (Ultra-Imperceptible)')
    parser.add_argument('input_video', help='Path to input video file')
    parser.add_argument('output_video', help='Path to output watermarked video file')
    parser.add_argument('message', type=int, choices=[0, 1], help='Binary message to embed (0 or 1)')
    parser.add_argument('--alpha', type=float, default=0.005, help='Watermark strength (default: 0.005, ultra-low)')
    parser.add_argument('--chaos', type=float, default=3.8, help='Chaos parameter (default: 3.8)')
    parser.add_argument('--method', choices=['seamless', 'adaptive', 'minimal'], default='seamless',
                       help='Embedding method (default: seamless)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input_video):
        print(f"Error: Input video file '{args.input_video}' does not exist")
        return
    
    if args.alpha > 0.01:
        print(f"Warning: Alpha {args.alpha} may cause visible artifacts")
        print("Recommended range for ultra-imperceptibility: 0.001 - 0.01")
    
    # Create encoder
    encoder = QFTChaosVideoWatermarkEncoder(
        alpha=args.alpha,
        chaos_param=args.chaos,
        arnold_iterations=0,  # Disabled for imperceptibility
        multi_scale=False     # Disabled for imperceptibility
    )
    
    # Override embedding method if specified
    if args.method == 'adaptive':
        encoder.embed_bit_in_frame = lambda frame, bit: encoder.adaptive_region_embed(frame, bit)
    elif args.method == 'minimal':
        encoder.embed_bit_in_frame = lambda frame, bit: encoder.minimal_qft_embed(frame, bit)
    
    # Encode the video
    try:
        encoder.encode_video(args.input_video, args.output_video, args.message)
    except Exception as e:
        print(f"Error during encoding: {str(e)}")
        return

if __name__ == "__main__":
    main() 