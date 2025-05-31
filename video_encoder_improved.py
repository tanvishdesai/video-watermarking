import cv2
import numpy as np
import argparse
from scipy.fftpack import dct, idct
import os

class ImprovedVideoWatermarkEncoder:
    def __init__(self, alpha=0.3, adaptive_strength=True):
        """
        Initialize the improved video watermark encoder
        
        Args:
            alpha (float): Base watermark strength parameter
            adaptive_strength (bool): Whether to adapt strength based on block variance
        """
        self.alpha = alpha
        self.adaptive_strength = adaptive_strength
        
        # Multiple frequency positions for better robustness
        # Original:
        # self.freq_positions = [
        #     ((2, 3), (3, 2)),  # Low-mid frequency
        #     ((3, 4), (4, 3)),  # Mid frequency  
        #     ((2, 4), (4, 2)),  # Alternative mid frequency
        #     ((1, 3), (3, 1)),  # Lower frequency
        # ]
        # New: Targeting slightly lower frequencies
        self.freq_positions = [
            ((1, 2), (2, 1)), 
            ((1, 3), (3, 1)),  
            ((2, 2), (1, 1)) 
        ]
        
    def dct2(self, block):
        """2D DCT transform"""
        return dct(dct(block.T, norm='ortho').T, norm='ortho')
    
    def idct2(self, block):
        """2D inverse DCT transform"""
        return idct(idct(block.T, norm='ortho').T, norm='ortho')
    
    def calculate_adaptive_strength(self, block):
        """Calculate adaptive strength based on block characteristics"""
        if not self.adaptive_strength:
            return self.alpha
        
        # Calculate block variance as a measure of texture
        variance = np.var(block.astype(np.float32))
        
        # Higher variance (more texture) = can handle stronger watermark
        # Lower variance (smooth areas) = need weaker watermark
        if variance > 100:
            return self.alpha * 1.5  # Stronger in textured areas
        elif variance > 50:
            return self.alpha
        else:
            return self.alpha * 0.7  # Weaker in smooth areas
    
    def embed_bit_in_block(self, block, bit):
        """
        Embed a bit into an 8x8 block using improved DCT domain watermarking
        
        Args:
            block: 8x8 image block
            bit: Binary bit to embed (0 or 1)
        
        Returns:
            Watermarked block
        """
        # Convert to float for DCT processing
        block = block.astype(np.float32)
        
        # Calculate adaptive strength
        strength = self.calculate_adaptive_strength(block)
        
        # Apply DCT
        dct_block = self.dct2(block)
        
        # Embed in multiple frequency positions for redundancy
        for pos1, pos2 in self.freq_positions:
            coeff1, coeff2 = dct_block[pos1], dct_block[pos2]
            
            # Calculate base magnitude for modification
            base_magnitude = max(abs(coeff1), abs(coeff2), 10.0)
            modification = strength * base_magnitude
            
            # Force a significant and consistent difference between coefficients
            if bit == 1:
                # Make coefficient at pos1 significantly larger than pos2
                dct_block[pos1] = abs(coeff1) + modification
                dct_block[pos2] = -abs(coeff2) - modification/2
            else:
                # Make coefficient at pos2 significantly larger than pos1  
                dct_block[pos2] = abs(coeff2) + modification
                dct_block[pos1] = -abs(coeff1) - modification/2
        
        # Apply inverse DCT
        watermarked_block = self.idct2(dct_block)
        
        # Clip values to valid range
        watermarked_block = np.clip(watermarked_block, 0, 255)
        
        return watermarked_block.astype(np.uint8)
    
    def encode_message_in_frame(self, frame, message_bit):
        """
        Encode a single bit into multiple blocks of a frame with enhanced redundancy
        
        Args:
            frame: Input frame
            message_bit: Binary bit to embed
        
        Returns:
            Watermarked frame
        """
        watermarked_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Work with luminance channel (convert to YUV if color)
        if len(frame.shape) == 3:
            yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            y_channel = yuv_frame[:, :, 0]
        else:
            y_channel = frame
        
        # Increased redundancy: embed in more blocks across the frame
        embed_positions = []
        
        # Create a grid of embedding positions
        step_size = 24  # Distance between embedding points
        for row in range(8, height - 8, step_size):
            for col in range(8, width - 8, step_size):
                if row + 8 <= height and col + 8 <= width:
                    embed_positions.append((row, col))
        
        # Limit to reasonable number to avoid over-processing
        if len(embed_positions) > 20:
            # Select distributed positions
            indices = np.linspace(0, len(embed_positions)-1, 20, dtype=int)
            embed_positions = [embed_positions[i] for i in indices]
        
        for row, col in embed_positions:
            block = y_channel[row:row+8, col:col+8]
            watermarked_block = self.embed_bit_in_block(block, message_bit)
            y_channel[row:row+8, col:col+8] = watermarked_block
        
        # Convert back to BGR if original was color
        if len(frame.shape) == 3:
            yuv_frame[:, :, 0] = y_channel
            watermarked_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR)
        else:
            watermarked_frame = y_channel
        
        return watermarked_frame
    
    def encode_video(self, input_path, output_path, message_bit):
        """
        Encode a binary message into a video with improved robustness
        
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
        
        print(f"Processing video: {total_frames} frames, {fps} FPS, {width}x{height}")
        print(f"Embedding message bit: {message_bit}")
        print(f"Using adaptive strength: {self.adaptive_strength}")
        print(f"Base alpha: {self.alpha}")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Embed the message bit in the frame
            watermarked_frame = self.encode_message_in_frame(frame, message_bit)
            
            # Write the watermarked frame
            out.write(watermarked_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        # Release everything
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"Watermarked video saved to: {output_path}")
        print(f"Original size: {os.path.getsize(input_path) / 1024 / 1024:.2f} MB")
        print(f"Watermarked size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

def main():
    parser = argparse.ArgumentParser(description='Improved Video Watermark Encoder')
    parser.add_argument('input_video', help='Path to input video file')
    parser.add_argument('output_video', help='Path to output watermarked video file')
    parser.add_argument('message', type=int, choices=[0, 1], 
                        help='Binary message to embed (0 or 1)')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='Base watermark strength (default: 0.3)')
    parser.add_argument('--no-adaptive', action='store_true',
                        help='Disable adaptive strength adjustment')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_video):
        print(f"Error: Input video file '{args.input_video}' not found!")
        return
    
    # Create encoder
    encoder = ImprovedVideoWatermarkEncoder(
        alpha=args.alpha, 
        adaptive_strength=not args.no_adaptive
    )
    
    # Encode the video
    encoder.encode_video(args.input_video, args.output_video, args.message)

if __name__ == "__main__":
    main() 