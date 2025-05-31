import cv2
import numpy as np
import argparse
import pywt
import os
from scipy.stats import entropy

class DWTSVDVideoWatermarkEncoder:
    def __init__(self, alpha=0.05, wavelet='haar', svd_blocks=True, spread_spectrum=True):
        """
        Initialize the DWT-SVD video watermark encoder
        
        Args:
            alpha (float): Base watermark strength parameter
            wavelet (str): Wavelet type for DWT
            svd_blocks (bool): Whether to use SVD on blocks
            spread_spectrum (bool): Whether to use spread spectrum technique
        """
        self.alpha = alpha
        self.wavelet = wavelet
        self.svd_blocks = svd_blocks
        self.spread_spectrum = spread_spectrum
        
        # Generate pseudo-random sequence for spread spectrum
        if self.spread_spectrum:
            np.random.seed(42)  # Fixed seed for reproducibility
            self.pn_sequence = np.random.randint(0, 2, 1024) * 2 - 1  # {-1, 1}
        
        # Multi-level decomposition for better frequency separation
        self.decomp_levels = 3
        
        # Content adaptation parameters
        self.texture_threshold = 0.15
        self.edge_threshold = 50
        
    def calculate_texture_strength(self, block):
        """Calculate adaptive strength based on texture complexity"""
        # Use local standard deviation as texture measure
        std_dev = np.std(block.astype(np.float32))
        
        # Normalize to [0, 1] range
        texture_measure = min(std_dev / 50.0, 1.0)
        
        # Higher texture = can handle stronger watermark
        if texture_measure > self.texture_threshold:
            return self.alpha * (1.0 + texture_measure)
        else:
            return self.alpha * 0.7  # Weaker in smooth areas
    
    def detect_edges(self, frame):
        """Detect edges for content-adaptive embedding"""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Use Canny edge detection
        edges = cv2.Canny(gray, self.edge_threshold, self.edge_threshold * 2)
        return edges
    
    def embed_bit_dwt_svd(self, block, bit, strength):
        """
        Embed a bit using DWT-SVD approach
        
        Args:
            block: Image block
            bit: Binary bit to embed (0 or 1)
            strength: Embedding strength
        
        Returns:
            Watermarked block
        """
        # Convert to float
        block_float = block.astype(np.float32)
        
        # Apply DWT
        coeffs = pywt.dwt2(block_float, self.wavelet)
        cA, (cH, cV, cD) = coeffs
        
        if self.svd_blocks:
            # Apply SVD to the approximation coefficients
            U, S, Vt = np.linalg.svd(cA, full_matrices=False)
            
            # Modify singular values based on the bit
            if len(S) >= 2:
                if self.spread_spectrum:
                    # Use spread spectrum with pseudo-random sequence
                    pn_bit = self.pn_sequence[hash(str(block.flatten()[:4])) % len(self.pn_sequence)]
                    modification = strength * pn_bit * (1 if bit == 1 else -1)
                else:
                    modification = strength * (1 if bit == 1 else -1)
                
                # Embed in the ratio of first two singular values
                S[0] = S[0] + modification * S[0]
                S[1] = S[1] - modification * S[1] * 0.5
            
            # Reconstruct the approximation coefficients
            cA_modified = np.dot(U, np.dot(np.diag(S), Vt))
        else:
            # Direct modification of approximation coefficients
            if self.spread_spectrum:
                pn_bit = self.pn_sequence[hash(str(block.flatten()[:4])) % len(self.pn_sequence)]
                modification = strength * pn_bit * (1 if bit == 1 else -1)
            else:
                modification = strength * (1 if bit == 1 else -1)
            
            cA_modified = cA + modification
        
        # Also embed in detail coefficients for redundancy
        cH_modified = cH + modification * 0.3
        cV_modified = cV + modification * 0.3
        cD_modified = cD + modification * 0.2
        
        # Reconstruct the block
        coeffs_modified = (cA_modified, (cH_modified, cV_modified, cD_modified))
        watermarked_block = pywt.idwt2(coeffs_modified, self.wavelet)
        
        # Clip to valid range
        watermarked_block = np.clip(watermarked_block, 0, 255)
        
        return watermarked_block.astype(np.uint8)
    
    def embed_bit_in_frame(self, frame, message_bit):
        """
        Embed a bit into a frame using content-adaptive DWT-SVD
        
        Args:
            frame: Input frame
            message_bit: Binary bit to embed
        
        Returns:
            Watermarked frame
        """
        watermarked_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Work with luminance channel
        if len(frame.shape) == 3:
            yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            y_channel = yuv_frame[:, :, 0].astype(np.float32)
        else:
            y_channel = frame.astype(np.float32)
        
        # Detect edges for content-adaptive embedding
        edges = self.detect_edges(frame)
        
        # Define embedding positions with content adaptation
        embed_positions = []
        block_size = 16  # Larger blocks for DWT
        
        # Create grid of potential embedding positions
        for row in range(0, height - block_size, block_size):
            for col in range(0, width - block_size, block_size):
                if row + block_size <= height and col + block_size <= width:
                    # Check edge density in this region
                    edge_region = edges[row:row+block_size, col:col+block_size]
                    edge_density = np.sum(edge_region) / (block_size * block_size * 255)
                    
                    # Prefer regions with moderate edge density (textured but not too busy)
                    if 0.1 <= edge_density <= 0.6:
                        embed_positions.append((row, col, edge_density))
        
        # Sort by edge density and select best positions
        embed_positions.sort(key=lambda x: abs(x[2] - 0.3))  # Prefer moderate texture
        
        # Limit number of embedding positions
        max_positions = min(20, len(embed_positions))
        selected_positions = embed_positions[:max_positions]
        
        # Embed in selected positions
        for row, col, edge_density in selected_positions:
            block = y_channel[row:row+block_size, col:col+block_size]
            
            # Calculate adaptive strength
            strength = self.calculate_texture_strength(block)
            
            # Embed the bit
            watermarked_block = self.embed_bit_dwt_svd(block, message_bit, strength)
            y_channel[row:row+block_size, col:col+block_size] = watermarked_block
        
        # Convert back to original format
        if len(frame.shape) == 3:
            yuv_frame[:, :, 0] = np.clip(y_channel, 0, 255).astype(np.uint8)
            watermarked_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR)
        else:
            watermarked_frame = np.clip(y_channel, 0, 255).astype(np.uint8)
        
        return watermarked_frame
    
    def encode_video(self, input_path, output_path, message_bit):
        """
        Encode a binary message into a video using DWT-SVD watermarking
        
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
        print(f"Using DWT-SVD method with {self.wavelet} wavelet")
        print(f"SVD blocks: {self.svd_blocks}, Spread spectrum: {self.spread_spectrum}")
        print(f"Base alpha: {self.alpha}")
        
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
        cv2.destroyAllWindows()
        
        print(f"Watermarked video saved to: {output_path}")
        print(f"Original size: {os.path.getsize(input_path) / 1024 / 1024:.2f} MB")
        print(f"Watermarked size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

def main():
    parser = argparse.ArgumentParser(description='DWT-SVD Video Watermark Encoder')
    parser.add_argument('input_video', help='Path to input video file')
    parser.add_argument('output_video', help='Path to output watermarked video file')
    parser.add_argument('message', type=int, choices=[0, 1], 
                        help='Binary message to embed (0 or 1)')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Base watermark strength (default: 0.05)')
    parser.add_argument('--wavelet', type=str, default='haar',
                        choices=['haar', 'db4', 'db8', 'bior2.2'],
                        help='Wavelet type (default: haar)')
    parser.add_argument('--no-svd', action='store_true',
                        help='Disable SVD enhancement')
    parser.add_argument('--no-spread', action='store_true',
                        help='Disable spread spectrum')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_video):
        print(f"Error: Input video file '{args.input_video}' not found!")
        return
    
    # Create encoder
    encoder = DWTSVDVideoWatermarkEncoder(
        alpha=args.alpha,
        wavelet=args.wavelet,
        svd_blocks=not args.no_svd,
        spread_spectrum=not args.no_spread
    )
    
    # Encode the video
    encoder.encode_video(args.input_video, args.output_video, args.message)

if __name__ == "__main__":
    main() 