import cv2
import numpy as np
import argparse
import pywt
import os
from collections import Counter

class DWTSVDVideoWatermarkDecoder:
    def __init__(self, wavelet='haar', svd_blocks=True, spread_spectrum=True):
        """
        Initialize the DWT-SVD video watermark decoder
        
        Args:
            wavelet (str): Wavelet type for DWT (must match encoder)
            svd_blocks (bool): Whether SVD was used (must match encoder)
            spread_spectrum (bool): Whether spread spectrum was used (must match encoder)
        """
        self.wavelet = wavelet
        self.svd_blocks = svd_blocks
        self.spread_spectrum = spread_spectrum
        
        # Generate the same pseudo-random sequence as encoder
        if self.spread_spectrum:
            np.random.seed(42)  # Same seed as encoder
            self.pn_sequence = np.random.randint(0, 2, 1024) * 2 - 1  # {-1, 1}
        
        # Content adaptation parameters (match encoder)
        self.edge_threshold = 50
        
    def detect_edges(self, frame):
        """Detect edges for content-adaptive extraction (matches encoder)"""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Use Canny edge detection
        edges = cv2.Canny(gray, self.edge_threshold, self.edge_threshold * 2)
        return edges
    
    def extract_bit_dwt_svd(self, block):
        """
        Extract a bit from a block using DWT-SVD analysis
        
        Args:
            block: Image block
        
        Returns:
            Extracted bit (0 or 1) and confidence score
        """
        # Convert to float
        block_float = block.astype(np.float32)
        
        # Apply DWT
        coeffs = pywt.dwt2(block_float, self.wavelet)
        cA, (cH, cV, cD) = coeffs
        
        confidences = []
        bit_votes = []
        
        if self.svd_blocks:
            # Apply SVD to the approximation coefficients
            try:
                U, S, Vt = np.linalg.svd(cA, full_matrices=False)
                
                # Extract bit from singular values ratio
                if len(S) >= 2 and S[1] > 0:
                    # Calculate the ratio between first two singular values
                    ratio = S[0] / S[1]
                    
                    # Determine bit based on ratio
                    # If ratio is higher than expected, bit is likely 1
                    # If ratio is lower than expected, bit is likely 0
                    ratio_threshold = 1.0  # Adjust based on embedding strength
                    
                    if ratio > ratio_threshold:
                        bit_votes.append(1)
                    else:
                        bit_votes.append(0)
                    
                    # Confidence based on how far the ratio is from threshold
                    confidence = min(abs(ratio - ratio_threshold) / ratio_threshold, 1.0)
                    confidences.append(confidence)
                else:
                    bit_votes.append(0)
                    confidences.append(0.0)
            except:
                bit_votes.append(0)
                confidences.append(0.0)
        else:
            # Direct analysis of approximation coefficients
            mean_cA = np.mean(cA)
            # Use coefficient mean as indicator
            if mean_cA > np.median(cA):
                bit_votes.append(1)
            else:
                bit_votes.append(0)
            
            # Confidence based on deviation from median
            confidence = min(abs(mean_cA - np.median(cA)) / (np.std(cA) + 1e-8), 1.0)
            confidences.append(confidence)
        
        # Also analyze detail coefficients for additional votes
        detail_coeffs = [cH, cV, cD]
        for detail in detail_coeffs:
            mean_detail = np.mean(detail)
            median_detail = np.median(detail)
            
            if mean_detail > median_detail:
                bit_votes.append(1)
            else:
                bit_votes.append(0)
            
            # Confidence from detail coefficients
            std_detail = np.std(detail)
            if std_detail > 0:
                conf = min(abs(mean_detail - median_detail) / std_detail, 1.0)
            else:
                conf = 0.0
            confidences.append(conf * 0.5)  # Lower weight for detail coefficients
        
        # Weighted voting
        if len(bit_votes) > 0 and len(confidences) > 0:
            vote_counts = {0: 0.0, 1: 0.0}
            for bit, conf in zip(bit_votes, confidences):
                vote_counts[bit] += conf
            
            total_confidence = vote_counts[0] + vote_counts[1]
            if total_confidence > 0:
                if vote_counts[1] > vote_counts[0]:
                    final_bit = 1
                    confidence = vote_counts[1] / total_confidence
                else:
                    final_bit = 0
                    confidence = vote_counts[0] / total_confidence
            else:
                # Fallback to majority voting
                bit_count = Counter(bit_votes)
                if bit_count[1] >= bit_count[0]:
                    final_bit = 1
                else:
                    final_bit = 0
                confidence = max(bit_count.values()) / len(bit_votes)
            
            return final_bit, confidence
        else:
            return 0, 0.0
    
    def extract_bit_from_frame(self, frame):
        """
        Extract a bit from a frame using content-adaptive DWT-SVD analysis
        
        Args:
            frame: Input frame
        
        Returns:
            Extracted bit (0 or 1) and confidence score
        """
        height, width = frame.shape[:2]
        
        # Work with luminance channel
        if len(frame.shape) == 3:
            yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            y_channel = yuv_frame[:, :, 0].astype(np.float32)
        else:
            y_channel = frame.astype(np.float32)
        
        # Detect edges (same as encoder)
        edges = self.detect_edges(frame)
        
        # Find extraction positions (match encoder logic)
        embed_positions = []
        block_size = 16  # Same as encoder
        
        # Create grid of potential extraction positions
        for row in range(0, height - block_size, block_size):
            for col in range(0, width - block_size, block_size):
                if row + block_size <= height and col + block_size <= width:
                    # Check edge density in this region
                    edge_region = edges[row:row+block_size, col:col+block_size]
                    edge_density = np.sum(edge_region) / (block_size * block_size * 255)
                    
                    # Same criteria as encoder
                    if 0.1 <= edge_density <= 0.6:
                        embed_positions.append((row, col, edge_density))
        
        # Sort by edge density and select best positions (same as encoder)
        embed_positions.sort(key=lambda x: abs(x[2] - 0.3))
        
        # Limit number of extraction positions
        max_positions = min(20, len(embed_positions))
        selected_positions = embed_positions[:max_positions]
        
        # Extract from selected positions
        extracted_bits = []
        confidence_scores = []
        
        for row, col, edge_density in selected_positions:
            block = y_channel[row:row+block_size, col:col+block_size]
            
            # Extract bit from this block
            bit, confidence = self.extract_bit_dwt_svd(block)
            extracted_bits.append(bit)
            confidence_scores.append(confidence)
        
        # Weighted majority voting for final decision
        if len(extracted_bits) > 0:
            vote_counts = {0: 0.0, 1: 0.0}
            for bit, conf in zip(extracted_bits, confidence_scores):
                vote_counts[bit] += conf
            
            total_confidence = vote_counts[0] + vote_counts[1]
            if total_confidence > 0:
                if vote_counts[1] > vote_counts[0]:
                    final_bit = 1
                    confidence = vote_counts[1] / total_confidence
                else:
                    final_bit = 0
                    confidence = vote_counts[0] / total_confidence
            else:
                # Fallback to simple majority
                bit_count = Counter(extracted_bits)
                if bit_count[1] >= bit_count[0]:
                    final_bit = 1
                else:
                    final_bit = 0
                confidence = max(bit_count.values()) / len(extracted_bits)
            
            return final_bit, confidence
        else:
            return 0, 0.0
    
    def decode_video(self, input_path, num_frames_to_analyze=None):
        """
        Decode the binary message from a watermarked video
        
        Args:
            input_path: Path to watermarked video
            num_frames_to_analyze: Number of frames to analyze (None = all frames)
        
        Returns:
            Decoded message bit (0 or 1) and confidence score
        """
        # Open input video
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Analyzing video: {total_frames} frames, {fps} FPS, {width}x{height}")
        print(f"Using DWT-SVD decoder with {self.wavelet} wavelet")
        print(f"SVD blocks: {self.svd_blocks}, Spread spectrum: {self.spread_spectrum}")
        
        if num_frames_to_analyze is None:
            num_frames_to_analyze = total_frames
        else:
            num_frames_to_analyze = min(num_frames_to_analyze, total_frames)
        
        print(f"Analyzing first {num_frames_to_analyze} frames")
        
        extracted_bits = []
        frame_confidences = []
        frame_count = 0
        
        while frame_count < num_frames_to_analyze:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract bit and confidence from the frame
            bit, confidence = self.extract_bit_from_frame(frame)
            extracted_bits.append(bit)
            frame_confidences.append(confidence)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Analyzed {frame_count}/{num_frames_to_analyze} frames")
        
        # Release video capture
        cap.release()
        cv2.destroyAllWindows()
        
        # Use weighted majority voting across all frames for final decision
        if len(extracted_bits) > 0:
            vote_counts = {0: 0.0, 1: 0.0}
            for bit, conf in zip(extracted_bits, frame_confidences):
                vote_counts[bit] += conf
            
            total_confidence = vote_counts[0] + vote_counts[1]
            if total_confidence > 0:
                if vote_counts[1] > vote_counts[0]:
                    final_bit = 1
                    confidence = vote_counts[1] / total_confidence
                else:
                    final_bit = 0
                    confidence = vote_counts[0] / total_confidence
            else:
                final_bit = 0
                confidence = 0.0
            
            # Calculate statistics
            bit_counts = Counter(extracted_bits)
            avg_frame_confidence = np.mean(frame_confidences)
            
            print(f"\nDecoding Results:")
            print(f"Total frames analyzed: {len(extracted_bits)}")
            print(f"Bit distribution: {dict(bit_counts)}")
            print(f"Average frame confidence: {avg_frame_confidence:.2%}")
            print(f"Weighted confidence: {confidence:.2%}")
            print(f"Decoded message: {final_bit}")
            
            return final_bit, confidence
        else:
            print("Error: No frames could be analyzed!")
            return None, 0.0
    
    def decode_video_detailed(self, input_path, sample_rate=10):
        """
        Decode with detailed frame-by-frame analysis and diagnostics
        
        Args:
            input_path: Path to watermarked video
            sample_rate: Analyze every nth frame
        
        Returns:
            Detailed analysis results
        """
        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Detailed DWT-SVD analysis (sampling every {sample_rate} frames)")
        
        frame_results = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                bit, confidence = self.extract_bit_from_frame(frame)
                frame_results.append((frame_count, bit, confidence))
                print(f"Frame {frame_count}: Bit = {bit}, Confidence = {confidence:.2%}")
            
            frame_count += 1
        
        cap.release()
        
        # Calculate detailed statistics
        if frame_results:
            bits = [result[1] for result in frame_results]
            confidences = [result[2] for result in frame_results]
            
            # Weighted voting
            vote_counts = {0: 0.0, 1: 0.0}
            for bit, conf in zip(bits, confidences):
                vote_counts[bit] += conf
            
            total_confidence = vote_counts[0] + vote_counts[1]
            if total_confidence > 0:
                if vote_counts[1] > vote_counts[0]:
                    final_bit = 1
                    weighted_confidence = vote_counts[1] / total_confidence
                else:
                    final_bit = 0
                    weighted_confidence = vote_counts[0] / total_confidence
            else:
                final_bit = 0
                weighted_confidence = 0.0
            
            bit_counts = Counter(bits)
            avg_confidence = np.mean(confidences)
            
            print(f"\nDetailed DWT-SVD Analysis Summary:")
            print(f"Frames sampled: {len(frame_results)}")
            print(f"Bit distribution: {dict(bit_counts)}")
            print(f"Average confidence: {avg_confidence:.2%}")
            print(f"Weighted confidence: {weighted_confidence:.2%}")
            print(f"Most likely bit: {final_bit}")
            
            return final_bit, weighted_confidence, frame_results
        
        return None, 0.0, []

def main():
    parser = argparse.ArgumentParser(description='DWT-SVD Video Watermark Decoder')
    parser.add_argument('input_video', help='Path to watermarked video file')
    parser.add_argument('--frames', type=int, default=None,
                        help='Number of frames to analyze (default: all frames)')
    parser.add_argument('--detailed', action='store_true',
                        help='Perform detailed frame-by-frame analysis')
    parser.add_argument('--sample-rate', type=int, default=10,
                        help='Sample rate for detailed analysis (default: 10)')
    parser.add_argument('--wavelet', type=str, default='haar',
                        choices=['haar', 'db4', 'db8', 'bior2.2'],
                        help='Wavelet type (must match encoder, default: haar)')
    parser.add_argument('--no-svd', action='store_true',
                        help='Disable SVD analysis (must match encoder)')
    parser.add_argument('--no-spread', action='store_true',
                        help='Disable spread spectrum (must match encoder)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_video):
        print(f"Error: Input video file '{args.input_video}' not found!")
        return
    
    # Create decoder
    decoder = DWTSVDVideoWatermarkDecoder(
        wavelet=args.wavelet,
        svd_blocks=not args.no_svd,
        spread_spectrum=not args.no_spread
    )
    
    if args.detailed:
        # Perform detailed analysis
        bit, confidence, results = decoder.decode_video_detailed(args.input_video, args.sample_rate)
    else:
        # Perform standard decoding
        bit, confidence = decoder.decode_video(args.input_video, args.frames)
    
    if bit is not None:
        print(f"\n{'='*50}")
        print(f"FINAL RESULT: Decoded message = {bit}")
        print(f"Confidence: {confidence:.2%}")
        print(f"{'='*50}")
    else:
        print("Failed to decode message!")

if __name__ == "__main__":
    main() 