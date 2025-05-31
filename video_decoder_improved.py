import cv2
import numpy as np
import argparse
from scipy.fftpack import dct, idct
import os
from collections import Counter

class ImprovedVideoWatermarkDecoder:
    def __init__(self):
        """Initialize the improved video watermark decoder"""
        # Match the frequency positions used in the improved encoder
        # Original:
        # self.freq_positions = [
        #     ((2, 3), (3, 2)),  # Low-mid frequency
        #     ((3, 4), (4, 3)),  # Mid frequency  
        #     ((2, 4), (4, 2)),  # Alternative mid frequency
        #     ((1, 3), (3, 1)),  # Lower frequency
        # ]
        # New: Matching encoder's lower frequency targets
        self.freq_positions = [
            ((1, 2), (2, 1)), 
            ((1, 3), (3, 1)),  
            ((2, 2), (1, 1)) 
        ]
        
    def dct2(self, block):
        """2D DCT transform"""
        return dct(dct(block.T, norm='ortho').T, norm='ortho')
    
    def extract_bit_from_block(self, block):
        """
        Extract a bit from an 8x8 block using improved DCT domain analysis
        
        Args:
            block: 8x8 image block
        
        Returns:
            Extracted bit (0 or 1) and confidence score
        """
        # Convert to float for DCT processing
        block = block.astype(np.float32)
        
        # Apply DCT
        dct_block = self.dct2(block)
        
        # Analyze all frequency positions for robustness
        bit_votes = []
        confidence_scores = []
        
        for pos1, pos2 in self.freq_positions:
            coeff1, coeff2 = dct_block[pos1], dct_block[pos2]
            
            # Calculate the difference between coefficients
            diff = coeff1 - coeff2
            
            # Determine bit based on which coefficient is larger
            if diff > 0:
                bit_votes.append(1)
            else:
                bit_votes.append(0)
            
            # Confidence is based on the magnitude of the difference
            # Larger differences indicate stronger watermarks
            magnitude_sum = abs(coeff1) + abs(coeff2)
            if magnitude_sum > 0:
                confidence = min(abs(diff) / magnitude_sum, 1.0)
            else:
                confidence = 0.0
            confidence_scores.append(confidence)
        
        # Use majority voting with weighted confidence
        if len(bit_votes) > 0:
            # Calculate weighted votes
            vote_counts = {0: 0.0, 1: 0.0}
            for bit, conf in zip(bit_votes, confidence_scores):
                vote_counts[bit] += conf
            
            # Return the bit with higher weighted score
            total_confidence = vote_counts[0] + vote_counts[1]
            if total_confidence > 0:
                if vote_counts[1] > vote_counts[0]:
                    final_bit = 1
                    confidence = vote_counts[1] / total_confidence
                else:
                    final_bit = 0
                    confidence = vote_counts[0] / total_confidence
            else:
                # Fallback to simple majority if no confidence
                bit_count = {0: 0, 1: 0}
                for bit in bit_votes:
                    bit_count[bit] += 1
                
                if bit_count[1] > bit_count[0]:
                    final_bit = 1
                    confidence = bit_count[1] / len(bit_votes)
                else:
                    final_bit = 0
                    confidence = bit_count[0] / len(bit_votes)
            
            return final_bit, confidence
        else:
            return 0, 0.0
    
    def extract_message_from_frame(self, frame):
        """
        Extract bits from multiple blocks of a frame with enhanced analysis
        
        Args:
            frame: Input frame
        
        Returns:
            Extracted bit (0 or 1) and confidence score
        """
        height, width = frame.shape[:2]
        
        # Work with luminance channel (convert to YUV if color)
        if len(frame.shape) == 3:
            yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            y_channel = yuv_frame[:, :, 0]
        else:
            y_channel = frame
        
        # Extract from the same grid pattern used in encoding
        embed_positions = []
        
        # Create a grid of extraction positions (matching encoder)
        step_size = 24  # Distance between embedding points
        for row in range(8, height - 8, step_size):
            for col in range(8, width - 8, step_size):
                if row + 8 <= height and col + 8 <= width:
                    embed_positions.append((row, col))
        
        # Limit to reasonable number
        if len(embed_positions) > 20:
            indices = np.linspace(0, len(embed_positions)-1, 20, dtype=int)
            embed_positions = [embed_positions[i] for i in indices]
        
        extracted_bits = []
        confidence_scores = []
        
        for row, col in embed_positions:
            block = y_channel[row:row+8, col:col+8]
            bit, confidence = self.extract_bit_from_block(block)
            extracted_bits.append(bit)
            confidence_scores.append(confidence)
        
        # Use weighted majority voting for final decision
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
                # Fallback to simple majority if no confidence
                bit_count = {0: 0, 1: 0}
                for bit in extracted_bits:
                    bit_count[bit] += 1
                
                if bit_count[1] > bit_count[0]:
                    final_bit = 1
                    confidence = bit_count[1] / len(extracted_bits)
                else:
                    final_bit = 0
                    confidence = bit_count[0] / len(extracted_bits)
            
            return final_bit, confidence
        else:
            return 0, 0.0
    
    def decode_video(self, input_path, num_frames_to_analyze=None):
        """
        Decode the binary message from a watermarked video with improved analysis
        
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
            bit, confidence = self.extract_message_from_frame(frame)
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
            vote_counts = {0: 0, 1: 0}
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
        
        print(f"Detailed analysis (sampling every {sample_rate} frames)")
        
        frame_results = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                bit, confidence = self.extract_message_from_frame(frame)
                frame_results.append((frame_count, bit, confidence))
                print(f"Frame {frame_count}: Bit = {bit}, Confidence = {confidence:.2%}")
            
            frame_count += 1
        
        cap.release()
        
        # Calculate detailed statistics
        if frame_results:
            bits = [result[1] for result in frame_results]
            confidences = [result[2] for result in frame_results]
            
            # Weighted voting
            vote_counts = {0: 0, 1: 0}
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
            
            print(f"\nDetailed Analysis Summary:")
            print(f"Frames sampled: {len(frame_results)}")
            print(f"Bit distribution: {dict(bit_counts)}")
            print(f"Average confidence: {avg_confidence:.2%}")
            print(f"Weighted confidence: {weighted_confidence:.2%}")
            print(f"Most likely bit: {final_bit}")
            
            return final_bit, weighted_confidence, frame_results
        
        return None, 0.0, []

def main():
    parser = argparse.ArgumentParser(description='Improved Video Watermark Decoder')
    parser.add_argument('input_video', help='Path to watermarked video file')
    parser.add_argument('--frames', type=int, default=None,
                        help='Number of frames to analyze (default: all frames)')
    parser.add_argument('--detailed', action='store_true',
                        help='Perform detailed frame-by-frame analysis')
    parser.add_argument('--sample-rate', type=int, default=10,
                        help='Sample rate for detailed analysis (default: 10)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_video):
        print(f"Error: Input video file '{args.input_video}' not found!")
        return
    
    # Create decoder
    decoder = ImprovedVideoWatermarkDecoder()
    
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