import cv2
import numpy as np
import argparse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
import matplotlib.pyplot as plt
from video_decoder_improved import ImprovedVideoWatermarkDecoder
import tempfile

class ImprovedVideoWatermarkTester:
    def __init__(self):
        """Initialize the improved video watermark tester"""
        self.decoder = ImprovedVideoWatermarkDecoder()
    
    def calculate_frame_metrics(self, frame1, frame2):
        """
        Calculate PSNR and SSIM between two frames
        
        Args:
            frame1: Original frame
            frame2: Watermarked/compressed frame
        
        Returns:
            PSNR and SSIM values
        """
        # Convert to grayscale for PSNR/SSIM calculation
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray1, gray2 = frame1, frame2
        
        # Calculate PSNR
        psnr_value = psnr(gray1, gray2, data_range=255)
        
        # Calculate SSIM
        ssim_value = ssim(gray1, gray2, data_range=255)
        
        return psnr_value, ssim_value
    
    def compare_videos(self, original_path, watermarked_path, num_frames=None):
        """
        Compare original and watermarked videos
        
        Args:
            original_path: Path to original video
            watermarked_path: Path to watermarked video
            num_frames: Number of frames to compare (None = all frames)
        
        Returns:
            Comparison results
        """
        cap_orig = cv2.VideoCapture(original_path)
        cap_wm = cv2.VideoCapture(watermarked_path)
        
        # Get video properties
        total_frames_orig = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames_wm = int(cap_wm.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames_orig != total_frames_wm:
            print(f"Warning: Frame count mismatch - Original: {total_frames_orig}, Watermarked: {total_frames_wm}")
        
        total_frames = min(total_frames_orig, total_frames_wm)
        
        if num_frames is None:
            num_frames = total_frames
        else:
            num_frames = min(num_frames, total_frames)
        
        print(f"Comparing {num_frames} frames...")
        
        psnr_values = []
        ssim_values = []
        frame_count = 0
        
        while frame_count < num_frames:
            ret_orig, frame_orig = cap_orig.read()
            ret_wm, frame_wm = cap_wm.read()
            
            if not ret_orig or not ret_wm:
                break
            
            # Calculate metrics
            psnr_val, ssim_val = self.calculate_frame_metrics(frame_orig, frame_wm)
            psnr_values.append(psnr_val)
            ssim_values.append(ssim_val)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{num_frames} frames")
        
        cap_orig.release()
        cap_wm.release()
        
        # Calculate statistics
        results = {
            'frames_compared': len(psnr_values),
            'psnr_mean': np.mean(psnr_values),
            'psnr_std': np.std(psnr_values),
            'psnr_min': np.min(psnr_values),
            'psnr_max': np.max(psnr_values),
            'ssim_mean': np.mean(ssim_values),
            'ssim_std': np.std(ssim_values),
            'ssim_min': np.min(ssim_values),
            'ssim_max': np.max(ssim_values),
            'psnr_values': psnr_values,
            'ssim_values': ssim_values
        }
        
        return results
    
    def apply_compression(self, input_path, output_path, quality):
        """
        Apply compression to a video
        
        Args:
            input_path: Input video path
            output_path: Output compressed video path
            quality: Compression quality (lower = more compression)
        """
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create writer with compression
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Applying compression (quality={quality})...")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply JPEG compression to simulate quality loss
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encimg = cv2.imencode('.jpg', frame, encode_param)
            compressed_frame = cv2.imdecode(encimg, 1)
            
            out.write(compressed_frame)
            frame_count += 1
        
        cap.release()
        out.release()
        
        print(f"Compressed video saved: {output_path}")
    
    def apply_gaussian_noise(self, input_path, output_path, noise_strength=10):
        """
        Apply Gaussian noise to a video
        
        Args:
            input_path: Input video path
            output_path: Output noisy video path
            noise_strength: Standard deviation of Gaussian noise
        """
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Applying Gaussian noise (strength={noise_strength})...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add Gaussian noise
            noise = np.random.normal(0, noise_strength, frame.shape).astype(np.float32)
            noisy_frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            
            out.write(noisy_frame)
        
        cap.release()
        out.release()
        
        print(f"Noisy video saved: {output_path}")
    
    def apply_resize_attack(self, input_path, output_path, scale_factor=0.5):
        """
        Apply resize attack (scale down and back up)
        
        Args:
            input_path: Input video path
            output_path: Output resized video path
            scale_factor: Scale factor for resize attack
        """
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Applying resize attack (scale={scale_factor})...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize down and back up
            small_height = int(height * scale_factor)
            small_width = int(width * scale_factor)
            
            small_frame = cv2.resize(frame, (small_width, small_height))
            resized_frame = cv2.resize(small_frame, (width, height))
            
            out.write(resized_frame)
        
        cap.release()
        out.release()
        
        print(f"Resized video saved: {output_path}")
    
    def test_robustness(self, watermarked_path, original_message):
        """
        Test robustness against various attacks with improved analysis
        
        Args:
            watermarked_path: Path to watermarked video
            original_message: Original embedded message
        
        Returns:
            Robustness test results
        """
        results = {}
        
        # Test 1: Original watermarked video
        print("\n=== Testing Original Watermarked Video ===")
        decoded_msg, confidence = self.decoder.decode_video(watermarked_path)
        results['original'] = {
            'decoded_message': decoded_msg,
            'confidence': confidence,
            'correct': decoded_msg == original_message
        }
        
        # Test 2: Compression attacks with broader range
        compression_qualities = [95, 80, 60, 40, 20, 10]
        results['compression'] = {}
        
        for quality in compression_qualities:
            print(f"\n=== Testing Compression (Quality {quality}) ===")
            
            # Create temporary compressed video
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                compressed_path = tmp_file.name
            
            try:
                self.apply_compression(watermarked_path, compressed_path, quality)
                decoded_msg, confidence = self.decoder.decode_video(compressed_path)
                
                results['compression'][quality] = {
                    'decoded_message': decoded_msg,
                    'confidence': confidence,
                    'correct': decoded_msg == original_message
                }
                
                print(f"Quality {quality}: Decoded={decoded_msg}, Confidence={confidence:.2%}, Correct={decoded_msg == original_message}")
                
            finally:
                # Clean up temporary file
                if os.path.exists(compressed_path):
                    os.unlink(compressed_path)
        
        # Test 3: Noise attacks
        noise_levels = [3, 7, 12, 18, 25]
        results['noise'] = {}
        
        for noise_level in noise_levels:
            print(f"\n=== Testing Gaussian Noise (Level {noise_level}) ===")
            
            # Create temporary noisy video
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                noisy_path = tmp_file.name
            
            try:
                self.apply_gaussian_noise(watermarked_path, noisy_path, noise_level)
                decoded_msg, confidence = self.decoder.decode_video(noisy_path)
                
                results['noise'][noise_level] = {
                    'decoded_message': decoded_msg,
                    'confidence': confidence,
                    'correct': decoded_msg == original_message
                }
                
                print(f"Noise {noise_level}: Decoded={decoded_msg}, Confidence={confidence:.2%}, Correct={decoded_msg == original_message}")
                
            finally:
                # Clean up temporary file
                if os.path.exists(noisy_path):
                    os.unlink(noisy_path)
        
        # Test 4: Resize attacks
        resize_factors = [0.7, 0.5, 0.3]
        results['resize'] = {}
        
        for scale_factor in resize_factors:
            print(f"\n=== Testing Resize Attack (Scale {scale_factor}) ===")
            
            # Create temporary resized video
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                resized_path = tmp_file.name
            
            try:
                self.apply_resize_attack(watermarked_path, resized_path, scale_factor)
                decoded_msg, confidence = self.decoder.decode_video(resized_path)
                
                results['resize'][scale_factor] = {
                    'decoded_message': decoded_msg,
                    'confidence': confidence,
                    'correct': decoded_msg == original_message
                }
                
                print(f"Resize {scale_factor}: Decoded={decoded_msg}, Confidence={confidence:.2%}, Correct={decoded_msg == original_message}")
                
            finally:
                # Clean up temporary file
                if os.path.exists(resized_path):
                    os.unlink(resized_path)
        
        return results
    
    def generate_report(self, comparison_results, robustness_results, original_message):
        """
        Generate a comprehensive test report with improved analysis
        
        Args:
            comparison_results: Results from video comparison
            robustness_results: Results from robustness testing
            original_message: Original embedded message
        """
        print("\n" + "="*80)
        print("              IMPROVED VIDEO WATERMARKING TEST REPORT")
        print("="*80)
        
        # Quality metrics
        print("\n📊 QUALITY METRICS (Original vs Watermarked)")
        print("-" * 50)
        print(f"Frames analyzed: {comparison_results['frames_compared']}")
        print(f"PSNR: {comparison_results['psnr_mean']:.2f} ± {comparison_results['psnr_std']:.2f} dB")
        print(f"      Range: [{comparison_results['psnr_min']:.2f}, {comparison_results['psnr_max']:.2f}] dB")
        print(f"SSIM: {comparison_results['ssim_mean']:.4f} ± {comparison_results['ssim_std']:.4f}")
        print(f"      Range: [{comparison_results['ssim_min']:.4f}, {comparison_results['ssim_max']:.4f}]")
        
        # Quality assessment
        if comparison_results['psnr_mean'] > 40:
            quality_level = "Excellent (imperceptible)"
        elif comparison_results['psnr_mean'] > 30:
            quality_level = "Good (barely perceptible)"
        elif comparison_results['psnr_mean'] > 20:
            quality_level = "Fair (perceptible)"
        else:
            quality_level = "Poor (very noticeable)"
        
        print(f"Quality Assessment: {quality_level}")
        
        # Robustness results
        print(f"\n🛡️ ROBUSTNESS TEST RESULTS (Original message: {original_message})")
        print("-" * 50)
        
        # Original watermarked video
        orig_result = robustness_results['original']
        print(f"Original Watermarked: ✅ {orig_result['decoded_message']} ({orig_result['confidence']:.1%})" if orig_result['correct'] else f"Original Watermarked: ❌ {orig_result['decoded_message']} ({orig_result['confidence']:.1%})")
        
        # Compression robustness
        print("\nCompression Robustness:")
        for quality in sorted(robustness_results['compression'].keys(), reverse=True):
            result = robustness_results['compression'][quality]
            status = "✅" if result['correct'] else "❌"
            print(f"  Quality {quality:2d}: {status} {result['decoded_message']} ({result['confidence']:.1%})")
        
        # Noise robustness
        print("\nNoise Robustness:")
        for noise_level in sorted(robustness_results['noise'].keys()):
            result = robustness_results['noise'][noise_level]
            status = "✅" if result['correct'] else "❌"
            print(f"  Noise {noise_level:2d}: {status} {result['decoded_message']} ({result['confidence']:.1%})")
        
        # Resize robustness
        print("\nResize Robustness:")
        for scale_factor in sorted(robustness_results['resize'].keys(), reverse=True):
            result = robustness_results['resize'][scale_factor]
            status = "✅" if result['correct'] else "❌"
            print(f"  Scale {scale_factor:.1f}: {status} {result['decoded_message']} ({result['confidence']:.1%})")
        
        # Summary statistics
        compression_success = sum(1 for r in robustness_results['compression'].values() if r['correct'])
        noise_success = sum(1 for r in robustness_results['noise'].values() if r['correct'])
        resize_success = sum(1 for r in robustness_results['resize'].values() if r['correct'])
        
        print(f"\n📈 SUMMARY STATISTICS")
        print("-" * 50)
        print(f"Compression Robustness: {compression_success}/{len(robustness_results['compression'])} tests passed ({compression_success/len(robustness_results['compression']):.1%})")
        print(f"Noise Robustness: {noise_success}/{len(robustness_results['noise'])} tests passed ({noise_success/len(robustness_results['noise']):.1%})")
        print(f"Resize Robustness: {resize_success}/{len(robustness_results['resize'])} tests passed ({resize_success/len(robustness_results['resize']):.1%})")
        
        total_tests = 1 + len(robustness_results['compression']) + len(robustness_results['noise']) + len(robustness_results['resize'])
        total_passed = (1 if orig_result['correct'] else 0) + compression_success + noise_success + resize_success
        
        print(f"Overall Success Rate: {total_passed}/{total_tests} ({total_passed/total_tests:.1%})")
        
        # Confidence analysis
        all_confidences = []
        all_confidences.append(orig_result['confidence'])
        all_confidences.extend([r['confidence'] for r in robustness_results['compression'].values()])
        all_confidences.extend([r['confidence'] for r in robustness_results['noise'].values()])
        all_confidences.extend([r['confidence'] for r in robustness_results['resize'].values()])
        
        print(f"\n📊 CONFIDENCE ANALYSIS")
        print("-" * 50)
        print(f"Average confidence: {np.mean(all_confidences):.1%}")
        print(f"Min confidence: {np.min(all_confidences):.1%}")
        print(f"Max confidence: {np.max(all_confidences):.1%}")
        
        print("\n" + "="*80)
    
    def plot_metrics(self, comparison_results, save_path=None):
        """
        Plot PSNR and SSIM values over frames
        
        Args:
            comparison_results: Results from video comparison
            save_path: Optional path to save the plot
        """
        psnr_values = comparison_results['psnr_values']
        ssim_values = comparison_results['ssim_values']
        frames = range(1, len(psnr_values) + 1)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # PSNR plot
        ax1.plot(frames, psnr_values, 'b-', linewidth=1, alpha=0.7)
        ax1.axhline(y=np.mean(psnr_values), color='r', linestyle='--', label=f'Mean: {np.mean(psnr_values):.2f} dB')
        ax1.set_xlabel('Frame Number')
        ax1.set_ylabel('PSNR (dB)')
        ax1.set_title('Peak Signal-to-Noise Ratio (PSNR)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # SSIM plot
        ax2.plot(frames, ssim_values, 'g-', linewidth=1, alpha=0.7)
        ax2.axhline(y=np.mean(ssim_values), color='r', linestyle='--', label=f'Mean: {np.mean(ssim_values):.4f}')
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('SSIM')
        ax2.set_title('Structural Similarity Index (SSIM)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()

def main():
    parser = argparse.ArgumentParser(description='Improved Video Watermark Robustness Tester')
    parser.add_argument('original_video', help='Path to original video file')
    parser.add_argument('watermarked_video', help='Path to watermarked video file')
    parser.add_argument('original_message', type=int, choices=[0, 1], 
                        help='Original embedded message (0 or 1)')
    parser.add_argument('--frames', type=int, default=None,
                        help='Number of frames to compare (default: all frames)')
    parser.add_argument('--plot', action='store_true',
                        help='Generate quality metrics plot')
    parser.add_argument('--plot-save', type=str,
                        help='Save plot to specified path')
    parser.add_argument('--skip-robustness', action='store_true',
                        help='Skip robustness testing (only do quality comparison)')
    
    args = parser.parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.original_video):
        print(f"Error: Original video file '{args.original_video}' not found!")
        return
    
    if not os.path.exists(args.watermarked_video):
        print(f"Error: Watermarked video file '{args.watermarked_video}' not found!")
        return
    
    # Create tester
    tester = ImprovedVideoWatermarkTester()
    
    # Compare videos
    print("🔍 Comparing original and watermarked videos...")
    comparison_results = tester.compare_videos(args.original_video, args.watermarked_video, args.frames)
    
    # Test robustness (if not skipped)
    if not args.skip_robustness:
        print("\n🧪 Testing robustness against attacks...")
        robustness_results = tester.test_robustness(args.watermarked_video, args.original_message)
    else:
        robustness_results = None
    
    # Generate report
    if robustness_results:
        tester.generate_report(comparison_results, robustness_results, args.original_message)
    else:
        print("\n📊 QUALITY METRICS ONLY")
        print(f"PSNR: {comparison_results['psnr_mean']:.2f} ± {comparison_results['psnr_std']:.2f} dB")
        print(f"SSIM: {comparison_results['ssim_mean']:.4f} ± {comparison_results['ssim_std']:.4f}")
    
    # Generate plot if requested
    if args.plot or args.plot_save:
        print("\n📈 Generating quality metrics plot...")
        tester.plot_metrics(comparison_results, args.plot_save)

if __name__ == "__main__":
    main() 