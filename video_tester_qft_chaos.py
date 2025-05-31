import cv2
import numpy as np
import argparse
import os
import tempfile
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.ndimage import median_filter, rotate
import time

# Import our QFT-Chaos modules
from video_encoder_qft_chaos import QFTChaosVideoWatermarkEncoder
from video_decoder_qft_chaos import QFTChaosVideoWatermarkDecoder

class QFTChaosVideoWatermarkTester:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def calculate_video_metrics(self, original_path, watermarked_path, max_frames=100):
        """
        Calculate PSNR and SSIM between original and watermarked videos
        
        Args:
            original_path: Path to original video
            watermarked_path: Path to watermarked video
            max_frames: Maximum number of frames to compare
        
        Returns:
            Dictionary with metrics
        """
        cap_orig = cv2.VideoCapture(original_path)
        cap_water = cv2.VideoCapture(watermarked_path)
        
        psnr_values = []
        ssim_values = []
        frame_count = 0
        
        print("Calculating quality metrics...")
        
        while frame_count < max_frames:
            ret_orig, frame_orig = cap_orig.read()
            ret_water, frame_water = cap_water.read()
            
            if not ret_orig or not ret_water:
                break
            
            # Convert to grayscale for SSIM calculation
            gray_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
            gray_water = cv2.cvtColor(frame_water, cv2.COLOR_BGR2GRAY)
            
            # Calculate PSNR
            psnr = peak_signal_noise_ratio(frame_orig, frame_water)
            psnr_values.append(psnr)
            
            # Calculate SSIM
            ssim = structural_similarity(gray_orig, gray_water)
            ssim_values.append(ssim)
            
            frame_count += 1
            if frame_count % 20 == 0:
                print(f"  Processed {frame_count} frames")
        
        cap_orig.release()
        cap_water.release()
        
        return {
            'avg_psnr': np.mean(psnr_values),
            'avg_ssim': np.mean(ssim_values),
            'std_psnr': np.std(psnr_values),
            'std_ssim': np.std(ssim_values),
            'min_psnr': np.min(psnr_values),
            'min_ssim': np.min(ssim_values),
            'frames_analyzed': frame_count,
            'psnr_values': psnr_values,
            'ssim_values': ssim_values
        }
    
    def apply_compression_attack(self, input_path, output_path, quality):
        """Apply JPEG compression attack"""
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Define codec with quality
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply JPEG compression to each frame
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, compressed_frame = cv2.imencode('.jpg', frame, encode_param)
            decompressed_frame = cv2.imdecode(compressed_frame, cv2.IMREAD_COLOR)
            
            out.write(decompressed_frame)
            frame_count += 1
        
        cap.release()
        out.release()
        return frame_count
    
    def apply_gaussian_noise_attack(self, input_path, output_path, noise_std):
        """Apply Gaussian noise attack"""
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add Gaussian noise
            noise = np.random.normal(0, noise_std, frame.shape).astype(np.float32)
            noisy_frame = frame.astype(np.float32) + noise
            noisy_frame = np.clip(noisy_frame, 0, 255).astype(np.uint8)
            
            out.write(noisy_frame)
            frame_count += 1
        
        cap.release()
        out.release()
        return frame_count
    
    def apply_median_filter_attack(self, input_path, output_path, filter_size):
        """Apply median filter attack"""
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply median filter to each channel
            filtered_frame = np.zeros_like(frame)
            for c in range(3):
                filtered_frame[:,:,c] = median_filter(frame[:,:,c], size=filter_size)
            
            out.write(filtered_frame)
            frame_count += 1
        
        cap.release()
        out.release()
        return frame_count
    
    def apply_rotation_attack(self, input_path, output_path, angle):
        """Apply rotation attack"""
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Rotate frame
            rotated_frame = rotate(frame, angle, reshape=False, mode='nearest')
            rotated_frame = np.clip(rotated_frame, 0, 255).astype(np.uint8)
            
            out.write(rotated_frame)
            frame_count += 1
        
        cap.release()
        out.release()
        return frame_count
    
    def test_robustness(self, watermarked_path, original_message, 
                       encoder_params=None, decoder_params=None):
        """
        Test robustness against various attacks
        
        Args:
            watermarked_path: Path to watermarked video
            original_message: Original embedded message
            encoder_params: Parameters used for encoding
            decoder_params: Parameters for decoding
        
        Returns:
            Dictionary with robustness test results
        """
        if encoder_params is None:
            encoder_params = {'chaos_param': 3.8, 'arnold_iterations': 5}
        if decoder_params is None:
            decoder_params = encoder_params.copy()
        
        decoder = QFTChaosVideoWatermarkDecoder(
            chaos_param=decoder_params['chaos_param'],
            arnold_iterations=decoder_params['arnold_iterations'],
            multi_scale=decoder_params.get('multi_scale', True)
        )
        
        results = {}
        
        print("\n=== ROBUSTNESS TESTING ===")
        
        # Test 1: No attack (baseline)
        print("Testing baseline (no attack)...")
        result = decoder.decode_video(watermarked_path, max_frames=50)
        if result:
            baseline_success = (result['message'] == original_message)
            baseline_confidence = result['confidence']
        else:
            baseline_success = False
            baseline_confidence = 0.0
        
        results['baseline'] = {
            'success': baseline_success,
            'confidence': baseline_confidence,
            'extracted_message': result['message'] if result else None
        }
        
        # Test 2: Compression attacks
        print("Testing compression attacks...")
        compression_results = {}
        compression_qualities = [90, 70, 50, 30, 20, 10, 5]
        
        for quality in compression_qualities:
            print(f"  Testing compression quality {quality}%...")
            
            # Apply compression
            compressed_path = os.path.join(self.temp_dir, f'compressed_{quality}.mp4')
            try:
                self.apply_compression_attack(watermarked_path, compressed_path, quality)
                
                # Test extraction
                result = decoder.decode_video(compressed_path, max_frames=30)
                if result:
                    success = (result['message'] == original_message)
                    confidence = result['confidence']
                else:
                    success = False
                    confidence = 0.0
                
                compression_results[quality] = {
                    'success': success,
                    'confidence': confidence,
                    'extracted_message': result['message'] if result else None
                }
                
                # Clean up
                if os.path.exists(compressed_path):
                    os.remove(compressed_path)
                    
            except Exception as e:
                print(f"    Error with compression {quality}%: {str(e)}")
                compression_results[quality] = {'success': False, 'confidence': 0.0, 'error': str(e)}
        
        results['compression'] = compression_results
        
        # Test 3: Gaussian noise attacks
        print("Testing Gaussian noise attacks...")
        noise_results = {}
        noise_levels = [5, 10, 15, 20, 25, 30]
        
        for noise_std in noise_levels:
            print(f"  Testing noise std {noise_std}...")
            
            # Apply noise
            noisy_path = os.path.join(self.temp_dir, f'noisy_{noise_std}.mp4')
            try:
                self.apply_gaussian_noise_attack(watermarked_path, noisy_path, noise_std)
                
                # Test extraction
                result = decoder.decode_video(noisy_path, max_frames=30)
                if result:
                    success = (result['message'] == original_message)
                    confidence = result['confidence']
                else:
                    success = False
                    confidence = 0.0
                
                noise_results[noise_std] = {
                    'success': success,
                    'confidence': confidence,
                    'extracted_message': result['message'] if result else None
                }
                
                # Clean up
                if os.path.exists(noisy_path):
                    os.remove(noisy_path)
                    
            except Exception as e:
                print(f"    Error with noise {noise_std}: {str(e)}")
                noise_results[noise_std] = {'success': False, 'confidence': 0.0, 'error': str(e)}
        
        results['noise'] = noise_results
        
        # Test 4: Median filter attacks
        print("Testing median filter attacks...")
        filter_results = {}
        filter_sizes = [3, 5, 7]
        
        for filter_size in filter_sizes:
            print(f"  Testing median filter size {filter_size}x{filter_size}...")
            
            # Apply median filter
            filtered_path = os.path.join(self.temp_dir, f'filtered_{filter_size}.mp4')
            try:
                self.apply_median_filter_attack(watermarked_path, filtered_path, filter_size)
                
                # Test extraction
                result = decoder.decode_video(filtered_path, max_frames=30)
                if result:
                    success = (result['message'] == original_message)
                    confidence = result['confidence']
                else:
                    success = False
                    confidence = 0.0
                
                filter_results[filter_size] = {
                    'success': success,
                    'confidence': confidence,
                    'extracted_message': result['message'] if result else None
                }
                
                # Clean up
                if os.path.exists(filtered_path):
                    os.remove(filtered_path)
                    
            except Exception as e:
                print(f"    Error with filter {filter_size}: {str(e)}")
                filter_results[filter_size] = {'success': False, 'confidence': 0.0, 'error': str(e)}
        
        results['filter'] = filter_results
        
        # Test 5: Rotation attacks
        print("Testing rotation attacks...")
        rotation_results = {}
        rotation_angles = [1, 2, 5, 10]
        
        for angle in rotation_angles:
            print(f"  Testing rotation {angle} degrees...")
            
            # Apply rotation
            rotated_path = os.path.join(self.temp_dir, f'rotated_{angle}.mp4')
            try:
                self.apply_rotation_attack(watermarked_path, rotated_path, angle)
                
                # Test extraction
                result = decoder.decode_video(rotated_path, max_frames=30)
                if result:
                    success = (result['message'] == original_message)
                    confidence = result['confidence']
                else:
                    success = False
                    confidence = 0.0
                
                rotation_results[angle] = {
                    'success': success,
                    'confidence': confidence,
                    'extracted_message': result['message'] if result else None
                }
                
                # Clean up
                if os.path.exists(rotated_path):
                    os.remove(rotated_path)
                    
            except Exception as e:
                print(f"    Error with rotation {angle}°: {str(e)}")
                rotation_results[angle] = {'success': False, 'confidence': 0.0, 'error': str(e)}
        
        results['rotation'] = rotation_results
        
        return results
    
    def print_results_summary(self, quality_metrics, robustness_results):
        """Print a comprehensive summary of test results"""
        print("\n" + "="*60)
        print("         QFT-CHAOS WATERMARKING TEST RESULTS")
        print("="*60)
        
        # Quality metrics
        print(f"\n=== QUALITY METRICS ===")
        print(f"Average PSNR: {quality_metrics['avg_psnr']:.2f} dB")
        print(f"Average SSIM: {quality_metrics['avg_ssim']:.4f}")
        print(f"Minimum PSNR: {quality_metrics['min_psnr']:.2f} dB")
        print(f"Minimum SSIM: {quality_metrics['min_ssim']:.4f}")
        print(f"Frames analyzed: {quality_metrics['frames_analyzed']}")
        
        # Quality assessment
        if quality_metrics['avg_psnr'] > 35:
            quality_grade = "EXCELLENT"
        elif quality_metrics['avg_psnr'] > 30:
            quality_grade = "GOOD"
        elif quality_metrics['avg_psnr'] > 25:
            quality_grade = "FAIR"
        else:
            quality_grade = "POOR"
        print(f"Quality Grade: {quality_grade}")
        
        # Robustness results
        print(f"\n=== ROBUSTNESS RESULTS ===")
        
        # Baseline
        baseline = robustness_results['baseline']
        print(f"Baseline (no attack): {'PASS' if baseline['success'] else 'FAIL'} (confidence: {baseline['confidence']:.3f})")
        
        # Compression attacks
        compression = robustness_results['compression']
        compression_passes = sum(1 for result in compression.values() if result['success'])
        compression_total = len(compression)
        print(f"\nCompression attacks: {compression_passes}/{compression_total} passed")
        for quality, result in compression.items():
            status = "PASS" if result['success'] else "FAIL"
            conf = result.get('confidence', 0.0)
            print(f"  Quality {quality:2d}%: {status} (confidence: {conf:.3f})")
        
        # Noise attacks
        noise = robustness_results['noise']
        noise_passes = sum(1 for result in noise.values() if result['success'])
        noise_total = len(noise)
        print(f"\nGaussian noise attacks: {noise_passes}/{noise_total} passed")
        for std, result in noise.items():
            status = "PASS" if result['success'] else "FAIL"
            conf = result.get('confidence', 0.0)
            print(f"  Noise std {std:2d}: {status} (confidence: {conf:.3f})")
        
        # Filter attacks
        filter_results = robustness_results['filter']
        filter_passes = sum(1 for result in filter_results.values() if result['success'])
        filter_total = len(filter_results)
        print(f"\nMedian filter attacks: {filter_passes}/{filter_total} passed")
        for size, result in filter_results.items():
            status = "PASS" if result['success'] else "FAIL"
            conf = result.get('confidence', 0.0)
            print(f"  Filter {size}x{size}: {status} (confidence: {conf:.3f})")
        
        # Rotation attacks
        rotation = robustness_results['rotation']
        rotation_passes = sum(1 for result in rotation.values() if result['success'])
        rotation_total = len(rotation)
        print(f"\nRotation attacks: {rotation_passes}/{rotation_total} passed")
        for angle, result in rotation.items():
            status = "PASS" if result['success'] else "FAIL"
            conf = result.get('confidence', 0.0)
            print(f"  Rotation {angle:2d}°: {status} (confidence: {conf:.3f})")
        
        # Overall robustness
        total_passes = compression_passes + noise_passes + filter_passes + rotation_passes
        total_tests = compression_total + noise_total + filter_total + rotation_total
        overall_success_rate = total_passes / total_tests if total_tests > 0 else 0
        
        print(f"\n=== OVERALL SUMMARY ===")
        print(f"Overall robustness: {total_passes}/{total_tests} ({overall_success_rate:.1%})")
        
        if overall_success_rate > 0.8:
            robustness_grade = "EXCELLENT"
        elif overall_success_rate > 0.6:
            robustness_grade = "GOOD"
        elif overall_success_rate > 0.4:
            robustness_grade = "FAIR"
        else:
            robustness_grade = "POOR"
        
        print(f"Robustness Grade: {robustness_grade}")
        print(f"Quality Grade: {quality_grade}")
    
    def plot_metrics(self, quality_metrics, save_path=None):
        """Plot quality metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # PSNR plot
        frames = range(len(quality_metrics['psnr_values']))
        ax1.plot(frames, quality_metrics['psnr_values'], 'b-', alpha=0.7)
        ax1.axhline(y=quality_metrics['avg_psnr'], color='r', linestyle='--', 
                   label=f'Average: {quality_metrics["avg_psnr"]:.2f} dB')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('PSNR (dB)')
        ax1.set_title('Peak Signal-to-Noise Ratio')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # SSIM plot
        ax2.plot(frames, quality_metrics['ssim_values'], 'g-', alpha=0.7)
        ax2.axhline(y=quality_metrics['avg_ssim'], color='r', linestyle='--',
                   label=f'Average: {quality_metrics["avg_ssim"]:.4f}')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('SSIM')
        ax2.set_title('Structural Similarity Index')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Quality metrics plot saved to: {save_path}")
        else:
            plt.show()
    
    def run_complete_test(self, original_video, watermarked_video, original_message,
                         max_frames=100, plot=False, plot_save=None, skip_robustness=False,
                         encoder_params=None, decoder_params=None):
        """
        Run complete test suite
        
        Args:
            original_video: Path to original video
            watermarked_video: Path to watermarked video
            original_message: Original embedded message
            max_frames: Maximum frames to analyze for quality
            plot: Whether to show plots
            plot_save: Path to save plots
            skip_robustness: Whether to skip robustness testing
            encoder_params: Parameters used for encoding
            decoder_params: Parameters for decoding
        
        Returns:
            Complete test results
        """
        print("QFT-Chaos Video Watermarking Test Suite")
        print("="*50)
        
        start_time = time.time()
        
        # Calculate quality metrics
        print("Calculating quality metrics...")
        quality_metrics = self.calculate_video_metrics(original_video, watermarked_video, max_frames)
        
        robustness_results = None
        if not skip_robustness:
            # Test robustness
            robustness_results = self.test_robustness(
                watermarked_video, original_message, encoder_params, decoder_params
            )
        
        # Print results
        if robustness_results:
            self.print_results_summary(quality_metrics, robustness_results)
        else:
            print(f"\n=== QUALITY METRICS ONLY ===")
            print(f"Average PSNR: {quality_metrics['avg_psnr']:.2f} dB")
            print(f"Average SSIM: {quality_metrics['avg_ssim']:.4f}")
        
        # Plot if requested
        if plot or plot_save:
            self.plot_metrics(quality_metrics, plot_save)
        
        end_time = time.time()
        print(f"\nTotal testing time: {end_time - start_time:.1f} seconds")
        
        return {
            'quality_metrics': quality_metrics,
            'robustness_results': robustness_results
        }

def main():
    parser = argparse.ArgumentParser(description='QFT-Chaos Video Watermark Tester')
    parser.add_argument('original_video', help='Path to original video file')
    parser.add_argument('watermarked_video', help='Path to watermarked video file')
    parser.add_argument('original_message', type=int, choices=[0, 1], help='Original embedded message')
    parser.add_argument('--frames', type=int, default=100, help='Maximum frames to analyze for quality')
    parser.add_argument('--plot', action='store_true', help='Show quality metrics plots')
    parser.add_argument('--plot-save', help='Save plots to specified path')
    parser.add_argument('--skip-robustness', action='store_true', help='Skip robustness testing')
    parser.add_argument('--chaos', type=float, default=3.8, help='Chaos parameter used for encoding')
    parser.add_argument('--arnold', type=int, default=5, help='Arnold iterations used for encoding')
    parser.add_argument('--no-multiscale', action='store_true', help='Disable multi-scale (if used in encoding)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.original_video):
        print(f"Error: Original video file '{args.original_video}' does not exist")
        return
    
    if not os.path.exists(args.watermarked_video):
        print(f"Error: Watermarked video file '{args.watermarked_video}' does not exist")
        return
    
    # Set up parameters
    encoder_params = {
        'chaos_param': args.chaos,
        'arnold_iterations': args.arnold,
        'multi_scale': not args.no_multiscale
    }
    decoder_params = encoder_params.copy()
    
    # Create tester and run tests
    tester = QFTChaosVideoWatermarkTester()
    
    try:
        results = tester.run_complete_test(
            args.original_video,
            args.watermarked_video,
            args.original_message,
            max_frames=args.frames,
            plot=args.plot,
            plot_save=args.plot_save,
            skip_robustness=args.skip_robustness,
            encoder_params=encoder_params,
            decoder_params=decoder_params
        )
        
        print("\nTesting completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        return

if __name__ == "__main__":
    main() 