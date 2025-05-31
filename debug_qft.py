import cv2
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift

def debug_qft_embedding():
    # Create a simple test frame
    test_frame = np.random.randint(0, 255, (480, 854, 3), dtype=np.uint8)
    
    print("=== QFT EMBEDDING DEBUG ===")
    
    # Test embedding bit 1
    message_bit = 1
    alpha = 0.01
    chaos_param = 3.8
    chaos_seed = 0.123456789
    
    # Generate chaos sequence
    x = chaos_seed
    chaos_seq = []
    for _ in range(10):
        x = chaos_param * x * (1 - x)
        chaos_seq.append(1 if x > 0.5 else -1)
    
    print(f"Chaos sequence (first 4): {chaos_seq[:4]}")
    print(f"Message bit: {message_bit}")
    print(f"Alpha: {alpha}")
    
    # Convert to YUV
    frame_yuv = cv2.cvtColor(test_frame, cv2.COLOR_BGR2YUV)
    y_channel = frame_yuv[:, :, 0].astype(np.float64)
    
    # Apply FFT
    fft_y = fft2(y_channel)
    fft_shifted = fftshift(fft_y)
    
    height, width = y_channel.shape
    center_h, center_w = height // 2, width // 2
    
    # Define embedding positions
    embed_positions = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            if abs(i) == 1 and abs(j) == 1:
                pos_h = center_h + i
                pos_w = center_w + j
                embed_positions.append((pos_h, pos_w))
    
    print(f"Embedding positions: {embed_positions}")
    
    # Store original coefficients
    original_coeffs = []
    for pos_h, pos_w in embed_positions:
        original_coeffs.append(fft_shifted[pos_h, pos_w])
    
    # Embed
    for idx, (pos_h, pos_w) in enumerate(embed_positions):
        chaos_val = chaos_seq[idx]
        
        coeff = fft_shifted[pos_h, pos_w]
        magnitude = abs(coeff)
        phase = np.angle(coeff)
        
        base_modification = alpha * magnitude * 0.0001
        
        # Fixed embedding logic
        if message_bit == 1:
            # For bit 1, always increase magnitude (regardless of chaos sign)
            modification = abs(base_modification)
        else:
            # For bit 0, always decrease magnitude (regardless of chaos sign)
            modification = -abs(base_modification)
        
        new_magnitude = magnitude + modification
        new_coeff = new_magnitude * np.exp(1j * phase)
        fft_shifted[pos_h, pos_w] = new_coeff
        
        print(f"Position {idx}: chaos={chaos_val}, orig_mag={magnitude:.6f}, new_mag={new_magnitude:.6f}, diff={modification:.8f}")
    
    # Apply inverse FFT
    fft_ishifted = ifftshift(fft_shifted)
    y_watermarked = np.real(ifft2(fft_ishifted))
    y_watermarked = np.clip(y_watermarked, 0, 255)
    
    # Create watermarked frame
    frame_yuv_watermarked = frame_yuv.copy()
    frame_yuv_watermarked[:, :, 0] = y_watermarked.astype(np.uint8)
    watermarked_frame = cv2.cvtColor(frame_yuv_watermarked, cv2.COLOR_YUV2BGR)
    
    print("\n=== QFT EXTRACTION DEBUG ===")
    
    # Extract
    frame_yuv_extract = cv2.cvtColor(watermarked_frame, cv2.COLOR_BGR2YUV)
    y_channel_extract = frame_yuv_extract[:, :, 0].astype(np.float64)
    
    fft_y_extract = fft2(y_channel_extract)
    fft_shifted_extract = fftshift(fft_y_extract)
    
    bit_estimates = []
    for idx, (pos_h, pos_w) in enumerate(embed_positions):
        coeff = fft_shifted_extract[pos_h, pos_w]
        magnitude = abs(coeff)
        
        # Get reference magnitudes
        ref_magnitudes = []
        for di in [-3, -2, 2, 3]:
            for dj in [-3, -2, 2, 3]:
                ref_h = pos_h + di
                ref_w = pos_w + dj
                if 0 <= ref_h < height and 0 <= ref_w < width:
                    ref_magnitudes.append(abs(fft_shifted_extract[ref_h, ref_w]))
        
        if ref_magnitudes:
            avg_ref_magnitude = np.mean(ref_magnitudes)
            magnitude_diff = magnitude - avg_ref_magnitude
            
            # Simplified extraction: just compare magnitude to reference
            bit_estimate = 1 if magnitude_diff > 0 else 0
            bit_estimates.append(bit_estimate)
            
            print(f"Extract {idx}: mag={magnitude:.6f}, ref={avg_ref_magnitude:.6f}, diff={magnitude_diff:.8f}, bit={bit_estimate}")
    
    final_bit = 1 if np.mean(bit_estimates) > 0.5 else 0
    print(f"\nBit estimates: {bit_estimates}")
    print(f"Final extracted bit: {final_bit}")
    print(f"Expected bit: {message_bit}")
    print(f"Success: {'YES' if final_bit == message_bit else 'NO'}")

if __name__ == "__main__":
    debug_qft_embedding() 