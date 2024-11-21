import torch
import pytest

from source.nnet.feature_extraction.stft import STFT, STFTConfig


def test_invalid_window_type():
    with pytest.raises(ValueError, match="Unsupported window type"):
        # Create STFT with an invalid window type
        STFT(n_fft=512, window_length=400, hop_length=160, sample_rate=16000, window_type="unsupported_window")

def test_invalid_output_type():
    with pytest.raises(ValueError, match="Unsupported output_type"):
        # Create STFT with an invalid output type
        STFT(n_fft=512, window_length=400, hop_length=160, sample_rate=16000, output_type="unsupported_output")

@pytest.mark.parametrize("output_type", ["raw", "power", "log_power", "power_phase", "log_power_phase"])
def test_stft(output_type):
    # Generate a random waveform (batch_size=4, channels=1, signal_length=1024)
    batch_size, channels, signal_length = 4, 1, 1024
    waveform = torch.randn(batch_size, channels, signal_length)

    # Normalize the waveform
    rms = torch.sqrt(torch.mean(waveform ** 2, dim=-1, keepdim=True))
    normalized_waveform = waveform / rms  # Unit RMS

    # Create STFT instance
    config = STFTConfig(output_type=output_type)
    stft = STFT(**config.__dict__)

    # Apply STFT
    stft_output = stft(normalized_waveform)

    # Check output
    if output_type == "log_power":
        # Ensure log power is valid
        assert torch.all(~stft_output.isnan()), "Log-power values should not be NaN."
    elif output_type == "power":
        # Ensure power is non-negative
        assert torch.all(stft_output >= 0), "Power values should be >= 0."
    elif output_type == "raw":
        # For raw values, we expect complex numbers, so no specific range check
        assert stft_output.is_complex(), "Raw STFT output should be complex."
    elif output_type == "power_phase":
        # Ensure power and phase are correctly concatenated
        assert stft_output.shape[-1] == (2 * (stft.n_fft // 2 + 1)), "Power-phase output should have 2 * (n_fft // 2 + 1) features."
    elif output_type == "log_power_phase":
        # Ensure log power and phase are correctly concatenated
        assert stft_output.shape[-1] == (2 * (stft.n_fft // 2 + 1)), "Power-phase output should have 2 * (n_fft // 2 + 1) features."
        # Ensure log power is valid
        assert torch.all(~stft_output.isnan()), "Log-power values should not be NaN."
