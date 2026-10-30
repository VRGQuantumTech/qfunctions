"""
Frequency-response model of the t0 CRS polyphase filter bank (PFB).

This first version builds equivalent frequency-domain subband filters.

The response used by rfmux for readout correction is a model, not a measured
end-to-end CRS transfer function. I

Example:

    from CRS import CRS
    import matplotlib.pyplot as plt

    crs = CRS()
    fig, ax = crs.plot_pfbs(bins=[-1, 0, 1])
    plt.show()

All input frequencies and sampling rates are in Hz, not rad/s.

"""

import operator
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal.windows import chebwin
from scipy import fft


class CRS:
    """Construct a bank of shifted copies of the rfmux prototype response.

    Parameters
    ----------
    bandwidth : float
        Complex bandwidth of one module, default 625e6 Hz.
    n_bins : int
        Number of equally spaced PFB bins, default 512. Must be even.
        These are subbands, distinct from the 1024 configurable tone channels.
    decimation : int
        Ratio of module sampling rate to per-channel rate, default 256.
    n_taps : int
        Number of prototype FIR coefficients, default 4096.
    attenuation_db : float
        Chebyshev sidelobe attenuation, default 103 dB.
    n_fft : int
        FFT length for sampling the prototype response, default 262144.
    nco_frequency : float
        Module centre frequency, default 0 Hz (complex baseband).

    Attributes
    ----------
    bin_indices, bin_centers : ndarray
        Signed bin numbers (-256..255 by default) and centre frequencies.
    bin_spacing, channel_sample_rate : float
        Defaults: 1.220703125e6 Hz and 2.44140625e6 samples/s, respectively.
    oversampling : float
        channel_sample_rate / bin_spacing, default 2. This is not a
        specification of flat-passband overlap.
    prototype_taps : ndarray
        Real FIR coefficients, normalized to unit DC amplitude gain.
    prototype_sample_rate : float
        Effective rate implied by rfmux's prototype frequency axis.
        This is not the per-channel rate or the physical DAC rate.
    """

    def __init__(
        self,
        bandwidth       : float = 625e6,
        n_bins          : float = 512,
        decimation      : float = 256,
        n_taps          : float = 4096,
        attenuation_db  : float = 103.0,
        n_fft           : float = 262144,
        nco_frequency   : float = 0.0,
    ):
        for name, value in (
            ("n_bins", n_bins), ("decimation", decimation),
            ("n_taps", n_taps), ("n_fft", n_fft),
        ):
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be a positive integer.")
            try:
                value = operator.index(value)
            except TypeError as exc:
                raise ValueError(f"{name} must be a positive integer.") from exc
            if value < 1:
                raise ValueError(f"{name} must be a positive integer.")
            setattr(self, name, value)

        if self.n_bins % 2 or self.n_taps < 2 or self.n_fft < self.n_taps:
            raise ValueError("Require even n_bins, n_taps >= 2, n_fft >= n_taps.")
        self.bandwidth = float(bandwidth)
        self.attenuation_db = float(attenuation_db)
        self.nco_frequency = float(nco_frequency)
        if not np.isfinite(self.bandwidth) or self.bandwidth <= 0:
            raise ValueError("bandwidth must be finite and positive.")
        if not np.isfinite(self.attenuation_db) or self.attenuation_db < 45:
            raise ValueError("attenuation_db must be finite and >= 45 dB.")
        if not np.isfinite(self.nco_frequency):
            raise ValueError("nco_frequency must be finite.")

        self.bin_spacing = self.bandwidth / self.n_bins
        self.channel_sample_rate = self.bandwidth / self.decimation
        self.oversampling = self.n_bins / self.decimation
        self.bin_indices = np.arange(-self.n_bins // 2, self.n_bins // 2)
        self.bin_centers = self.nco_frequency + self.bin_indices * self.bin_spacing

        self.prototype_taps = chebwin(self.n_taps, self.attenuation_db, sym=True)
        self.prototype_taps /= self.prototype_taps.sum()

        # rfmux maps the prototype FFT onto [-bandwidth, +bandwidth].
        # Preserve that effective 2*bandwidth scale. fftfreq places DC at
        # exactly zero, avoiding the half-step shift of its inclusive linspace.
        self.prototype_sample_rate = 2 * self.bandwidth
        self._prototype_frequency = np.fft.fftshift(
            np.fft.fftfreq(self.n_fft, d=1 / self.prototype_sample_rate)
        )
        self._prototype_response = np.fft.fftshift(
            np.fft.fft(self.prototype_taps, n=self.n_fft)
        )
        
    def transfer_function(self, frequency, bins=None):
        
        # Frequencies are relative to NCO.
        
        frequency = np.atleast_1d(np.asarray(frequency, dtype=float))
    
        response = self.frequency_response(
            self.nco_frequency + frequency,
            bins=bins,
        )
    
        selected = np.argmax(np.abs(response), axis=0)
    
        return response[selected, np.arange(frequency.size)]

    def _select_bins(self, bins):
        """Validate signed bin indices without silently rounding floats."""
        indices = np.atleast_1d(bins)
        if indices.ndim != 1 or indices.size == 0 or indices.dtype.kind not in "iu":
            raise ValueError("bins must contain one or more signed integer indices.")
        if np.any(indices < -self.n_bins // 2) or np.any(indices >= self.n_bins // 2):
            raise ValueError(
                f"bins must lie in [{-self.n_bins // 2}, {self.n_bins // 2 - 1}]."
            )
        return indices.astype(int)

    def frequency_response(self, frequency, bins=None):
        """Return complex subband gains, with shape (number of bins, frequencies).

        ``frequency`` is a scalar or 1D array in Hz, including the NCO offset.
        ``bins`` selects signed bin indices; None selects the complete bank.
        Responses wrap at the module's complex-band boundaries. The result
        describes the filters before decimation; aliasing and the subsequent
        channel DDS are not simulated. No amplitude floor is applied here.
        """
        frequency = np.atleast_1d(np.asarray(frequency, dtype=float))
        if frequency.ndim != 1 or not np.all(np.isfinite(frequency)):
            raise ValueError("frequency must be a finite scalar or 1D array in Hz.")
        indices = self.bin_indices if bins is None else self._select_bins(bins)
        centers = self.nco_frequency + indices * self.bin_spacing
        offsets = frequency[np.newaxis, :] - centers[:, np.newaxis]
        offsets = (offsets + self.bandwidth / 2) % self.bandwidth - self.bandwidth / 2
        return np.interp(
            offsets, self._prototype_frequency, self._prototype_response
        )

    def plot_pfbs(self, 
                  bins       : list | np.ndarray | None = None, 
                  db         : bool = True, 
                  floor_db   : float = -110.0, 
                  n_points   : int = 4097,
                  ax         : None = None,
                  colormap   : str = 'Spectral',
                  cmap_limit : tuple[float, float] = (0.6, 1.0),
                  dpi        : float = 300,
                  ):
        
        """Plot selected subbands and return (figure, axes), without calling show.

        The default displays the five bins nearest DC. ``db=True`` plots
        20*log10(abs(H)); False plots amplitude. floor_db only clips the plot.
        The horizontal axis is frequency relative to the NCO, in MHz.
        """
        
        from matplotlib import colormaps as cm
        import matplotlib.colors as colors
        
    

        if bins is None:
            bins = self.bin_indices[np.abs(self.bin_indices) <= 2]
            
        indices = self._select_bins(bins)
        if isinstance(n_points, (bool, np.bool_)):
            raise ValueError("n_points must be an integer >= 2.")
        try:
            n_points = operator.index(n_points)
        except TypeError as exc:
            raise ValueError("n_points must be an integer >= 2.") from exc
        if n_points < 2 or not np.isfinite(floor_db) or floor_db >= 0:
            raise ValueError("Require n_points >= 2 and finite floor_db < 0.")

        frequency_offset = np.linspace(
            (indices.min() - 1) * self.bin_spacing,
            (indices.max() + 1) * self.bin_spacing,
            n_points,
        )
        
        # Response is a matrix with the number of n_bins x n_points
        response = self.frequency_response(
            self.nco_frequency + frequency_offset, bins=indices
        )
        
        
        cmap = cm.get_cmap(colormap)
        
        cmap = cmap(
            np.linspace(cmap_limit[0], cmap_limit[1], len(indices))
        )
        
        if ax is None:
            _, ax = plt.subplots(
                            figsize=(9, 5.5), 
                            constrained_layout = True,
                            dpi = dpi
                            )
            
        for i, (index, gain) in enumerate(zip(indices, response)):
            amplitude = np.abs(gain)
            
            if db:
                amplitude = 20 * np.log10(np.maximum(amplitude, 10 ** (floor_db / 20)))
                
            line, = ax.plot(
                        frequency_offset / 1e6, 
                        amplitude, 
                        lw = 2.0, 
                        color=cmap[i]
                        )
            
            ax.axvline(index * self.bin_spacing / 1e6,
                       color=line.get_color(), alpha=0.2, linewidth=0.8)
        
        function = np.abs(
            self.transfer_function(frequency_offset, bins=indices)
        )
                
        if db:
            function = 20 * np.log10(np.maximum(function, 10 ** (floor_db / 20)))
            
        ax.plot(frequency_offset / 1e6, function, lw = 4.0, color='black', alpha = 0.2)
        
        ax.set_xlabel("Frequency relative to NCO (MHz)", size = 20)
        ax.set_ylabel("Amplitude gain (dB)" if db else "Amplitude gain", size = 20)
        ax.set_title(f"CRS PFB response model (oversampling = {self.oversampling:g})")
        ax.set_xlim(frequency_offset[0] / 1e6, frequency_offset[-1] / 1e6)
        ax.set_ylim(floor_db if db else 0, 1.05)
        ax.tick_params(which="both", direction="in", top=True, right=True, labelsize = 18)
        ax.grid(alpha=0.2)
        
        return ax.figure, ax

class Pulse:
    
    def __init__(self,
                 fwhm         : 100,           # in sampling unit
                 width_scale  : int = 10,     
                 amplitude    : float = 1.0,
                 rate         : float = 625e6
                 ):
        
        self.width      = fwhm * width_scale
        self.tau        = 1 / rate           # seconds
        self.time       = np.arange(0, self.width, 1) * self.tau
        
        self.center = self.width * self.tau / 2
        self.pulse  = np.exp(- 0.5 * ((self.time - self.center)/ (fwhm * self.tau))**2)
        self.pulse *= (fwhm * self.tau * np.sqrt(2 * np.pi)) **-1
        self.pulse /= self.pulse.max()
        
        self.transfered_spectrum = None
        self.transfered_pulse    = None
        
        
        self._xlim  = (self.center * 1e9 - self.width * self.tau * 1e9/2, 
                       self.center * 1e9 + self.width * self.tau * 1e9 * 5)
        
    
    @property
    def spectrum(self):
        frequency = fft.fftshift(
            fft.fftfreq(len(self.pulse), self.tau)
        )
        spectrum = fft.fftshift(fft.fft(self.pulse))
    
        return frequency, spectrum 
    
    def apply_transfer_function(self, function):
        _, spectrum = self.spectrum
    
        self.transfered_spectrum = spectrum * function
    
        self.transfered_pulse = fft.ifft(
            fft.ifftshift(self.transfered_spectrum)
        )
    
        
    def _add_zeros(self):
        
        length = int(10e-6 / self.tau)
        time   = np.arange(0, length, 1) * self.tau
        pulse  = np.zeros(len(time), dtype = np.float64)
        
        pulse[:self.width] = self.pulse
        
        self.time  = time.copy()
        self.pulse = pulse.copy()
        
        
    def plot(self, db : bool = True):
        
        fig, ax = plt.subplots(2, 1, dpi = 300, constrained_layout = True, figsize=(9, 6),)
        ax[0].plot(self.time * 1e9, self.pulse, lw = 2.0, ls = '--', color = 'black', zorder = 1)
        
        if self.transfered_pulse is not None:
            ax[0].plot(
                self.time * 1e9,
                np.abs(self.transfered_pulse),
                lw=2.0,
                ls='--',
                color='orange',
                zorder=2,
            )
        
        # ax[0].scatter(self.time * 1e9, self.pulse, marker = 'o', s = 25, color = 'black', zorder = 2)
        ax[0].set_ylabel('amplitude [V]', size = 20)
        ax[0].set_xlabel('time [ns]', size = 20)
        ax[0].tick_params(direction = 'in', which = 'both', labelsize = 18)
        ax[0].set_xlim(self._xlim)
        ax[0].grid(alpha = 0.2)
        
        x, y = self.spectrum
        if db: amplitude = 20 * np.log10(abs(y)/abs(y).max())
        else: amplitude = abs(y)/abs(y).max()
        ax[1].plot(x * 1e-6, amplitude, lw = 2.0, color = 'black', zorder=20)
        ax[1].set_ylabel('amplitude [V]', size = 20)
        ax[1].set_xlabel('frequency [MHz]', size = 20)
        ax[1].tick_params(direction = 'in', which = 'both', labelsize = 18)
        
        return ax

if __name__ == "__main__":
        
    pulse = Pulse(fwhm = 10)
    pulse._add_zeros()
    fftfreq, _ = pulse.spectrum

    crs = CRS(decimation = 128, n_bins = 512)
    

    
    ax = pulse.plot()
    crs.plot_pfbs(
                bins = range(-20, 21), 
                n_points = len(fftfreq),
                ax = ax[1]
                )
    
    plt.show()
