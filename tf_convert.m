close all; clear; clc;

% === Load Epoched EEG Data ===
load('epochs_data.mat');  % Variables: epoch_data, times, sfreq, channels, labels

SR = 250;      % EEG rate

[n_epochs, n_channels, n_times] = size(epoch_data);
time_ms = times * 1000;

% === Frequency Setup ===
min_freq = 2; max_freq = 40;
num_frex = max_freq - min_freq + 1;
frex = logspace(log10(min_freq),log10(max_freq), num_frex);
wave_cycles = linspace(3, 10, num_frex);  % Variable cycles

% other wavelet parameters
range_cycles = [ 2 12 ];
s    = logspace(log10(range_cycles(1)),log10(range_cycles(2)),num_frex)./(2*pi*frex);
wavtime = -2:1/SR:2;
half_wave = (length(wavtime)-1)/2;

% === Time Vector for Wavelet ===
wavtime = -2:1/sfreq:2;
half_wave = floor(length(wavtime)/2);

% === Morlet Convolution and Power Computation ===
tf_data = zeros(n_epochs, num_frex, n_times, n_channels);

for ep = 1:n_epochs
    for ch = 1:n_channels
        signal = squeeze(epoch_data(ep, ch, :))';
        signalX = fft(signal, n_times + length(wavtime) - 1);
        for fi = 1:num_frex
            f = frex(fi);
            s = wave_cycles(fi) / (2*pi*f);
            wavelet = exp(2*1i*pi*f.*wavtime) .* exp(-wavtime.^2 / (2*s^2));
            waveletX = fft(wavelet, n_times + length(wavtime) - 1);
            waveletX = waveletX ./ max(waveletX);
            conv_res = ifft(waveletX .* signalX);
            conv_res = conv_res(half_wave+1:end-half_wave);
            tf_data(ep, fi, :, ch) = abs(conv_res).^2;
        end
    end
end

% === Average Over Epochs ===
tf_avg = squeeze(mean(tf_data, 1));  % freqs x time x channels

% === Baseline Normalization (dB) ===
baseline_window = [-200 0];  % ms
[~, b_start] = min(abs(time_ms - baseline_window(1)));
[~, b_end]   = min(abs(time_ms - baseline_window(2)));
baseline_power = mean(tf_avg(:, b_start:b_end, :), 2);  % freqs x 1 x channels
tf_db = 10 * log10(bsxfun(@rdivide, tf_avg, baseline_power));

% === Load Channel Locations ===
load('EEG_chlocs.mat');  % Variable: EEG_chlocs
chan_labels = {chanlocsEasyCapNoRef.labels};
match_idx = ismember(chan_labels, channels);
EEG_chlocs = chanlocsEasyCapNoRef(match_idx);

% === Topography Plot Every 200ms (Alpha Band 8–12Hz) ===
f_band = [8 12];
f_idx = find(frex >= f_band(1) & frex <= f_band(2));
step_ms = 200;
t0_idx = find(time_ms >= 0, 1);  % Zero time index
n_plots = 12;

figure;
cmap = jet;
clims = [-10 10];  % dB scale limits

for i = 1:n_plots
    t_idx = t0_idx + round((i-1) * (step_ms / (1000/sfreq)));
    data_topo = squeeze(mean(mean(tf_db(f_idx, t_idx, :), 1), 2));
    subplot(3, 4, i);
    topoplot(data_topo, EEG_chlocs, 'maplimits', clims, 'electrodes', 'off', 'colormap', cmap, 'style', 'map');
    title(sprintf('%d ms', round(time_ms(t_idx))));
end
set(gcf, 'color', 'w');
subtitle('Alpha Power Topographies Over Time');

% === Time-Frequency Plot for Selected Channels ===
selected_chs = {'C3','Cz','C4'};  % Example motor cortex channels
idx = find(ismember(channels, selected_chs));

% Average TF over selected channels
tf_ch_avg = squeeze(mean(tf_db(:, :, idx), 3));  % freqs x time

figure;
contourf(time_ms, frex, tf_ch_avg, 40, 'linecolor', 'none');
set(gca, 'clim', clims, 'ydir', 'normal');
xlabel('Time (ms)'); ylabel('Frequency (Hz)');
title('Time-Frequency Power (dB)'); colormap jet; colorbar;
xline(0, '--r', 'LineWidth', 1.5);
yticks([4 8 13 30 40]);
