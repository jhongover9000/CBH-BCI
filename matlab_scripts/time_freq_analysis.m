%%%% Time-Frequency Analysis using Morlet Wavelet
%%%% Including dB conversion and topography plotting
%%%% Conditions: MI and Rest
%%%% Time of interest: 0-2000ms with 10 time bins

% clear all;
% close all;
clc;

%% Parameters
expTitle = "";
nSubject = 19;  % Total number of subjects (CBH0001 to CBH0019)
nChannels = 60;
SR = 250;  % Sampling rate

% Define conditions
events = {
    'MI';
    'Rest';
    % 'TapStart'
};
nEvents = length(events);

% SELECT CONDITION TO ANALYZE
% Set to true for haptic (odd subjects), false for non-haptic (even subjects)
hapticsCondition = false;  % CHANGE THIS TO SELECT CONDITION

% Calculate number of subjects for selected condition
if hapticsCondition
    % Haptic: odd subjects (1, 3, 5, ..., 19) = 10 subjects
    conditional_subject_count = ceil(nSubject/2);
    fprintf('Analyzing HAPTIC condition: %d subjects (odd numbered)\n', conditional_subject_count);
else
    % Non-haptic: even subjects (2, 4, 6, ..., 18) = 9 subjects
    conditional_subject_count = floor(nSubject/2);
    fprintf('Analyzing NON-HAPTIC condition: %d subjects (even numbered)\n', conditional_subject_count);
end

% firstHalf = true;
% lastHalf = false;

% firstHalf = false;
% lastHalf = true;

firstHalf = false;
lastHalf = false;

half_condition = "";
haptic_condition = "";


% Path to epoched data
eegset_dir = './epoched/';

% Frequency parameters
min_freq = 2;
max_freq = 80;
num_frex = max_freq - min_freq;
frex = logspace(log10(min_freq), log10(max_freq), num_frex);

% Wavelet parameters
range_cycles = [2 12];
s = logspace(log10(range_cycles(1)), log10(range_cycles(2)), num_frex) ./ (2*pi*frex);
wavtime = -2:1/SR:2;
half_wave = (length(wavtime)-1)/2;

% Time parameters
epoch_period = [-2 3];  % -2 to 3 seconds
nTimes = diff(epoch_period) * SR;
times = epoch_period(1)*SR:epoch_period(2)*SR-1; % -2000 to 2999 ms

% Baseline for dB conversion
baseline_window = [-1.0*SR -0.5*SR];  % -1000ms to -500ms
baseidx = reshape(dsearchn(times', baseline_window(:)), [], 2);

%% Initialize output matrices
fprintf('Initializing output matrices...\n');
tf_MI = zeros(conditional_subject_count, length(frex), nTimes, nChannels, 'single');
tf_Rest = zeros(conditional_subject_count, length(frex), nTimes, nChannels, 'single');

%% Time-Frequency Analysis
fprintf('Starting time-frequency analysis...\n');

% Counter for indexing into output arrays
cond_sub_idx = 0;

for sub = 1:nSubject
    
    % Skip subjects based on condition selection
    if hapticsCondition
        % For haptic condition, process only odd subjects
        if mod(sub, 2) == 0
            fprintf('Skipping subject %d (non-haptic)\n', sub);
            continue;
        end
    else
        % For non-haptic condition, process only even subjects
        if mod(sub, 2) == 1
            fprintf('Skipping subject %d (haptic)\n', sub);
            continue;
        end
    end
    
    % Increment conditional subject index
    cond_sub_idx = cond_sub_idx + 1;
    fprintf('\nProcessing Subject %d (CBH%04d) - Conditional index: %d/%d\n', ...
        sub, sub, cond_sub_idx, conditional_subject_count);
    
    % Format filename
    if (sub < 10)
        filename = ['CBH000' int2str(sub)];
    else
        filename = ['CBH00' int2str(sub)];
    end
    
    % Process each condition
    for evt = 1:nEvents
        
        % Load epoched data
        if firstHalf
            eeg_file = sprintf('%s_%s_%s.set', filename, events{evt},'pre');
        elseif lastHalf
            eeg_file = sprintf('%s_%s_%s.set', filename, events{evt},'post');
        else
            eeg_file = sprintf('%s_%s_%s.set', filename, events{evt},'all');
        end

        filepath = fullfile(eegset_dir, eeg_file);
        
        if ~exist(filepath, 'file')
            fprintf('Warning: File not found for subject %d, condition %s\n', sub, events{evt});
            continue;
        end
        
        % Channels x timepoints x trials
        EEG = pop_loadset('filename', eeg_file, 'filepath', eegset_dir);
        fprintf('  Processing %s: %d trials\n', events{evt}, EEG.trials);
        
        % FFT parameters
        nWave = length(wavtime);
        nData = EEG.pnts * EEG.trials;
        nConv = nWave + nData - 1;
        
        % Process each channel
        for ch = 1:nChannels
            
            % Get channel data (all trials concatenated)
            alldata = reshape(EEG.data(ch, :, :), 1, []);
            dataX = fft(alldata, nConv);
            
            % Process each frequency
            for fi = 1:length(frex)
                
                % Create wavelet
                wavelet = exp(2*1i*pi*frex(fi).*wavtime) .* exp(-wavtime.^2./(2*s(fi)^2));
                waveletX = fft(wavelet, nConv);
                waveletX = waveletX ./ max(waveletX);
                
                % Convolution
                as = ifft(waveletX .* dataX);
                as = as(half_wave+1:end-half_wave);
                
                % Reshape to trials
                as = reshape(as, EEG.pnts, EEG.trials);
                
                % Compute power and average over trials
                % Use cond_sub_idx instead of sub for array indexing
                if evt == 1  % MI
                    tf_MI(cond_sub_idx, fi, :, ch) = mean(abs(as).^2, 2);
                else  % Rest
                    tf_Rest(cond_sub_idx, fi, :, ch) = mean(abs(as).^2, 2);
                end
            end
        end
    end
end

%% Convert to dB
fprintf('\nConverting to dB...\n');

tf_MI_db = zeros(size(tf_MI), 'single');
tf_Rest_db = zeros(size(tf_Rest), 'single');

for sub = 1:conditional_subject_count
    
    for ch = 1:nChannels
        
        % Get the power data for this subject and channel
        power_MI = squeeze(tf_MI(sub, :, :, ch));  % [num_frex x nTimes]
        power_Rest = squeeze(tf_Rest(sub, :, :, ch));  % [num_frex x nTimes]
        
        % Get baseline for each condition
        baseline_MI = mean(tf_MI(sub, :, baseidx(1):baseidx(2), ch), 3)';  % [num_frex x 1]
        baseline_Rest = mean(tf_Rest(sub, :, baseidx(1):baseidx(2), ch), 3)';  % [num_frex x 1]
        
        % Convert to dB using bsxfun for proper broadcasting
        tf_MI_db(sub, :, :, ch) = 10*log10(bsxfun(@rdivide, power_MI, baseline_MI));
        tf_Rest_db(sub, :, :, ch) = 10*log10(bsxfun(@rdivide, power_Rest, baseline_Rest));
    end
    
end

%% Save processed data with condition label

% Check Half Condition
if firstHalf
    half_condition = "Pre";
elseif lastHalf
    half_condition = "Post";
else
    half_condition = "";
end

% Check Haptics Condition
if hapticsCondition
    haptic_condition = "H";
else
    haptic_condition = "NH";
end

% Save Data
fprintf('Saving processed data...\n');
save_name_1 = sprintf('%s_%s_%s_%s.mat', 'tf_db', events{1}, haptic_condition, half_condition);
save(save_name_1, 'tf_MI_db', '-v7.3');

save_name_2 = sprintf('%s_%s_%s_%s.mat', 'tf_db', events{2}, haptic_condition, half_condition);
save(save_name_2, 'tf_MI_db', '-v7.3');

%% Topography plots - 10 time bins across 0-2000ms
fprintf('Creating topography plots...\n');

% Load channel locations
load reference/EEG_chlocs_60.mat

% Define time windows for topography (10 bins from 0 to 2000ms)
topo_start = -125;  % 0ms
topo_end = 3*SR;  % 2000ms
bin_width = 125;
n_bins = (topo_end - topo_start) / bin_width;

% Frequency band of interest (e.g., alpha band 8-13 Hz)
freq_band = [8 30];
freq_idx = find(frex >= freq_band(1) & frex <= freq_band(2));

% Color limits for topography
clim = [-2 2];

% Create figure for topography
figure('Position', [100 100 1400 600]);
if hapticsCondition
    condition_str = 'Haptic';
else
    condition_str = 'Non-Haptic';
end

sgtitle(sprintf('Topography (%s): %d to %d Hz', condition_str, freq_band(1), freq_band(2)), 'FontSize', 16);

% MI condition
for i = 1:n_bins
    subplot(2, n_bins, i);
    
    % Define time window
    t_start = topo_start + (i-1) * bin_width;
    t_end = topo_start + i * bin_width;
    t_idx = find(times >= t_start & times < t_end);
    
    % Average over subjects, frequencies, and time window
    topo_data = squeeze(mean(mean(mean(tf_MI_db(1, freq_idx, t_idx, :), 1), 2), 3))';
    
    % Plot topography
    topoplot(topo_data, EEG_chlocs, ...
        'maplimits', clim, ...
        'electrodes', 'off', ...
        'colormap', jet, ...
        'style', 'both');
    
    title(sprintf('MI: %d-%d ms', round(t_start* 1000/SR), round(t_end* 1000/SR)), 'FontSize', 10);
end

% Rest condition
for i = 1:n_bins
    subplot(2, n_bins, i + n_bins);
    
    % Define time window
    t_start = topo_start + (i-1) * bin_width;
    t_end = topo_start + i * bin_width;
    t_idx = find(times >= t_start & times < t_end);
    
    % Average over subjects, frequencies, and time window
    topo_data = squeeze(mean(mean(mean(tf_Rest_db(1, freq_idx, t_idx, :), 1), 2), 3))';
    
    % Plot topography
    topoplot(topo_data, EEG_chlocs, ...
        'maplimits', clim, ...
        'electrodes', 'off', ...
        'colormap', jet, ...
        'style', 'both');
    
    title(sprintf('Rest: %d-%d ms', round(t_start* 1000/SR), round(t_end* 1000/SR)), 'FontSize', 10);
end

colorbar('Position', [0.92 0.25 0.02 0.5]);
set(gcf, 'color', 'w');

%% Time-Frequency plots for specific channels
fprintf('Creating time-frequency plots...\n');

% Define channels of interest (e.g., motor areas)
channels_of_interest = {'C3'};

% Find channel indices
ch_idx = [];
for i = 1:length(channels_of_interest)
    idx = find(strcmp({EEG_chlocs.labels}, channels_of_interest{i}));
    if ~isempty(idx)
        ch_idx = [ch_idx idx];
    end
end

% Create TF plots
figure('Position', [100 100 1200 800]);
sgtitle(sprintf('Time-Frequency Analysis (%s)', condition_str), 'FontSize', 14);
colormap(jet);

% MI condition
subplot(2,2,1);
tf_data = squeeze(mean(mean(tf_MI_db(:, :, :, ch_idx), 1), 4));
contourf(times, frex, tf_data, 40, 'linecolor', 'none');
set(gca, 'clim', [-2 2], 'ydir', 'normal', 'xlim', [-500 750], 'yscale', 'log');
ylim([2 50]);
xlabel('Time (timepoints)');
ylabel('Frequency (Hz)');
title('MI - Motor Channels');
xline(0, '-.r', 'LineWidth', 2);
yticks([4 8 13 30 50]);
colorbar;

% Rest condition
subplot(2,2,2);
tf_data = squeeze(mean(mean(tf_Rest_db(:, :, :, ch_idx), 1), 4));
contourf(times, frex, tf_data, 40, 'linecolor', 'none');
set(gca, 'clim', [-2 2], 'ydir', 'normal', 'xlim', [-500 750], 'yscale', 'log');
ylim([2 50]);
xlabel('Time (timepoints)');
ylabel('Frequency (Hz)');
title('Rest - Motor Channels');
xline(0, '-.r', 'LineWidth', 2);
yticks([4 8 13 30 50]);
colorbar;

% Difference (MI - Rest)
subplot(2,2,3);
tf_diff = squeeze(mean(mean(tf_MI_db(:, :, :, ch_idx), 1), 4)) - ...
          squeeze(mean(mean(tf_Rest_db(:, :, :, ch_idx), 1), 4));
contourf(times, frex, tf_diff, 40, 'linecolor', 'none');
set(gca, 'clim', [-1 1], 'ydir', 'normal', 'xlim', [-500 750], 'yscale', 'log');
ylim([2 50]);
xlabel('Time (timepoints)');
ylabel('Frequency (Hz)');
title('MI - Rest Difference');
xline(0, '-.r', 'LineWidth', 2);
yticks([4 8 13 30 50]);
colorbar;

set(gcf, 'color', 'w');

%% Power over time for specific frequency bands
fprintf('Creating power time series plots...\n');

figure('Position', [100 100 1000 600]);
sgtitle(sprintf('Power Time Series (%s)', condition_str), 'FontSize', 14);

% Define frequency bands
bands = {'Theta', [4 8]; 'Alpha', [8 13]; 'Beta', [13 30]; 'Gamma', [30 50]};

for b = 1:length(bands)
    subplot(2, 2, b);
    
    % Find frequency indices for band
    band_idx = find(frex >= bands{b, 2}(1) & frex <= bands{b, 2}(2));
    
    % Average over subjects, frequencies in band, and channels
    power_MI = squeeze(mean(mean(mean(tf_MI_db(:, band_idx, :, ch_idx), 1), 2), 4));
    power_Rest = squeeze(mean(mean(mean(tf_Rest_db(:, band_idx, :, ch_idx), 1), 2), 4));
    
    % Calculate SEM (if multiple subjects)
    if conditional_subject_count > 1
        sem_MI = squeeze(std(mean(mean(tf_MI_db(:, band_idx, :, ch_idx), 2), 4), [], 1)) / sqrt(conditional_subject_count);
        sem_Rest = squeeze(std(mean(mean(tf_Rest_db(:, band_idx, :, ch_idx), 2), 4), [], 1)) / sqrt(conditional_subject_count);
    else
        sem_MI = zeros(size(power_MI));
        sem_Rest = zeros(size(power_Rest));
    end
    
    % Plot with error bars
    hold on;
    h1 = plot(times, power_MI, 'b-', 'LineWidth', 2);
    h2 = plot(times, power_Rest, 'r-', 'LineWidth', 2);
    
    % Add shaded error bars (uncomment if desired)
    % fill([times fliplr(times)], [power_MI+sem_MI fliplr(power_MI-sem_MI)], 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    % fill([times fliplr(times)], [power_Rest+sem_Rest fliplr(power_Rest-sem_Rest)], 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    
    xlim([-500 750]);
    xlabel('Time (timepoints)');
    ylabel('Power (dB)');
    title(sprintf('%s Band (%d-%d Hz)', bands{b, 1}, bands{b, 2}(1), bands{b, 2}(2)));
    xline(0, 'k--');
    yline(0, 'k:');
    legend([h1 h2], {'MI', 'Rest'}, 'Location', 'best');
    grid on;
end

set(gcf, 'color', 'w');

fprintf('\nAnalysis complete for %s condition!\n', condition_str);