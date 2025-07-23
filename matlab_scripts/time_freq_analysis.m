%% Time-Frequency Analysis using Morlet Wavelet with Topography
%
% MODIFIED SCRIPT V2
%
% This script is designed to:
% 1. Automatically loop through different experimental groups (Haptic, Non-Haptic)
%    and time points (Pre, Post).
% 2. Perform time-frequency analysis for each combination.
% 3. Store all results in a single structured variable.
% 4. Generate plots that directly compare groups and time points for each
%    event type, including both time-frequency spectrograms and scalp
%    topographies.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
clear;
close all;

%% Core Parameters
nSubject = 27;
nChannels = 60;
SR = 250;  % Sampling rate

% Define event types to analyze
events = {
    'MI';
    'Rest';
    % 'TapStart' % Uncomment to include this event
};
nEvents = length(events);

% Define the groups and timepoints to iterate through
groups = {'Haptic', 'NonHaptic'};
timepoints = {'Pre', 'Post'};

% Path to epoched data
eegset_dir = './epoched/';

% Load channel locations (needed for plotting)
load reference/EEG_chlocs_60.mat

%% Analysis Parameters (Frequency, Wavelet, Time)

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
epoch_period = [-3 4];
times = linspace(epoch_period(1), epoch_period(2), diff(epoch_period) * SR);
nTimes = length(times);

% Baseline for dB conversion
baseline_window = [-0.6 -0.1]; % in seconds
baseidx = dsearchn(times', baseline_window');

%% Initialize Master Data Structure
% This structure will hold all the results
all_tf_data = struct();
fprintf('Master data structure initialized.\n');

%% Main Analysis Loop
% This loop iterates over groups (Haptic/Non-Haptic) and timepoints (Pre/Post)

for g = 1:length(groups)
    group_name = groups{g};
    is_haptic = strcmp(group_name, 'Haptic');

    for t = 1:length(timepoints)
        timepoint_name = timepoints{t};
        is_pre = strcmp(timepoint_name, 'Pre');

        fprintf('\n============================================================\n');
        fprintf('Starting Analysis for Group: %s, Timepoint: %s\n', group_name, timepoint_name);
        fprintf('============================================================\n');

        % Determine which subjects to process for this group
        if is_haptic
            subject_list = 1:2:nSubject; % Odd subjects
        else
            subject_list = 2:2:nSubject; % Even subjects
        end
        n_group_subjects = length(subject_list);

        % Initialize temporary storage for the current group/timepoint
        temp_tf_data = struct();
        for evt = 1:nEvents
            temp_tf_data.(events{evt}) = zeros(n_group_subjects, num_frex, nTimes, nChannels, 'single');
        end

        % Loop through subjects in the current group
        for s_idx = 1:n_group_subjects
            sub = subject_list(s_idx);

            fprintf('\nProcessing Subject %d (CBH%04d) - Index %d/%d\n', sub, sub, s_idx, n_group_subjects);

            % Format filename
            filename = sprintf('CBH%04d', sub);

            % Process each event type
            for evt = 1:nEvents
                event_name = events{evt};

                % Construct the file name based on timepoint
                if is_pre
                    eeg_file = sprintf('%s_%s_%s.set', filename, event_name, 'pre');
                else % Post
                    eeg_file = sprintf('%s_%s_%s.set', filename, event_name, 'post');
                end
                filepath = fullfile(eegset_dir, eeg_file);

                if ~exist(filepath, 'file')
                    fprintf('Warning: File not found, skipping: %s\n', eeg_file);
                    continue;
                end

                % Load EEG data
                EEG = pop_loadset('filename', eeg_file, 'filepath', eegset_dir);
                fprintf('  Processing %s: %d trials\n', event_name, EEG.trials);

                % FFT parameters
                nWave = length(wavtime);
                nData = EEG.pnts * EEG.trials;
                nConv = nWave + nData - 1;

                % Analyze each channel
                for ch = 1:nChannels
                    alldata = reshape(EEG.data(ch, :, :), 1, []);
                    dataX = fft(alldata, nConv);

                    % Convolve with wavelets for each frequency
                    for fi = 1:num_frex
                        wavelet = exp(2*1i*pi*frex(fi).*wavtime) .* exp(-wavtime.^2./(2*s(fi)^2));
                        waveletX = fft(wavelet, nConv);
                        waveletX = waveletX ./ max(waveletX);

                        as = ifft(waveletX .* dataX);
                        as = as(half_wave+1:end-half_wave);
                        as = reshape(as, EEG.pnts, EEG.trials);

                        % Compute power and store it
                        power = mean(abs(as).^2, 2);
                        temp_tf_data.(event_name)(s_idx, fi, :, ch) = power;
                    end
                end
            end
        end

        % dB Conversion for the current group/timepoint
        fprintf('\nConverting to dB for %s - %s...\n', group_name, timepoint_name);
        temp_tf_db = struct();
        for evt = 1:nEvents
            event_name = events{evt};
            power_data = temp_tf_data.(event_name);
            db_data = zeros(size(power_data), 'single');

            for s_idx = 1:n_group_subjects
                for ch = 1:nChannels
                    baseline_power = mean(power_data(s_idx, :, baseidx(1):baseidx(2), ch), 3);
                    db_data(s_idx, :, :, ch) = 10*log10(bsxfun(@rdivide, squeeze(power_data(s_idx, :, :, ch)), baseline_power'));
                end
            end
            temp_tf_db.(event_name) = db_data;
        end

        % Store the processed dB data in the master structure
        all_tf_data.(group_name).(timepoint_name) = temp_tf_db;
        fprintf('Finished analysis for %s - %s. Data stored.\n', group_name, timepoint_name);

    end % End of timepoint loop
end % End of group loop


%% Save All Processed Data
fprintf('\nSaving all processed data...\n');
save('time_frequency_analysis_ALL_RESULTS.mat', 'all_tf_data', '-v7.3');
fprintf('Data saved to time_frequency_analysis_ALL_RESULTS.mat\n');


%% PLOTTING SECTION
% This section generates comparative plots with topographies
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\nCreating comparative plots...\n');

% --- Plotting Parameters ---
tf_clim = [-2 2];
topo_clim = [-2 2];
diff_clim = [-1 1];
channels_of_interest = {'C3', 'C4', 'Cz'}; % For TF plots

% --- Topography Parameters ---
% Define frequency bands of interest for topographies
freq_bands = {
    'Alpha', [8 13];
    'Beta', [13 30]
};
% Define time window for topographies
topo_time_window = [0.5 1.5]; % in seconds

% Find indices for channels, frequencies, and times
ch_idx = find(ismember({EEG_chlocs.labels}, channels_of_interest));
topo_time_idx = dsearchn(times', topo_time_window');
fprintf('Plotting for channels: %s\n', strjoin(channels_of_interest, ', '));
fprintf('Topography time window: %.2f to %.2f s\n', times(topo_time_idx(1)), times(topo_time_idx(2)));

% --- Main Plotting Loop ---
% Iterate through each event type to create a set of plots
for evt = 1:nEvents
    event_name = events{evt};

    % Iterate through each defined frequency band
    for fb = 1:size(freq_bands, 1)
        band_name = freq_bands{fb, 1};
        band_range = freq_bands{fb, 2};
        freq_idx = dsearchn(frex', band_range');
        fprintf('  Generating plots for %s band (%.1f-%.1f Hz)\n', band_name, frex(freq_idx(1)), frex(freq_idx(2)));

        % --- FIGURE 1: Haptic vs Non-Haptic Comparison ---
        for t = 1:length(timepoints)
            timepoint_name = timepoints{t};
            figure('Position', [100 100 1500 900], 'color', 'w');
            sgtitle(sprintf('Group Comparison (%s at %s): Haptic vs. Non-Haptic [%s Band]', event_name, timepoint_name, band_name), 'FontSize', 16, 'FontWeight', 'bold');

            % Get data for both groups at the current timepoint
            data_haptic = all_tf_data.Haptic.(timepoint_name).(event_name);
            data_nonhaptic = all_tf_data.NonHaptic.(timepoint_name).(event_name);

            % --- Haptic TF Plot (Top Left) ---
            subplot(2,3,1);
            tf_haptic_mean = squeeze(mean(mean(data_haptic(:, :, :, ch_idx), 1), 4));
            contourf(times, frex, tf_haptic_mean, 40, 'linecolor', 'none');
            set(gca, 'clim', tf_clim, 'ydir', 'normal', 'xlim', [-0.5 2], 'yscale', 'log');
            ylim([2 50]); yticks([4 8 13 30 50]);
            title('Haptic Group TF'); xlabel('Time (s)'); ylabel('Frequency (Hz)');
            xline(0, '-.r', 'LineWidth', 2); colorbar;

            % --- Non-Haptic TF Plot (Top Middle) ---
            subplot(2,3,2);
            tf_nonhaptic_mean = squeeze(mean(mean(data_nonhaptic(:, :, :, ch_idx), 1), 4));
            contourf(times, frex, tf_nonhaptic_mean, 40, 'linecolor', 'none');
            set(gca, 'clim', tf_clim, 'ydir', 'normal', 'xlim', [-0.5 2], 'yscale', 'log');
            ylim([2 50]); yticks([4 8 13 30 50]);
            title('Non-Haptic Group TF'); xlabel('Time (s)');
            xline(0, '-.r', 'LineWidth', 2); colorbar;

            % --- Difference TF Plot (Top Right) ---
            subplot(2,3,3);
            tf_diff = tf_haptic_mean - tf_nonhaptic_mean;
            contourf(times, frex, tf_diff, 40, 'linecolor', 'none');
            set(gca, 'clim', diff_clim, 'ydir', 'normal', 'xlim', [-0.5 2], 'yscale', 'log');
            ylim([2 50]); yticks([4 8 13 30 50]);
            title('Difference: Haptic - Non-Haptic'); xlabel('Time (s)');
            xline(0, '-.r', 'LineWidth', 2); colorbar;

            % --- Haptic Topography (Bottom Left) ---
            subplot(2,3,4);
            topo_haptic = squeeze(mean(mean(mean(data_haptic(:, freq_idx(1):freq_idx(2), topo_time_idx(1):topo_time_idx(2), :), 1), 2), 3));
            topoplot(topo_haptic, EEG_chlocs, 'maplimits', topo_clim, 'style', 'map', 'electrodes', 'ptslabels');
            title(sprintf('Haptic Topo (%s)', band_name)); colorbar;

            % --- Non-Haptic Topography (Bottom Middle) ---
            subplot(2,3,5);
            topo_nonhaptic = squeeze(mean(mean(mean(data_nonhaptic(:, freq_idx(1):freq_idx(2), topo_time_idx(1):topo_time_idx(2), :), 1), 2), 3));
            topoplot(topo_nonhaptic, EEG_chlocs, 'maplimits', topo_clim, 'style', 'map', 'electrodes', 'ptslabels');
            title(sprintf('Non-Haptic Topo (%s)', band_name)); colorbar;

            % --- Difference Topography (Bottom Right) ---
            subplot(2,3,6);
            topo_diff = topo_haptic - topo_nonhaptic;
            topoplot(topo_diff, EEG_chlocs, 'maplimits', diff_clim, 'style', 'map', 'electrodes', 'ptslabels');
            title(sprintf('Difference Topo (%s)', band_name)); colorbar;
        end

        % --- FIGURE 2: Pre vs. Post Training Comparison ---
        for g = 1:length(groups)
            group_name = groups{g};
            figure('Position', [150 150 1500 900], 'color', 'w');
            sgtitle(sprintf('Training Comparison (%s for %s Group): Pre vs. Post [%s Band]', event_name, group_name, band_name), 'FontSize', 16, 'FontWeight', 'bold');

            % Get data for both timepoints for the current group
            data_pre = all_tf_data.(group_name).Pre.(event_name);
            data_post = all_tf_data.(group_name).Post.(event_name);

            % --- Pre-Training TF Plot (Top Left) ---
            subplot(2,3,1);
            tf_pre_mean = squeeze(mean(mean(data_pre(:, :, :, ch_idx), 1), 4));
            contourf(times, frex, tf_pre_mean, 40, 'linecolor', 'none');
            set(gca, 'clim', tf_clim, 'ydir', 'normal', 'xlim', [-0.5 2], 'yscale', 'log');
            ylim([2 50]); yticks([4 8 13 30 50]);
            title('Pre-Training TF'); xlabel('Time (s)'); ylabel('Frequency (Hz)');
            xline(0, '-.r', 'LineWidth', 2); colorbar;

            % --- Post-Training TF Plot (Top Middle) ---
            subplot(2,3,2);
            tf_post_mean = squeeze(mean(mean(data_post(:, :, :, ch_idx), 1), 4));
            contourf(times, frex, tf_post_mean, 40, 'linecolor', 'none');
            set(gca, 'clim', tf_clim, 'ydir', 'normal', 'xlim', [-0.5 2], 'yscale', 'log');
            ylim([2 50]); yticks([4 8 13 30 50]);
            title('Post-Training TF'); xlabel('Time (s)');
            xline(0, '-.r', 'LineWidth', 2); colorbar;

            % --- Difference TF Plot (Top Right) ---
            subplot(2,3,3);
            tf_diff = tf_post_mean - tf_pre_mean;
            contourf(times, frex, tf_diff, 40, 'linecolor', 'none');
            set(gca, 'clim', diff_clim, 'ydir', 'normal', 'xlim', [-0.5 2], 'yscale', 'log');
            ylim([2 50]); yticks([4 8 13 30 50]);
            title('Difference: Post - Pre'); xlabel('Time (s)');
            xline(0, '-.r', 'LineWidth', 2); colorbar;

            % --- Pre-Training Topography (Bottom Left) ---
            subplot(2,3,4);
            topo_pre = squeeze(mean(mean(mean(data_pre(:, freq_idx(1):freq_idx(2), topo_time_idx(1):topo_time_idx(2), :), 1), 2), 3));
            topoplot(topo_pre, EEG_chlocs, 'maplimits', topo_clim, 'style', 'map', 'electrodes', 'ptslabels');
            title(sprintf('Pre-Training Topo (%s)', band_name)); colorbar;

            % --- Post-Training Topography (Bottom Middle) ---
            subplot(2,3,5);
            topo_post = squeeze(mean(mean(mean(data_post(:, freq_idx(1):freq_idx(2), topo_time_idx(1):topo_time_idx(2), :), 1), 2), 3));
            topoplot(topo_post, EEG_chlocs, 'maplimits', topo_clim, 'style', 'map', 'electrodes', 'ptslabels');
            title(sprintf('Post-Training Topo (%s)', band_name)); colorbar;

            % --- Difference Topography (Bottom Right) ---
            subplot(2,3,6);
            topo_diff = topo_post - topo_pre;
            topoplot(topo_diff, EEG_chlocs, 'maplimits', diff_clim, 'style', 'map', 'electrodes', 'ptslabels');
            title(sprintf('Difference Topo (%s)', band_name)); colorbar;
        end
    end
end

fprintf('\nAnalysis and plotting complete!\n');
