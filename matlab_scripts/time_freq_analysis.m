%% Time-Frequency Analysis using Morlet Wavelet with Topography and Statistics
%
% This script is designed to:
% 1. Automatically loop through experimental groups and time points.
% 2. Perform time-frequency analysis for each combination.
% 3. Store results in a single structured variable.
% 4. Perform a comprehensive statistical analysis to:
%    a. Identify data-driven Regions of Interest (ROIs) based on significant
%       activity across all subjects.
%    b. Conduct within-group tests (Pre vs. Post) on these ROIs.
%    c. Conduct between-group tests on the change (Post-Pre) in these ROIs.
% 5. Generate and save comparative plots for visualization.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
clear;
close all;

%% Core Parameters
nSubject = 31;
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
baseline_window = [-0.8 -0.2]; % in seconds
baseidx = dsearchn(times', baseline_window');

%% Initialize Master Data Structure
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
            filename = sprintf('CBH%04d', sub);

            for evt = 1:nEvents
                event_name = events{evt};
                if is_pre
                    eeg_file = sprintf('%s_%s_%s.set', filename, event_name, 'pre');
                else
                    eeg_file = sprintf('%s_%s_%s.set', filename, event_name, 'post');
                end
                filepath = fullfile(eegset_dir, eeg_file);

                if ~exist(filepath, 'file')
                    fprintf('Warning: File not found, skipping: %s\n', eeg_file);
                    continue;
                end

                EEG = pop_loadset('filename', eeg_file, 'filepath', eegset_dir);
                fprintf('  Processing %s: %d trials\n', event_name, EEG.trials);

                nWave = length(wavtime);
                nData = EEG.pnts * EEG.trials;
                nConv = nWave + nData - 1;

                for ch = 1:nChannels
                    alldata = reshape(EEG.data(ch, :, :), 1, []);
                    dataX = fft(alldata, nConv);
                    for fi = 1:num_frex
                        wavelet = exp(2*1i*pi*frex(fi).*wavtime) .* exp(-wavtime.^2./(2*s(fi)^2));
                        waveletX = fft(wavelet, nConv);
                        waveletX = waveletX ./ max(waveletX);
                        as = ifft(waveletX .* dataX);
                        as = as(half_wave+1:end-half_wave);
                        as = reshape(as, EEG.pnts, EEG.trials);
                        power = mean(abs(as).^2, 2);
                        temp_tf_data.(event_name)(s_idx, fi, :, ch) = power;
                    end
                end
            end
        end

        % dB Conversion
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

        all_tf_data.(group_name).(timepoint_name) = temp_tf_db;
        fprintf('Finished analysis for %s - %s. Data stored.\n', group_name, timepoint_name);
    end
end

%% Save All Processed Data
fprintf('\nSaving all processed data...\n');
save('time_frequency_analysis_ALL_RESULTS.mat', 'all_tf_data', '-v7.3');
fprintf('Data saved to time_frequency_analysis_ALL_RESULTS.mat\n');
