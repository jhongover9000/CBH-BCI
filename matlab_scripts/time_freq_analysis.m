%% Time-Frequency Analysis using Morlet Wavelet with Topography and Statistics
%
% MODIFIED SCRIPT V5 (Corrected)
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

%% STATISTICAL ANALYSIS SECTION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n============================================================\n');
fprintf('STARTING STATISTICAL ANALYSIS\n');
fprintf('============================================================\n');

% --- Statistical Parameters ---
stat_alpha = 0.05; % Alpha level for significance
n_haptic_subjects = size(all_tf_data.Haptic.Pre.MI, 1);
n_nonhaptic_subjects = size(all_tf_data.NonHaptic.Pre.MI, 1);

% This structure will hold all statistical results
all_stats = struct();

for evt = 1:nEvents
    event_name = events{evt};
    fprintf('\n--- Analyzing Event: %s ---\n', event_name);

    % --- Step 1: Identify Data-Driven Regions of Interest (ROIs) ---
    fprintf('Step 1: Identifying data-driven ROI for %s...\n', event_name);
    
    combined_pre_data = [all_tf_data.Haptic.Pre.(event_name); all_tf_data.NonHaptic.Pre.(event_name)];
    
    [~, pvals, ~, stats] = ttest(combined_pre_data);
    t_map = squeeze(stats.tstat);
    p_map = squeeze(pvals);
    
    % --- FIX IS HERE ---
    % Use a robust FDR function that returns a logical mask directly
    p_vector = reshape(p_map, [], 1);
    h_mask_vector = fdr_bh(p_vector, stat_alpha); % h is a logical vector
    significant_mask = reshape(h_mask_vector, size(p_map)); % Reshape the logical vector
    
    all_stats.(event_name).ROI_mask = significant_mask;
    all_stats.(event_name).ROI_tmap = t_map .* significant_mask;
    
    fprintf('  ROI identified. Found %d significant pixels.\n', sum(significant_mask(:)));

    if sum(significant_mask(:)) == 0
        fprintf('  WARNING: No significant ROI found for %s. Skipping further stats for this event.\n', event_name);
        continue;
    end

    % --- Step 2: Within-Group Comparisons (Pre vs. Post) in the ROI ---
    fprintf('Step 2: Performing within-group (Pre vs. Post) tests...\n');
    for g = 1:length(groups)
        group_name = groups{g};
        
        pre_data = all_tf_data.(group_name).Pre.(event_name);
        post_data = all_tf_data.(group_name).Post.(event_name);
        
        n_group_subjects = size(pre_data, 1);
        pre_roi_vals = zeros(n_group_subjects, 1);
        post_roi_vals = zeros(n_group_subjects, 1);
        
        for s_idx = 1:n_group_subjects
            subj_pre_data = squeeze(pre_data(s_idx, :, :, :));
            % The following line now works because significant_mask is logical
            pre_roi_vals(s_idx) = mean(subj_pre_data(significant_mask));
            
            subj_post_data = squeeze(post_data(s_idx, :, :, :));
            post_roi_vals(s_idx) = mean(subj_post_data(significant_mask));
        end
        
        [h, p, ~, stat] = ttest(post_roi_vals, pre_roi_vals);
        
        all_stats.(event_name).within_group.(group_name).p_value = p;
        all_stats.(event_name).within_group.(group_name).t_stat = stat.tstat;
        all_stats.(event_name).within_group.(group_name).df = stat.df;
        
        fprintf('  %s Group (Pre vs. Post): t(%d) = %.3f, p = %.4f', group_name, stat.df, stat.tstat, p);
        if h, fprintf(' (SIGNIFICANT)\n'); else, fprintf(' (not significant)\n'); end
    end
    
    % --- Step 3: Between-Group Comparison of Change (Post-Pre Delta) ---
    fprintf('Step 3: Performing between-group test on training effect (Post-Pre)...\n');
    
    delta_haptic_data = all_tf_data.Haptic.Post.(event_name) - all_tf_data.Haptic.Pre.(event_name);
    delta_haptic_roi_vals = zeros(n_haptic_subjects, 1);
    for s_idx = 1:n_haptic_subjects
        subj_delta_data = squeeze(delta_haptic_data(s_idx, :, :, :));
        delta_haptic_roi_vals(s_idx) = mean(subj_delta_data(significant_mask));
    end
    
    delta_nonhaptic_data = all_tf_data.NonHaptic.Post.(event_name) - all_tf_data.NonHaptic.Pre.(event_name);
    delta_nonhaptic_roi_vals = zeros(n_nonhaptic_subjects, 1);
    for s_idx = 1:n_nonhaptic_subjects
        subj_delta_data = squeeze(delta_nonhaptic_data(s_idx, :, :, :));
        delta_nonhaptic_roi_vals(s_idx) = mean(subj_delta_data(significant_mask));
    end
    
    [h, p, ~, stat] = ttest2(delta_haptic_roi_vals, delta_nonhaptic_roi_vals);
    
    all_stats.(event_name).between_group.p_value = p;
    all_stats.(event_name).between_group.t_stat = stat.tstat;
    all_stats.(event_name).between_group.df = stat.df;
    
    fprintf('  Between-Group (Haptic Delta vs. Non-Haptic Delta): t(%d) = %.3f, p = %.4f', stat.df, stat.tstat, p);
    if h, fprintf(' (SIGNIFICANT)\n'); else, fprintf(' (not significant)\n'); end
end

fprintf('\n============================================================\n');
fprintf('STATISTICAL ANALYSIS COMPLETE\n');
fprintf('============================================================\n');

%% PLOTTING SECTION
% This section is left unchanged but will run after the corrected stats
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\nCreating comparative plots...\n');
% (Plotting code remains the same as your previous version)
% ...

%% NEW, ROBUST FDR Function
% Benjamini-Hochberg FDR procedure
% Returns a logical vector 'h' where h=true for p-values that are
% significant after FDR correction.
function h = fdr_bh(pvals, q)
    % Ensure p-values are a column vector and remove NaNs
    pvals = pvals(~isnan(pvals));
    pvals = pvals(:);
    
    % Sort p-values in ascending order
    [sorted_pvals, sort_idx] = sort(pvals);
    
    V = length(sorted_pvals); % Number of tests
    I = (1:V)'; % Vector of ranks
    
    % Find the largest p-value that is smaller than its BH-corrected value
    p_threshold_idx = find(sorted_pvals <= (I./V)*q, 1, 'last');
    
    if isempty(p_threshold_idx)
        crit_p = 0;
    else
        crit_p = sorted_pvals(p_threshold_idx);
    end
    
    % The logical mask h indicates which of the original p-values are significant
    h = pvals <= crit_p;
end
