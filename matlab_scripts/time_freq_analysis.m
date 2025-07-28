%% Time-Frequency Analysis using Morlet Wavelet with Topography and Statistics
%
% MODIFIED SCRIPT V6
%
% This script is designed to:
% 1. Automatically loop through experimental groups and time points.
% 2. Perform time-frequency analysis for each combination.
% 3. Store results in a single structured variable.
% 4. Perform a comprehensive statistical analysis to:
%    a. Identify data-driven Regions of Interest (ROIs) as significant CHANNELS.
%    b. Conduct within-group (Pre vs. Post) statistical tests across TIME.
%    c. Conduct between-group statistical tests on the change (Post-Pre) across TIME.
% 5. Generate and save visualizations of both the data and the statistical results.
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

%% Main Analysis Loop (Data Processing)
% This section remains the same, processing data and saving it.
% ... (The data processing loop from the previous version is assumed to be here) ...
% For brevity, the original data processing loop is omitted. 
% We will load the processed data instead.
if ~exist('time_frequency_analysis_ALL_RESULTS.mat', 'file')
    error('Results file not found. Please run the full data processing loop first.');
else
    fprintf('Loading pre-processed data from time_frequency_analysis_ALL_RESULTS.mat...\n');
    load('time_frequency_analysis_ALL_RESULTS.mat');
end


%% STATISTICAL ANALYSIS SECTION (CHANNEL AND TEMPORAL)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n============================================================\n');
fprintf('STARTING CHANNEL-TEMPORAL STATISTICAL ANALYSIS\n');
fprintf('============================================================\n');

% --- Statistical Parameters ---
stat_alpha = 0.05; % Alpha level for significance

% --- Parameters for ROI definition and subsequent tests ---
% We will focus the analysis on a specific frequency band
stat_band_name = 'Alpha';
stat_band_range = [8 13];
stat_freq_idx = dsearchn(frex', stat_band_range');

% Time window for identifying ROI channels (e.g., during MI)
roi_time_window = [0.5 2.5]; 
roi_time_idx = dsearchn(times', roi_time_window');

% This structure will hold all statistical results
all_stats = struct();

for evt = 1:nEvents
    event_name = events{evt};
    fprintf('\n--- Analyzing Event: %s ---\n', event_name);

    % --- Step 1: Identify Data-Driven ROI (Significant Channels) ---
    fprintf('Step 1: Identifying ROI channels for %s in the %s band...\n', event_name, stat_band_name);
    
    % Combine 'Pre' data from both groups
    combined_pre_data = [all_tf_data.Haptic.Pre.(event_name); all_tf_data.NonHaptic.Pre.(event_name)];
    
    % Average power within the defined time-frequency window for each channel and subject
    roi_tf_window_data = squeeze(mean(mean(combined_pre_data(:, stat_freq_idx(1):stat_freq_idx(2), roi_time_idx(1):roi_time_idx(2), :), 2), 3));
    
    % Perform a one-sample t-test for each channel against 0 (baseline)
    [~, pvals, ~, stats] = ttest(roi_tf_window_data);
    
    % Correct for multiple comparisons across channels using FDR
    roi_mask = fdr_bh(pvals, stat_alpha); % roi_mask is a logical vector of length nChannels
    
    % Store the ROI channels
    all_stats.(event_name).ROI_channels_mask = roi_mask;
    all_stats.(event_name).ROI_channel_names = {EEG_chlocs(roi_mask).labels};
    
    fprintf('  ROI identified. Found %d significant channels: %s\n', sum(roi_mask), strjoin(all_stats.(event_name).ROI_channel_names, ', '));

    if sum(roi_mask) == 0
        fprintf('  WARNING: No significant ROI channels found for %s. Skipping further stats for this event.\n', event_name);
        continue;
    end

    % --- Step 2: Within-Group Comparisons (Pre vs. Post) Across Time ---
    fprintf('Step 2: Performing within-group (Pre vs. Post) tests across time...\n');
    for g = 1:length(groups)
        group_name = groups{g};
        
        % Get data, average across the Beta band and the ROI channels
        pre_data_ts = squeeze(mean(mean(all_tf_data.(group_name).Pre.(event_name)(:, stat_freq_idx(1):stat_freq_idx(2), :, roi_mask), 2), 4));
        post_data_ts = squeeze(mean(mean(all_tf_data.(group_name).Post.(event_name)(:, stat_freq_idx(1):stat_freq_idx(2), :, roi_mask), 2), 4));
        
        % Perform paired t-test at each time point
        p_timecourse = zeros(1, nTimes);
        t_timecourse = zeros(1, nTimes);
        for t_idx = 1:nTimes
            [~, p_timecourse(t_idx), ~, temp_stat] = ttest(post_data_ts(:, t_idx), pre_data_ts(:, t_idx));
            t_timecourse(t_idx) = temp_stat.tstat;
        end
        
        % Correct for multiple comparisons across time
        sig_time_mask = fdr_bh(p_timecourse, stat_alpha);
        
        % Store results
        all_stats.(event_name).within_group.(group_name).p_values = p_timecourse;
        all_stats.(event_name).within_group.(group_name).t_values = t_timecourse;
        all_stats.(event_name).within_group.(group_name).significant_mask = sig_time_mask;
        fprintf('  %s Group: Found %d significant time points.\n', group_name, sum(sig_time_mask));
    end
    
    % --- Step 3: Between-Group Comparison of Change (Post-Pre Delta) Across Time ---
    fprintf('Step 3: Performing between-group test on training effect across time...\n');
    
    % Calculate delta (Post - Pre) for each group, averaged over Beta and ROI channels
    delta_haptic_ts = squeeze(mean(mean(all_tf_data.Haptic.Post.(event_name)(:, stat_freq_idx(1):stat_freq_idx(2), :, roi_mask) - all_tf_data.Haptic.Pre.(event_name)(:, stat_freq_idx(1):stat_freq_idx(2), :, roi_mask), 2), 4));
    delta_nonhaptic_ts = squeeze(mean(mean(all_tf_data.NonHaptic.Post.(event_name)(:, stat_freq_idx(1):stat_freq_idx(2), :, roi_mask) - all_tf_data.NonHaptic.Pre.(event_name)(:, stat_freq_idx(1):stat_freq_idx(2), :, roi_mask), 2), 4));
    
    % Perform independent t-test at each time point
    p_timecourse = zeros(1, nTimes);
    t_timecourse = zeros(1, nTimes);
    for t_idx = 1:nTimes
        [~, p_timecourse(t_idx), ~, temp_stat] = ttest2(delta_haptic_ts(:, t_idx), delta_nonhaptic_ts(:, t_idx));
        t_timecourse(t_idx) = temp_stat.tstat;
    end
    
    % Correct for multiple comparisons across time
    sig_time_mask = fdr_bh(p_timecourse, stat_alpha);
    
    % Store results
    all_stats.(event_name).between_group.p_values = p_timecourse;
    all_stats.(event_name).between_group.t_values = t_timecourse;
    all_stats.(event_name).between_group.significant_mask = sig_time_mask;
    fprintf('  Between-Group (Delta vs Delta): Found %d significant time points.\n', sum(sig_time_mask));
end

fprintf('\n============================================================\n');
fprintf('STATISTICAL ANALYSIS COMPLETE\n');
fprintf('============================================================\n');


%% PLOTTING SECTION FOR STATISTICAL RESULTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\nCreating plots for statistical results...\n');

plot_output_dir = 'analysis_plots_with_stats';
if ~exist(plot_output_dir, 'dir'), mkdir(plot_output_dir); end

for evt = 1:nEvents
    event_name = events{evt};
    
    if ~isfield(all_stats, event_name) || isempty(all_stats.(event_name).ROI_channel_names)
        continue; % Skip if no ROI was found
    end
    
    roi_channels = all_stats.(event_name).ROI_channels_mask;
    roi_names = strjoin(all_stats.(event_name).ROI_channel_names, ', ');

    % --- Plot 1: Within-Group (Pre vs Post) ---
    for g = 1:length(groups)
        group_name = groups{g};
        
        figure('Position', [100 100 800 600], 'color', 'w');
        
        % Get time-series data averaged over ROI channels and Beta band
        pre_ts = squeeze(mean(mean(all_tf_data.(group_name).Pre.(event_name)(:, stat_freq_idx(1):stat_freq_idx(2), :, roi_channels), 2), 4));
        post_ts = squeeze(mean(mean(all_tf_data.(group_name).Post.(event_name)(:, stat_freq_idx(1):stat_freq_idx(2), :, roi_channels), 2), 4));
        
        % Plot the mean time courses
        hold on;
        plot(times, mean(pre_ts, 1), 'b', 'LineWidth', 2);
        plot(times, mean(post_ts, 1), 'r', 'LineWidth', 2);
        
        % Add shaded region for significant time points
        sig_mask = all_stats.(event_name).within_group.(group_name).significant_mask;
        yl = ylim;
        for t_idx = 1:nTimes
            if sig_mask(t_idx)
                patch([times(t_idx)-1/SR/2, times(t_idx)+1/SR/2, times(t_idx)+1/SR/2, times(t_idx)-1/SR/2], [yl(1) yl(1) yl(2) yl(2)], 'k', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
            end
        end
        
        hold off;
        legend({'Pre-Training', 'Post-Training', 'Significant Difference'});
        title(sprintf('Within-Group: Pre vs. Post (%s, %s)\nROI: %s', group_name, event_name, roi_names));
        xlabel('Time (s)'); ylabel(sprintf('%s Power (dB)', stat_band_name));
        xlim([-1 3]);
        grid on;
        
        fig_filename = sprintf('Stats_WithinGroup_%s_%s.png', group_name, event_name);
        print(gcf, fullfile(plot_output_dir, fig_filename), '-dpng', '-r300');
    end
    
    % --- Plot 2: Between-Group (Delta vs Delta) ---
    figure('Position', [150 150 800 600], 'color', 'w');
    
    % Get delta time-series data
    delta_haptic_ts = squeeze(mean(mean(all_tf_data.Haptic.Post.(event_name)(:, stat_freq_idx(1):stat_freq_idx(2), :, roi_channels) - all_tf_data.Haptic.Pre.(event_name)(:, stat_freq_idx(1):stat_freq_idx(2), :, roi_channels), 2), 4));
    delta_nonhaptic_ts = squeeze(mean(mean(all_tf_data.NonHaptic.Post.(event_name)(:, stat_freq_idx(1):stat_freq_idx(2), :, roi_channels) - all_tf_data.NonHaptic.Pre.(event_name)(:, stat_freq_idx(1):stat_freq_idx(2), :, roi_channels), 2), 4));

    % Plot the mean delta time courses
    hold on;
    plot(times, mean(delta_haptic_ts, 1), 'c', 'LineWidth', 2);
    plot(times, mean(delta_nonhaptic_ts, 1), 'm', 'LineWidth', 2);
    
    % Add shaded region for significant time points
    sig_mask = all_stats.(event_name).between_group.significant_mask;
    yl = ylim;
    for t_idx = 1:nTimes
        if sig_mask(t_idx)
            patch([times(t_idx)-1/SR/2, times(t_idx)+1/SR/2, times(t_idx)+1/SR/2, times(t_idx)-1/SR/2], [yl(1) yl(1) yl(2) yl(2)], 'k', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
        end
    end
    
    hold off;
    legend({'Haptic Change (Post-Pre)', 'Non-Haptic Change (Post-Pre)', 'Significant Difference'});
    title(sprintf('Between-Group: Comparison of Change (%s)\nROI: %s', event_name, roi_names));
    xlabel('Time (s)'); ylabel(sprintf('Change in %s Power (dB)', stat_band_name));
    xlim([-1 3]);
    grid on;
    
    fig_filename = sprintf('Stats_BetweenGroup_%s.png', event_name);
    print(gcf, fullfile(plot_output_dir, fig_filename), '-dpng', '-r300');
end

fprintf('\nStatistical plotting complete!\n');


%% FDR Function
% Benjamini-Hochberg FDR procedure
function h = fdr_bh(pvals, q)
    pvals = pvals(~isnan(pvals));
    pvals = pvals(:);
    [sorted_pvals, ~] = sort(pvals);
    V = length(sorted_pvals);
    I = (1:V)';
    p_threshold_idx = find(sorted_pvals <= (I./V)*q, 1, 'last');
    if isempty(p_threshold_idx)
        crit_p = 0;
    else
        crit_p = sorted_pvals(p_threshold_idx);
    end
    h = pvals <= crit_p;
end
