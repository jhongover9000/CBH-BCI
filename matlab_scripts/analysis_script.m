%% Time-Frequency Analysis using Morlet Wavelet with Topography and Statistics
%
% This script is designed to:
% 1. Load pre-processed time-frequency data.
% 2. Provide an option to exclude specific subjects from the analysis.
% 3. Generate detailed descriptive visualizations.
% 4. Perform a comprehensive statistical analysis with flexible ROI selection.
% 5. Conduct within-group and between-group statistical tests across TIME.
% 6. Generate and save plots of the statistical results.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
clear;
close all;

%% Core Parameters
nChannels = 60;
SR = 250;
nSubject_total = 29; % The total number of subjects in the original dataset

% Subject Exclusion
subjects_to_exclude = []; % Keep empty [] to include all subjects

% Define event types, groups, and timepoints
events = {'MI', 'Rest'};
groups = {'Haptic', 'NonHaptic'};
timepoints = {'Pre', 'Post'};

% Path to reference files
load reference/EEG_chlocs_60.mat

%% Main Analysis Loop (Data Processing)
% For brevity, we will load the processed data instead of re-running the loop.
if ~exist('time_frequency_analysis_ALL_RESULTS.mat', 'file')
    error('Results file not found. Please run the full data processing loop first.');
else
    fprintf('Loading pre-processed data from time_frequency_analysis_ALL_RESULTS.mat...\n');
    load('time_frequency_analysis_ALL_RESULTS.mat');
end

% Get time and frequency vectors from the loaded data
[~, num_frex, nTimes, ~] = size(all_tf_data.Haptic.Pre.MI);
frex = logspace(log10(2), log10(80), num_frex);
times = linspace(-3, 4, nTimes);

%% Exclude Subjects
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isempty(subjects_to_exclude)
    fprintf('\nExcluding subjects: %s\n', num2str(subjects_to_exclude));
    
    % Define original subject lists
    haptic_subjects_orig = 1:2:nSubject_total;
    nonhaptic_subjects_orig = 2:2:nSubject_total;
    
    % Find indices to remove from each group
    [~, haptic_remove_idx] = intersect(haptic_subjects_orig, subjects_to_exclude);
    [~, nonhaptic_remove_idx] = intersect(nonhaptic_subjects_orig, subjects_to_exclude);
    
    % Create logical masks for keeping subjects
    haptic_keep_mask = true(1, length(haptic_subjects_orig));
    haptic_keep_mask(haptic_remove_idx) = false;
    
    nonhaptic_keep_mask = true(1, length(nonhaptic_subjects_orig));
    nonhaptic_keep_mask(nonhaptic_remove_idx) = false;
    
    % Apply the mask to the data structure
    for t = 1:length(timepoints)
        for e = 1:length(events)
            % Filter Haptic group
            all_tf_data.Haptic.(timepoints{t}).(events{e}) = all_tf_data.Haptic.(timepoints{t}).(events{e})(haptic_keep_mask, :, :, :);
            % Filter Non-Haptic group
            all_tf_data.NonHaptic.(timepoints{t}).(events{e}) = all_tf_data.NonHaptic.(timepoints{t}).(events{e})(nonhaptic_keep_mask, :, :, :);
        end
    end
    fprintf('  Data filtered. New subject counts: Haptic=%d, Non-Haptic=%d\n', sum(haptic_keep_mask), sum(nonhaptic_keep_mask));
end


%% DESCRIPTIVE VISUALIZATION PLOTTING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n============================================================\n');
fprintf('CREATING DESCRIPTIVE VISUALIZATION PLOTS\n');
fprintf('============================================================\n');

% --- Plotting Parameters ---
tf_clim = [-2 2];
topo_clim = [-2 2];
diff_clim = [-1 1];
sum_clim = [-4 4]; % New color limit for sum topographies
channels_of_interest = {'C3'}; % For TF plots

% --- Topography Parameters ---
freq_bands = {'Alpha', [8 13]; 'Beta', [13 30]};
topo_time_window = [0.5 1.0]; % in seconds

% --- Plot Saving Parameters ---
save_descriptive_plots = true;
plot_output_dir = 'analysis_plots_descriptive';
if save_descriptive_plots && ~exist(plot_output_dir, 'dir')
    mkdir(plot_output_dir);
end

% Find indices for channels, frequencies, and times
ch_idx = find(ismember({EEG_chlocs.labels}, channels_of_interest));
topo_time_idx = dsearchn(times', topo_time_window');

for evt = 1:length(events)
    event_name = events{evt};
    for fb = 1:size(freq_bands, 1)
        band_name = freq_bands{fb, 1};
        band_range = freq_bands{fb, 2};
        freq_idx = dsearchn(frex', band_range');
        fprintf('  Generating descriptive plots for %s, %s band...\n', event_name, band_name);

        % --- FIGURE 1: Haptic vs Non-Haptic Comparison ---
        for t = 1:length(timepoints)
            timepoint_name = timepoints{t};
            figure('color', 'w');
            sgtitle(sprintf('Group Comparison (%s at %s): Haptic vs. Non-Haptic [%s Band]', event_name, timepoint_name, band_name), 'FontSize', 16, 'FontWeight', 'bold');
            
            data_haptic = all_tf_data.Haptic.(timepoint_name).(event_name);
            data_nonhaptic = all_tf_data.NonHaptic.(timepoint_name).(event_name);
            
            % TF plots
            subplot(2,4,1); tf_haptic_mean = squeeze(mean(mean(data_haptic(:, :, :, ch_idx), 1), 4)); contourf(times, frex, tf_haptic_mean, 40, 'linecolor', 'none'); set(gca, 'clim', tf_clim, 'ydir', 'normal'); title('Haptic TF'); xlabel('Time (s)'); ylabel('Frequency (Hz)'); colorbar;
            subplot(2,4,2); tf_nonhaptic_mean = squeeze(mean(mean(data_nonhaptic(:, :, :, ch_idx), 1), 4)); contourf(times, frex, tf_nonhaptic_mean, 40, 'linecolor', 'none'); set(gca, 'clim', tf_clim, 'ydir', 'normal'); title('Non-Haptic TF'); xlabel('Time (s)'); colorbar;
            
            % Topography plots
            topo_haptic = squeeze(mean(mean(mean(data_haptic(:, freq_idx(1):freq_idx(2), topo_time_idx(1):topo_time_idx(2), :), 1), 2), 3));
            topo_nonhaptic = squeeze(mean(mean(mean(data_nonhaptic(:, freq_idx(1):freq_idx(2), topo_time_idx(1):topo_time_idx(2), :), 1), 2), 3));
            
            subplot(2,4,5); topoplot(topo_haptic, EEG_chlocs, 'maplimits', topo_clim, 'style', 'map', 'electrodes', 'on'); title('Haptic Topo'); colorbar;
            subplot(2,4,6); topoplot(topo_nonhaptic, EEG_chlocs, 'maplimits', topo_clim, 'style', 'map', 'electrodes', 'on'); title('Non-Haptic Topo'); colorbar;
            subplot(2,4,7); topoplot(topo_haptic + topo_nonhaptic, EEG_chlocs, 'maplimits', sum_clim, 'style', 'map', 'electrodes', 'on'); title('Sum Topo'); colorbar;
            subplot(2,4,8); topoplot(topo_haptic - topo_nonhaptic, EEG_chlocs, 'maplimits', diff_clim, 'style', 'map', 'electrodes', 'on'); title('Difference Topo'); colorbar;
            
            if save_descriptive_plots
                fig_filename = sprintf('Desc_GroupComp_%s_%s_%s.png', event_name, timepoint_name, band_name);
                print(gcf, fullfile(plot_output_dir, fig_filename), '-dpng', '-r300');
            end
        end
    end
end


%% STATISTICAL ANALYSIS SECTION (CHANNEL AND TEMPORAL)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n============================================================\n');
fprintf('STARTING CHANNEL-TEMPORAL STATISTICAL ANALYSIS\n');
fprintf('============================================================\n');

% --- Statistical Parameters ---
stat_alpha = 0.05;

% --- Parameters for ROI definition and subsequent tests ---
stat_band_name = 'Alpha';
stat_band_range = [13 30];
stat_freq_idx = dsearchn(frex', stat_band_range');
roi_time_window = [0.5 2.5]; 
roi_time_idx = dsearchn(times', roi_time_window');

% --- Manual ROI Selection ---
use_manual_roi = true; % SET TO true TO USE THE MANUAL LIST BELOW
manual_roi_channels = {'C3', 'CP3'}; % Ignored if use_manual_roi is false

all_stats = struct();

for evt = 1:length(events)
    event_name = events{evt};
    fprintf('\n--- Analyzing Event: %s ---\n', event_name);

    % --- Step 1: Identify ROI (Significant Channels) ---
    if use_manual_roi
        fprintf('Step 1: Using manually defined ROI...\n');
        roi_mask = ismember({EEG_chlocs.labels}, manual_roi_channels);
    else
        fprintf('Step 1: Identifying data-driven ROI channels for %s in the %s band...\n', event_name, stat_band_name);
        combined_pre_data = [all_tf_data.Haptic.Pre.(event_name); all_tf_data.NonHaptic.Pre.(event_name)];
        roi_tf_window_data = squeeze(mean(mean(combined_pre_data(:, stat_freq_idx(1):stat_freq_idx(2), roi_time_idx(1):roi_time_idx(2), :), 2), 3));
        [~, pvals] = ttest(roi_tf_window_data);
        roi_mask = fdr_bh(pvals, stat_alpha);
    end
    
    all_stats.(event_name).ROI_channels_mask = roi_mask;
    all_stats.(event_name).ROI_channel_names = {EEG_chlocs(roi_mask).labels};
    fprintf('  ROI identified. Found %d significant channels: %s\n', sum(roi_mask), strjoin(all_stats.(event_name).ROI_channel_names, ', '));

    if sum(roi_mask) == 0
        fprintf('  WARNING: No significant ROI channels found. Skipping further stats for this event.\n');
        continue;
    end

    % --- Step 2 & 3: Within and Between Group Comparisons Across Time ---
    fprintf('Step 2: Performing within-group (Pre vs. Post) tests across time...\n');
    for g = 1:length(groups)
        group_name = groups{g};
        pre_data_ts = squeeze(mean(mean(all_tf_data.(group_name).Pre.(event_name)(:, stat_freq_idx(1):stat_freq_idx(2), :, roi_mask), 2), 4));
        post_data_ts = squeeze(mean(mean(all_tf_data.(group_name).Post.(event_name)(:, stat_freq_idx(1):stat_freq_idx(2), :, roi_mask), 2), 4));
        p_timecourse = zeros(1, nTimes);
        for t_idx = 1:nTimes, [~, p_timecourse(t_idx)] = ttest(post_data_ts(:, t_idx), pre_data_ts(:, t_idx)); end
        all_stats.(event_name).within_group.(group_name).significant_mask = fdr_bh(p_timecourse, stat_alpha);
        fprintf('  %s Group: Found %d significant time points.\n', group_name, sum(all_stats.(event_name).within_group.(group_name).significant_mask));
    end
    
    fprintf('Step 3: Performing between-group test on training effect across time...\n');
    delta_haptic_ts = squeeze(mean(mean(all_tf_data.Haptic.Post.(event_name)(:, stat_freq_idx(1):stat_freq_idx(2), :, roi_mask) - all_tf_data.Haptic.Pre.(event_name)(:, stat_freq_idx(1):stat_freq_idx(2), :, roi_mask), 2), 4));
    delta_nonhaptic_ts = squeeze(mean(mean(all_tf_data.NonHaptic.Post.(event_name)(:, stat_freq_idx(1):stat_freq_idx(2), :, roi_mask) - all_tf_data.NonHaptic.Pre.(event_name)(:, stat_freq_idx(1):stat_freq_idx(2), :, roi_mask), 2), 4));
    p_timecourse = zeros(1, nTimes);
    for t_idx = 1:nTimes, [~, p_timecourse(t_idx)] = ttest2(delta_haptic_ts(:, t_idx), delta_nonhaptic_ts(:, t_idx)); end
    all_stats.(event_name).between_group.significant_mask = fdr_bh(p_timecourse, stat_alpha);
    fprintf('  Between-Group (Delta vs Delta): Found %d significant time points.\n', sum(all_stats.(event_name).between_group.significant_mask));
end

fprintf('\n============================================================\n');
fprintf('STATISTICAL ANALYSIS COMPLETE\n');
fprintf('============================================================\n');


%% PLOTTING SECTION FOR STATISTICAL RESULTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\nCreating plots for statistical results...\n');
stats_plot_output_dir = 'analysis_plots_with_stats';
if ~exist(stats_plot_output_dir, 'dir'), mkdir(stats_plot_output_dir); end

for evt = 1:length(events)
    event_name = events{evt};
    if ~isfield(all_stats, event_name) || isempty(all_stats.(event_name).ROI_channel_names), continue; end
    roi_channels = all_stats.(event_name).ROI_channels_mask;
    roi_names = strjoin(all_stats.(event_name).ROI_channel_names, ', ');

    for g = 1:length(groups)
        group_name = groups{g};
        figure('Position', [100 100 800 600], 'color', 'w');
        pre_ts = squeeze(mean(mean(all_tf_data.(group_name).Pre.(event_name)(:, stat_freq_idx(1):stat_freq_idx(2), :, roi_channels), 2), 4));
        post_ts = squeeze(mean(mean(all_tf_data.(group_name).Post.(event_name)(:, stat_freq_idx(1):stat_freq_idx(2), :, roi_channels), 2), 4));
        hold on; plot(times, mean(pre_ts, 1), 'b', 'LineWidth', 2); plot(times, mean(post_ts, 1), 'r', 'LineWidth', 2);
        sig_mask = all_stats.(event_name).within_group.(group_name).significant_mask;
        yl = ylim;
        for t_idx = 1:nTimes
            if sig_mask(t_idx), patch([times(t_idx)-1/SR/2, times(t_idx)+1/SR/2, times(t_idx)+1/SR/2, times(t_idx)-1/SR/2], [yl(1) yl(1) yl(2) yl(2)], 'k', 'FaceAlpha', 0.2, 'EdgeColor', 'none'); end
        end
        hold off; legend({'Pre-Training', 'Post-Training', 'Significant Difference'}); title(sprintf('Within-Group: Pre vs. Post (%s, %s)\nROI: %s', group_name, event_name, roi_names)); xlabel('Time (s)'); ylabel(sprintf('%s Power (dB)', stat_band_name)); xlim([-1 3]); grid on;
        fig_filename = sprintf('Stats_WithinGroup_%s_%s.png', group_name, event_name);
        print(gcf, fullfile(stats_plot_output_dir, fig_filename), '-dpng', '-r300');
    end
    
    figure('color', 'w');
    delta_haptic_ts = squeeze(mean(mean(all_tf_data.Haptic.Post.(event_name)(:, stat_freq_idx(1):stat_freq_idx(2), :, roi_channels) - all_tf_data.Haptic.Pre.(event_name)(:, stat_freq_idx(1):stat_freq_idx(2), :, roi_channels), 2), 4));
    delta_nonhaptic_ts = squeeze(mean(mean(all_tf_data.NonHaptic.Post.(event_name)(:, stat_freq_idx(1):stat_freq_idx(2), :, roi_channels) - all_tf_data.NonHaptic.Pre.(event_name)(:, stat_freq_idx(1):stat_freq_idx(2), :, roi_channels), 2), 4));
    hold on; plot(times, mean(delta_haptic_ts, 1), 'c', 'LineWidth', 2); plot(times, mean(delta_nonhaptic_ts, 1), 'm', 'LineWidth', 2);
    sig_mask = all_stats.(event_name).between_group.significant_mask;
    yl = ylim;
    for t_idx = 1:nTimes
        if sig_mask(t_idx), patch([times(t_idx)-1/SR/2, times(t_idx)+1/SR/2, times(t_idx)+1/SR/2, times(t_idx)-1/SR/2], [yl(1) yl(1) yl(2) yl(2)], 'k', 'FaceAlpha', 0.2, 'EdgeColor', 'none'); end
    end
    hold off; legend({'Haptic Change (Post-Pre)', 'Non-Haptic Change (Post-Pre)', 'Significant Difference'}); title(sprintf('Between-Group: Comparison of Change (%s)\nROI: %s', event_name, roi_names)); xlabel('Time (s)'); ylabel(sprintf('Change in %s Power (dB)', stat_band_name)); xlim([-1 3]); grid on;
    fig_filename = sprintf('Stats_BetweenGroup_%s.png', event_name);
    print(gcf, fullfile(stats_plot_output_dir, fig_filename), '-dpng', '-r300');
end

fprintf('\nAll plotting complete!\n');

%% FDR Function
function h = fdr_bh(pvals, q)
    pvals = pvals(~isnan(pvals)); pvals = pvals(:);
    [sorted_pvals, ~] = sort(pvals);
    V = length(sorted_pvals); I = (1:V)';
    p_threshold_idx = find(sorted_pvals <= (I./V)*q, 1, 'last');
    if isempty(p_threshold_idx), crit_p = 0; else, crit_p = sorted_pvals(p_threshold_idx); end
    h = pvals <= crit_p;
end
