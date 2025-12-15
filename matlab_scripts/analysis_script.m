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

% clc;
% clear;
% close all;

%% Core Parameters
nChannels = 60;
SR = 250;
nSubject_total = 44; % The total number of subjects in the original dataset

% Subject Exclusion
subjects_to_exclude = [1,2]; % Keep empty [] to include all subjects
% subjects_to_exclude = [1,2,4,33];

% Define event types, groups, and timepoints
events = {'MI';
          % 'Rest'
};
groups = {'Haptic', 'NonHaptic'};
timepoints = {'Pre', 'Post'};

% Path to reference files
load reference/EEG_chlocs_60.mat

%% Main Analysis Loop (Data Processing)
% For brevity, we will load the processed data instead of re-running the loop.
if ~exist('./tfa/time_frequency_analysis_ALL_RESULTS_ICA_v2.mat', 'file')
    error('Results file not found. Please run the full data processing loop first.');
else
    fprintf('Loading pre-processed data from time_frequency_analysis_ALL_RESULTS.mat...\n');
    load('./tfa/time_frequency_analysis_ALL_RESULTS_ICA_v2.mat');
end

% Get time and frequency vectors from the loaded data
[~, num_frex, nTimes, ~] = size(all_tf_data.Haptic.Pre.MI);
frex = logspace(log10(2), log10(80), num_frex);
times = linspace(-5, 5, nTimes);

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
channels_of_interest = {'C1', 'CP3','CP5','P1','P3','P5'}; % For TF plots

% --- Topography Parameters ---
freq_bands = {'Alpha', [8 13]; 'Beta', [13 30]};
topo_time_window = [0.75 1.5]; % in seconds

% --- Plot Saving Parameters ---.
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

%% POOLED TOPOGRAPHIES: Haptic/NonHaptic × Pre/Post + Summed
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pooled_topo_output_dir = 'analysis_plots_pooled_topos';
if ~exist(pooled_topo_output_dir, 'dir')
    mkdir(pooled_topo_output_dir);
end

topo_time_idx = dsearchn(times', topo_time_window');  % you already have this
% freq_idx will be recomputed per band

for evt = 1:length(events)
    event_name = events{evt};

    for fb = 1:size(freq_bands, 1)
        band_name = freq_bands{fb, 1};
        band_range = freq_bands{fb, 2};
        freq_idx = dsearchn(frex', band_range');

        fprintf('Creating pooled topography figure for %s, %s band...\n', event_name, band_name);

        % --- Compute topographies for each combination ---
        % Haptic Pre
        data_HP = all_tf_data.Haptic.Pre.(event_name);
        topo_HP = squeeze(mean(mean(mean( ...
            data_HP(:, ...
                freq_idx(1):freq_idx(2), ...
                topo_time_idx(1):topo_time_idx(2), :), ...
            1), 2), 3));  % -> [nChannels x 1]

        % Haptic Post
        data_HPost = all_tf_data.Haptic.Post.(event_name);
        topo_HPost = squeeze(mean(mean(mean( ...
            data_HPost(:, ...
                freq_idx(1):freq_idx(2), ...
                topo_time_idx(1):topo_time_idx(2), :), ...
            1), 2), 3));

        % NonHaptic Pre
        data_NHP = all_tf_data.NonHaptic.Pre.(event_name);
        topo_NHP = squeeze(mean(mean(mean( ...
            data_NHP(:, ...
                freq_idx(1):freq_idx(2), ...
                topo_time_idx(1):topo_time_idx(2), :), ...
            1), 2), 3));

        % NonHaptic Post
        data_NHPost = all_tf_data.NonHaptic.Post.(event_name);
        topo_NHPost = squeeze(mean(mean(mean( ...
            data_NHPost(:, ...
                freq_idx(1):freq_idx(2), ...
                topo_time_idx(1):topo_time_idx(2), :), ...
            1), 2), 3));

        % Sum (or you can average by dividing by 4)
        topo_sum = topo_HP + topo_HPost + topo_NHP + topo_NHPost;
        % Alternatively:
        % topo_sum = (topo_HP + topo_HPost + topo_NHP + topo_NHPost) / 4;

        % --- Plot ---
        figure('color', 'w');
        sgtitle(sprintf('Pooled Topographies (%s, %s band, %.2f–%.2f s)', ...
            event_name, band_name, topo_time_window(1), topo_time_window(2)), ...
            'FontSize', 16, 'FontWeight', 'bold');

        % Haptic Pre
        subplot(2,3,1);
        topoplot(topo_HP, EEG_chlocs, 'maplimits', topo_clim, ...
            'style', 'map', 'electrodes', 'on');
        title('Haptic Pre');
        colorbar;

        % Haptic Post
        subplot(2,3,2);
        topoplot(topo_HPost, EEG_chlocs, 'maplimits', topo_clim, ...
            'style', 'map', 'electrodes', 'on');
        title('Haptic Post');
        colorbar;

        % NonHaptic Pre
        subplot(2,3,4);
        topoplot(topo_NHP, EEG_chlocs, 'maplimits', topo_clim, ...
            'style', 'map', 'electrodes', 'on');
        title('NonHaptic Pre');
        colorbar;

        % NonHaptic Post
        subplot(2,3,5);
        topoplot(topo_NHPost, EEG_chlocs, 'maplimits', topo_clim, ...
            'style', 'map', 'electrodes', 'on');
        title('NonHaptic Post');
        colorbar;

        % Summed / pooled topo
        subplot(2,3,3);
        topoplot(topo_sum, EEG_chlocs, 'maplimits', sum_clim, ...
            'style', 'map', 'electrodes', 'on');
        title('Pooled (sum of 4)');
        colorbar;

        % Save
        fig_filename = sprintf('PooledTopos_%s_%s.png', event_name, band_name);
        print(gcf, fullfile(pooled_topo_output_dir, fig_filename), '-dpng', '-r300');
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
stat_band_range = [8 13];
stat_freq_idx = dsearchn(frex', stat_band_range');
roi_time_window = [0.5 1.5]; 
roi_time_idx = dsearchn(times', roi_time_window');

% --- Manual ROI Selection ---
use_manual_roi = false; % SET TO true TO USE THE MANUAL LIST BELOW
manual_roi_channels = {'C3', 'C1', 'CP3','CP5','P1','P3','P5'};

all_stats = struct();

for evt = 1:length(events)
    event_name = events{evt};
    fprintf('\n--- Analyzing Event: %s ---\n', event_name);

    % --- Step 1: Identify ROI (Significant Channels) ---
    if use_manual_roi
        fprintf('Step 1: Using manually defined ROI...\n');
        roi_mask = ismember({EEG_chlocs.labels}, manual_roi_channels);

    else
        fprintf('Step 1: Identifying data-driven ROI channels for %s in the %s band (Pre+Post pooled)...\n', ...
            event_name, stat_band_name);

        % (A) Define a motor-strip search area (candidate channels)
        %     Adjust this list to match your montage / hypotheses.
        candidate_roi_channels = { ...
            'C1','C3','C5', ...
            'CP1','CP3','CP5', ...
            'P1','P3','P5' ...

            % 'C3', ...
            % 'CP1','CP3','CP5', ...
            % 'P1','P3','P5'
        };
        candidate_mask = ismember({EEG_chlocs.labels}, candidate_roi_channels);
        candidate_idx  = find(candidate_mask);
        nCand          = numel(candidate_idx);

        if nCand == 0
            error('No candidate ROI channels found in EEG_chlocs. Check candidate_roi_channels names.');
        end

        % (B) Combine PRE and POST across both groups (group- and time-agnostic)
        %     Shape: [ (subj_H*2 + subj_NH*2) x freqs x times x channels ]
        combined_all = cat(1, ...
            all_tf_data.Haptic.Pre.(event_name), ...
            all_tf_data.Haptic.Post.(event_name), ...
            all_tf_data.NonHaptic.Pre.(event_name), ...
            all_tf_data.NonHaptic.Post.(event_name));

        % (C) Average over the stats freq band and ROI time window,
        %     only within candidate channels:
        %     Result: [N_obs x N_candidate_channels]
        roi_tf_window_data = squeeze( ...
            mean( ...  % over freq
                mean( ...  % over time
                    combined_all(:, ...
                        stat_freq_idx(1):stat_freq_idx(2), ...
                        roi_time_idx(1):roi_time_idx(2), ...
                        candidate_mask), ...
                    2), ...   % freq dim
                3) ...       % time dim
        );  % -> N_obs x N_candidate

        % Ensure [observations x channels]
        if isvector(roi_tf_window_data)
            roi_tf_window_data = roi_tf_window_data(:).';
        end

        % (D) Mean effect and t-test vs 0 (assuming baseline-corrected dB)
        mean_effect = mean(roi_tf_window_data, 1);   % 1 x nCand
        [~, pvals]  = ttest(roi_tf_window_data);     % 1 x nCand

        % (E) FDR across candidate channels
        sig_cand = fdr_bh(pvals, stat_alpha);

        % (F) Fallback: if nothing survives FDR, pick top-N strongest ERD channels
        if ~any(sig_cand)
            fprintf('  No candidate channels survived FDR; using top-N ERD channels instead.\n');
            N_top = min(4, nCand);  % e.g. 4 channels max

            % For ERD, we want the most negative mean (largest decrease)
            [~, sort_idx] = sort(mean_effect, 'ascend');  % more negative first
            top_idx_cand  = sort_idx(1:N_top);

            sig_cand = false(1, nCand);
            sig_cand(top_idx_cand) = true;
        end

        % (G) Map candidate-level mask back to full channel mask
        roi_mask = false(1, nChannels);
        roi_mask(candidate_idx(sig_cand)) = true;
    end

    % Store ROI info
    all_stats.(event_name).ROI_channels_mask  = roi_mask;
    all_stats.(event_name).ROI_channel_names = {EEG_chlocs(roi_mask).labels};
    fprintf('  ROI identified. Using %d channels: %s\n', ...
        sum(roi_mask), strjoin(all_stats.(event_name).ROI_channel_names, ', '));
    
    if sum(roi_mask) == 0
        fprintf('  WARNING: No significant ROI channels found. Skipping further stats for this event.\n');
        continue;
    end

    % --- Sliding window definition (in seconds) ---
    bin_size_sec = 0.10;   % e.g. 200 ms window
    bin_step_sec = 0.005;  % e.g. 50 ms step (set = bin_size_sec for non-overlapping)

    bin_size_samp = round(bin_size_sec * SR);
    bin_step_samp = round(bin_step_sec * SR);

    % Restrict bins to the ROI time window [roi_time_window]
    analysis_idx = roi_time_idx(1):roi_time_idx(2);

    first_start = analysis_idx(1);
    last_start  = analysis_idx(end) - bin_size_samp + 1;
    bin_starts  = first_start:bin_step_samp:last_start;
    bin_ends    = bin_starts + bin_size_samp - 1;
    nBins       = numel(bin_starts);
    bin_centers = round((bin_starts + bin_ends)/2);

    all_stats.(event_name).bin_starts  = bin_starts;
    all_stats.(event_name).bin_ends    = bin_ends;
    all_stats.(event_name).bin_centers = times(bin_centers);


    % --- Step 2 & 3: Within and Between Group Comparisons Across Time ---
    fprintf('Step 2: Performing within-group (Pre vs. Post) tests using sliding windows...\n');
    for g = 1:length(groups)
        group_name = groups{g};

        % subjects × time (already averaged over freq band and ROI channels)
        pre_data_ts  = squeeze(mean(mean(all_tf_data.(group_name).Pre.(event_name)(:, ...
            stat_freq_idx(1):stat_freq_idx(2), :, roi_mask), 2), 4));
        post_data_ts = squeeze(mean(mean(all_tf_data.(group_name).Post.(event_name)(:, ...
            stat_freq_idx(1):stat_freq_idx(2), :, roi_mask), 2), 4));

        nSubj   = size(pre_data_ts, 1);
        pre_bins  = zeros(nSubj, nBins);
        post_bins = zeros(nSubj, nBins);

        % Average inside each time bin
        for b = 1:nBins
            idx_range = bin_starts(b):bin_ends(b);
            pre_bins(:,  b) = mean(pre_data_ts(:,  idx_range), 2);
            post_bins(:, b) = mean(post_data_ts(:, idx_range), 2);
        end

        % Paired t-test Post vs Pre for each bin
        p_bins = nan(1, nBins);
        for b = 1:nBins
            [~, p_bins(b)] = ttest(post_bins(:, b), pre_bins(:, b));
        end
        sig_bins = fdr_bh(p_bins, stat_alpha);

        % Expand bin significance to a sample-wise mask (for plotting)
        sig_mask_time = false(1, nTimes);
        for b = 1:nBins
            if sig_bins(b)
                sig_mask_time(bin_starts(b):bin_ends(b)) = true;
            end
        end

        all_stats.(event_name).within_group.(group_name).p_bins           = p_bins;
        all_stats.(event_name).within_group.(group_name).sig_bins        = sig_bins;
        all_stats.(event_name).within_group.(group_name).significant_mask = sig_mask_time;

        fprintf('  %s Group: %d/%d significant time bins.\n', group_name, sum(sig_bins), nBins);
    end

    
        fprintf('Step 3: Performing between-group test on training effect (Post-Pre) using sliding windows...\n');

    % subjects × time, delta within ROI (Post - Pre)
    delta_haptic_ts = squeeze(mean(mean( ...
        all_tf_data.Haptic.Post.(event_name)(:, stat_freq_idx(1):stat_freq_idx(2), :, roi_mask) - ...
        all_tf_data.Haptic.Pre.(event_name)(:,  stat_freq_idx(1):stat_freq_idx(2), :, roi_mask), 2), 4));

    delta_nonhaptic_ts = squeeze(mean(mean( ...
        all_tf_data.NonHaptic.Post.(event_name)(:, stat_freq_idx(1):stat_freq_idx(2), :, roi_mask) - ...
        all_tf_data.NonHaptic.Pre.(event_name)(:,  stat_freq_idx(1):stat_freq_idx(2), :, roi_mask), 2), 4));

    nSubj_H  = size(delta_haptic_ts, 1);
    nSubj_NH = size(delta_nonhaptic_ts, 1);
    delta_h_bins  = zeros(nSubj_H,  nBins);
    delta_nh_bins = zeros(nSubj_NH, nBins);

    for b = 1:nBins
        idx_range = bin_starts(b):bin_ends(b);
        delta_h_bins(:,  b) = mean(delta_haptic_ts(:,  idx_range), 2);
        delta_nh_bins(:, b) = mean(delta_nonhaptic_ts(:, idx_range), 2);
    end

    p_bins = nan(1, nBins);
    for b = 1:nBins
        [~, p_bins(b)] = ttest2(delta_h_bins(:, b), delta_nh_bins(:, b));
    end
    sig_bins = fdr_bh(p_bins, stat_alpha);

    sig_mask_time = false(1, nTimes);
    for b = 1:nBins
        if sig_bins(b)
            sig_mask_time(bin_starts(b):bin_ends(b)) = true;
        end
    end

    all_stats.(event_name).between_group.p_bins            = p_bins;
    all_stats.(event_name).between_group.sig_bins         = sig_bins;
    all_stats.(event_name).between_group.significant_mask = sig_mask_time;

    fprintf('  Between-Group: %d/%d significant time bins.\n', sum(sig_bins), nBins);

end

fprintf('\n============================================================\n');
fprintf('STATISTICAL ANALYSIS COMPLETE\n');
fprintf('============================================================\n');

%% REPORT SIGNIFICANT TIME BINS (AS TIME RANGES)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\nSignificant time bins (in seconds):\n');

for evt = 1:length(events)
    event_name = events{evt};
    if ~isfield(all_stats, event_name) || ...
       ~isfield(all_stats.(event_name), 'bin_starts')
        continue;
    end

    bin_starts = all_stats.(event_name).bin_starts;   % indices into "times"
    bin_ends   = all_stats.(event_name).bin_ends;

    fprintf('\n=== Event: %s ===\n', event_name);

    %% 1) WITHIN-GROUP (Pre vs Post) FOR EACH GROUP
    for g = 1:length(groups)
        group_name = groups{g};
        if ~isfield(all_stats.(event_name).within_group, group_name), continue; end

        sig_bins = all_stats.(event_name).within_group.(group_name).sig_bins;
        if isempty(sig_bins) || ~any(sig_bins)
            fprintf('  %s: no significant time bins.\n', group_name);
            continue;
        end

        % Find contiguous clusters of significant bins
        sig_idx = find(sig_bins);
        d       = diff(sig_idx);
        cluster_starts = sig_idx([1, find(d > 1) + 1]);
        cluster_ends   = sig_idx([find(d > 1), numel(sig_idx)]);

        fprintf('  %s (Pre vs Post):\n', group_name);
        for c = 1:numel(cluster_starts)
            b_start = cluster_starts(c);
            b_end   = cluster_ends(c);
            t_start = times(bin_starts(b_start));
            t_end   = times(bin_ends(b_end));
            fprintf('    Cluster %d: %.3f – %.3f s\n', c, t_start, t_end);
        end
    end

    %% 2) BETWEEN-GROUP (Δ Post-Pre Haptic vs NonHaptic)
    if isfield(all_stats.(event_name), 'between_group') && ...
       isfield(all_stats.(event_name).between_group, 'sig_bins')

        sig_bins = all_stats.(event_name).between_group.sig_bins;

        if ~isempty(sig_bins) && any(sig_bins)
            sig_idx = find(sig_bins);
            d       = diff(sig_idx);
            cluster_starts = sig_idx([1, find(d > 1) + 1]);
            cluster_ends   = sig_idx([find(d > 1), numel(sig_idx)]);

            fprintf('  Between-group (Δ Post-Pre: Haptic vs NonHaptic):\n');
            for c = 1:numel(cluster_starts)
                b_start = cluster_starts(c);
                b_end   = cluster_ends(c);
                t_start = times(bin_starts(b_start));
                t_end   = times(bin_ends(b_end));
                fprintf('    Cluster %d: %.3f – %.3f s\n', c, t_start, t_end);
            end
        else
            fprintf('  Between-group: no significant time bins.\n');
        end
    end
end


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
        ylim = [-2.5 0.5];
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

%% BOXPLOTS: Pre vs Post distributions in ROI
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\nCreating boxplots for Pre vs Post in each group...\n');

% Time window for boxplots (use the same as ROI, or change if you like)
box_time_window = roi_time_window;
box_time_idx    = dsearchn(times', box_time_window');

for evt = 1:length(events)
    event_name = events{evt};

    if ~isfield(all_stats, event_name) || isempty(all_stats.(event_name).ROI_channel_names)
        continue;
    end

    roi_channels = all_stats.(event_name).ROI_channels_mask;

    for g = 1:length(groups)
        group_name = groups{g};

        % One scalar per subject: mean over freq band, box time window, and ROI channels
        pre_vals = squeeze(mean(mean(mean( ...
            all_tf_data.(group_name).Pre.(event_name)(:, ...
                stat_freq_idx(1):stat_freq_idx(2), ...
                box_time_idx(1):box_time_idx(2), ...
                roi_channels), 2), 3), 4));

        post_vals = squeeze(mean(mean(mean( ...
            all_tf_data.(group_name).Post.(event_name)(:, ...
                stat_freq_idx(1):stat_freq_idx(2), ...
                box_time_idx(1):box_time_idx(2), ...
                roi_channels), 2), 3), 4));

        figure('color', 'w');
        data_box  = [pre_vals(:); post_vals(:)];
        group_var = [ones(numel(pre_vals),1); 2*ones(numel(post_vals),1)];
        boxplot(data_box, group_var, 'Labels', {'Pre','Post'});
        ylabel(sprintf('%s Power (dB)', stat_band_name));
        title(sprintf('ROI boxplot: %s group, %s (%.2f–%.2f s)', ...
            group_name, event_name, box_time_window(1), box_time_window(2)));

        hold on;
        % Optional: show each subject as paired points with lines
        nSubj = numel(pre_vals);
        x_jitter = 0.05*randn(nSubj,1);
        plot(1+x_jitter, pre_vals,  'ko', 'MarkerFaceColor',[0.3 0.3 0.3]);
        plot(2+x_jitter, post_vals, 'ko', 'MarkerFaceColor',[0.7 0.7 0.7]);
        for s = 1:nSubj
            plot([1+x_jitter(s), 2+x_jitter(s)], [pre_vals(s), post_vals(s)], '-', 'Color',[0.7 0.7 0.7]);
        end
        hold off;

        % Paired t-test for the boxplot window
        [~, p_pair] = ttest(post_vals, pre_vals);
        if exist('subtitle','file')
            subtitle(sprintf('Paired t-test Pre vs Post: p = %.3g', p_pair));
        else
            fprintf('  %s, %s: Pre vs Post boxplot window p = %.3g\n', group_name, event_name, p_pair);
        end

        % (Optional) save figure
        fig_filename = sprintf('Boxplot_%s_%s.png', group_name, event_name);
        print(gcf, fullfile(stats_plot_output_dir, fig_filename), '-dpng', '-r300');
    end
end

%% CROSS-GROUP BOXPLOTS: Haptic vs NonHaptic (ROI)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\nCreating cross-group boxplots for ROI (Post-Pre)...\n');

% Use same time window as before (or override here if you want a different one)
box_time_window = roi_time_window;   % e.g. [0.5 2.5]
box_time_idx    = dsearchn(times', box_time_window');

for evt = 1:length(events)
    event_name = events{evt};

    if ~isfield(all_stats, event_name) || isempty(all_stats.(event_name).ROI_channel_names)
        continue;
    end

    roi_channels = all_stats.(event_name).ROI_channels_mask;

    % ---------- 1) Extract ROI-averaged Pre/Post for each group ----------
    % HAPTIC
    pre_H = squeeze(mean(mean(mean( ...
        all_tf_data.Haptic.Pre.(event_name)(:, ...
            stat_freq_idx(1):stat_freq_idx(2), ...
            box_time_idx(1):box_time_idx(2), ...
            roi_channels), 2), 3), 4));

    post_H = squeeze(mean(mean(mean( ...
        all_tf_data.Haptic.Post.(event_name)(:, ...
            stat_freq_idx(1):stat_freq_idx(2), ...
            box_time_idx(1):box_time_idx(2), ...
            roi_channels), 2), 3), 4));

    % NON-HAPTIC
    pre_NH = squeeze(mean(mean(mean( ...
        all_tf_data.NonHaptic.Pre.(event_name)(:, ...
            stat_freq_idx(1):stat_freq_idx(2), ...
            box_time_idx(1):box_time_idx(2), ...
            roi_channels), 2), 3), 4));

    post_NH = squeeze(mean(mean(mean( ...
        all_tf_data.NonHaptic.Post.(event_name)(:, ...
            stat_freq_idx(1):stat_freq_idx(2), ...
            box_time_idx(1):box_time_idx(2), ...
            roi_channels), 2), 3), 4));

    % Δ (Post - Pre) per subject in each group
    delta_H  = post_H  - pre_H;
    delta_NH = post_NH - pre_NH;

    %% ---------- 2) 4-box plot: Haptic vs NonHaptic, Pre vs Post ----------
    figure('color', 'w');
    data_box  = [pre_H(:); post_H(:); pre_NH(:); post_NH(:)];
    group_idx = [ones(numel(pre_H),1); ...
                 2*ones(numel(post_H),1); ...
                 3*ones(numel(pre_NH),1); ...
                 4*ones(numel(post_NH),1)];

    boxplot(data_box, group_idx, ...
        'Labels', {'H-Pre','H-Post','NH-Pre','NH-Post'});
    ylabel(sprintf('%s Power (dB)', stat_band_name));
    title(sprintf('ROI boxplot: Group × Time (%s)\n%.2f–%.2f s', ...
        event_name, box_time_window(1), box_time_window(2)));

    hold on;
    % Optional: scatter points for each group/time
    jitter_scale = 0.05;
    x_pos = [1 2 3 4];
    % Haptic Pre
    xj = x_pos(1) + jitter_scale*randn(numel(pre_H),1);
    plot(xj, pre_H, 'ko', 'MarkerFaceColor',[0.5 0.5 0.5]);
    % Haptic Post
    xj = x_pos(2) + jitter_scale*randn(numel(post_H),1);
    plot(xj, post_H, 'ko', 'MarkerFaceColor',[0.5 0.5 0.5]);
    % NonHaptic Pre
    xj = x_pos(3) + jitter_scale*randn(numel(pre_NH),1);
    plot(xj, pre_NH, 'ko', 'MarkerFaceColor',[0.7 0.7 0.7]);
    % NonHaptic Post
    xj = x_pos(4) + jitter_scale*randn(numel(post_NH),1);
    plot(xj, post_NH, 'ko', 'MarkerFaceColor',[0.7 0.7 0.7]);
    hold off;

    % (Optional) save
    fig_filename = sprintf('Boxplot_4way_GroupTime_%s.png', event_name);
    print(gcf, fullfile(stats_plot_output_dir, fig_filename), '-dpng', '-r300');


    %% ---------- 3) 2-box plot: Δ(Post-Pre) Haptic vs NonHaptic ----------
    figure('color', 'w');
    data_box  = [delta_H(:); delta_NH(:)];
    group_idx = [ones(numel(delta_H),1); 2*ones(numel(delta_NH),1)];

    boxplot(data_box, group_idx, 'Labels', {'Haptic Δ','NonHaptic Δ'});
    ylabel(sprintf('Δ(Post-Pre) %s Power (dB)', stat_band_name));
    title(sprintf('Between-group ROI change (%s)\n%.2f–%.2f s', ...
        event_name, box_time_window(1), box_time_window(2)));

    hold on;
    % Scatter points for Δ values
    xj_H  = 1 + jitter_scale*randn(numel(delta_H),1);
    xj_NH = 2 + jitter_scale*randn(numel(delta_NH),1);
    plot(xj_H,  delta_H,  'ko', 'MarkerFaceColor',[0.3 0.5 1.0]);
    plot(xj_NH, delta_NH, 'ko', 'MarkerFaceColor',[1.0 0.5 0.3]);
    hold off;

    % Between-group t-test on Δ
    [~, p_delta] = ttest2(delta_H, delta_NH);
    if exist('subtitle','file')
        subtitle(sprintf('t-test on Δ(Post-Pre): p = %.3g', p_delta));
    else
        fprintf('  %s: Between-group Δ(Post-Pre) p = %.3g\n', event_name, p_delta);
    end

    % (Optional) save
    fig_filename = sprintf('Boxplot_Delta_Group_%s.png', event_name);
    print(gcf, fullfile(stats_plot_output_dir, fig_filename), '-dpng', '-r300');
end

%% LOCALIZATION INDEX: ROI vs Non-ROI
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Time window for localization (can use same as roi_time_window)
loc_time_window = roi_time_window;
loc_time_idx    = dsearchn(times', loc_time_window');

for evt = 1:length(events)
    event_name = events{evt};

    if ~isfield(all_stats, event_name) || isempty(all_stats.(event_name).ROI_channel_names)
        continue;
    end

    roi_mask    = all_stats.(event_name).ROI_channels_mask;
    nonroi_mask = ~roi_mask;  % all other channels

    fprintf('\nLocalization indices for event: %s\n', event_name);

    for g = 1:length(groups)
        group_name = groups{g};

        % Pre and Post data: [subjects x freqs x times x channels]
        data_pre  = all_tf_data.(group_name).Pre.(event_name);
        data_post = all_tf_data.(group_name).Post.(event_name);

        % ROI ERD (mean over band, time, ROI channels)
        pre_roi  = squeeze(mean(mean(mean( ...
            data_pre(:, ...
                stat_freq_idx(1):stat_freq_idx(2), ...
                loc_time_idx(1):loc_time_idx(2), ...
                roi_mask), 2), 3), 4));

        post_roi = squeeze(mean(mean(mean( ...
            data_post(:, ...
                stat_freq_idx(1):stat_freq_idx(2), ...
                loc_time_idx(1):loc_time_idx(2), ...
                roi_mask), 2), 3), 4));

        % Non-ROI ERD (same band/time, all non-ROI channels)
        pre_nonroi  = squeeze(mean(mean(mean( ...
            data_pre(:, ...
                stat_freq_idx(1):stat_freq_idx(2), ...
                loc_time_idx(1):loc_time_idx(2), ...
                nonroi_mask), 2), 3), 4));

        post_nonroi = squeeze(mean(mean(mean( ...
            data_post(:, ...
                stat_freq_idx(1):stat_freq_idx(2), ...
                loc_time_idx(1):loc_time_idx(2), ...
                nonroi_mask), 2), 3), 4));

        % Localization index: how much more ERD is in ROI vs NonROI
        % (assuming ERD is negative, this makes "more localized" positive)
        loc_pre  = pre_nonroi  - pre_roi;
        loc_post = post_nonroi - post_roi;

        % Training-induced change in localization
        delta_loc = loc_post - loc_pre;

        % Store in all_stats for later use
        all_stats.(event_name).localization.(group_name).pre   = loc_pre;
        all_stats.(event_name).localization.(group_name).post  = loc_post;
        all_stats.(event_name).localization.(group_name).delta = delta_loc;

        % Basic stats
        [~, p_prepost] = ttest(loc_post, loc_pre);
        fprintf('  %s: mean loc Pre=%.3f, Post=%.3f, Δ=%.3f (p_prepost=%.3g)\n', ...
            group_name, mean(loc_pre), mean(loc_post), mean(delta_loc), p_prepost);
    end

    % Between-group comparison on Δ localization
    delta_H  = all_stats.(event_name).localization.Haptic.delta;
    delta_NH = all_stats.(event_name).localization.NonHaptic.delta;
    [~, p_btwn] = ttest2(delta_H, delta_NH);
    fprintf('  Between groups: Δloc Haptic vs NonHaptic p = %.3g\n', p_btwn);
end

%% BOXPLOTS FOR LOCALIZATION INDEX (loc_pre, loc_post, delta_loc)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

loc_plot_output_dir = 'analysis_plots_localization';
if ~exist(loc_plot_output_dir, 'dir')
    mkdir(loc_plot_output_dir);
end

for evt = 1:length(events)
    event_name = events{evt};
    if ~isfield(all_stats, event_name) || ...
       ~isfield(all_stats.(event_name), 'localization')
        continue;
    end

    fprintf('\nCreating localization boxplots for event: %s\n', event_name);

    % Extract localization indices
    loc_H_pre   = all_stats.(event_name).localization.Haptic.pre;
    loc_H_post  = all_stats.(event_name).localization.Haptic.post;
    loc_H_delta = all_stats.(event_name).localization.Haptic.delta;

    loc_NH_pre   = all_stats.(event_name).localization.NonHaptic.pre;
    loc_NH_post  = all_stats.(event_name).localization.NonHaptic.post;
    loc_NH_delta = all_stats.(event_name).localization.NonHaptic.delta;

    %% 1A) 4-way boxplot: Haptic/NonHaptic × Pre/Post
    figure('color', 'w');
    data_box  = [loc_H_pre(:); loc_H_post(:); loc_NH_pre(:); loc_NH_post(:)];
    group_idx = [ones(numel(loc_H_pre),1); ...
                 2*ones(numel(loc_H_post),1); ...
                 3*ones(numel(loc_NH_pre),1); ...
                 4*ones(numel(loc_NH_post),1)];

    boxplot(data_box, group_idx, 'Labels', ...
        {'Haptic Pre','Haptic Post','NonHaptic Pre','NonHaptic Post'});
    ylabel('Localization index (NonROI - ROI)');
    title(sprintf('Localization (ROI vs Non-ROI): %s', event_name));

    hold on;
    jitter = 0.05;
    x_base = [1 2 3 4];
    % scatter points (optional)
    plot(x_base(1) + jitter*randn(numel(loc_H_pre),1),  loc_H_pre,  'ko', 'MarkerFaceColor',[0.5 0.5 0.5]);
    plot(x_base(2) + jitter*randn(numel(loc_H_post),1), loc_H_post, 'ko', 'MarkerFaceColor',[0.5 0.5 0.5]);
    plot(x_base(3) + jitter*randn(numel(loc_NH_pre),1), loc_NH_pre, 'ko', 'MarkerFaceColor',[0.7 0.7 0.7]);
    plot(x_base(4) + jitter*randn(numel(loc_NH_post),1),loc_NH_post,'ko', 'MarkerFaceColor',[0.7 0.7 0.7]);
    hold off;

    fig_filename = sprintf('Loc_Boxplot_PrePost_%s.png', event_name);
    print(gcf, fullfile(loc_plot_output_dir, fig_filename), '-dpng', '-r300');

    %% 1B) Boxplot of delta_loc: Haptic vs NonHaptic
    figure('color', 'w');
    data_box  = [loc_H_delta(:); loc_NH_delta(:)];
    group_idx = [ones(numel(loc_H_delta),1); 2*ones(numel(loc_NH_delta),1)];

    boxplot(data_box, group_idx, 'Labels', {'Haptic Δloc','NonHaptic Δloc'});
    ylabel('Δ Localization index (Post-Pre)');
    title(sprintf('Change in localization: %s', event_name));

    hold on;
    plot(1 + jitter*randn(numel(loc_H_delta),1),  loc_H_delta,  'ko', 'MarkerFaceColor',[0.3 0.5 1.0]);
    plot(2 + jitter*randn(numel(loc_NH_delta),1), loc_NH_delta, 'ko', 'MarkerFaceColor',[1.0 0.5 0.3]);
    hold off;

    fig_filename = sprintf('Loc_Boxplot_Delta_%s.png', event_name);
    print(gcf, fullfile(loc_plot_output_dir, fig_filename), '-dpng', '-r300');
end

%% BAR PLOTS WITH ERROR BARS FOR LOCALIZATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for evt = 1:length(events)
    event_name = events{evt};
    if ~isfield(all_stats, event_name) || ...
       ~isfield(all_stats.(event_name), 'localization')
        continue;
    end

    loc_H_pre   = all_stats.(event_name).localization.Haptic.pre;
    loc_H_post  = all_stats.(event_name).localization.Haptic.post;
    loc_NH_pre  = all_stats.(event_name).localization.NonHaptic.pre;
    loc_NH_post = all_stats.(event_name).localization.NonHaptic.post;

    % Means
    m_H_pre   = mean(loc_H_pre);
    m_H_post  = mean(loc_H_post);
    m_NH_pre  = mean(loc_NH_pre);
    m_NH_post = mean(loc_NH_post);

    % SEMs
    s_H_pre   = std(loc_H_pre)  / sqrt(numel(loc_H_pre));
    s_H_post  = std(loc_H_post) / sqrt(numel(loc_H_post));
    s_NH_pre  = std(loc_NH_pre) / sqrt(numel(loc_NH_pre));
    s_NH_post = std(loc_NH_post)/ sqrt(numel(loc_NH_post));

    % Bar matrix: rows = groups, cols = time (Pre, Post)
    means = [m_H_pre,  m_H_post; ...
             m_NH_pre, m_NH_post];

    sems  = [s_H_pre,  s_H_post; ...
             s_NH_pre, s_NH_post];

    figure('color', 'w');
    hb = bar(means); hold on;

    % Add error bars
    numGroups    = size(means,1);
    numBarsEach  = size(means,2);
    groupWidth   = min(0.8, numBarsEach/(numBarsEach + 1.5));

    for i = 1:numBarsEach
        x = (1:numGroups) - groupWidth/2 + (2*i-1) * groupWidth / (2*numBarsEach);
        errorbar(x, means(:,i), sems(:,i), 'k', 'linestyle', 'none', 'LineWidth', 1.5);
    end

    set(gca, 'XTickLabel', {'Haptic','NonHaptic'});
    ylabel('Localization index (NonROI - ROI)');
    legend({'Pre','Post'}, 'Location','best');
    title(sprintf('Localization (ROI vs Non-ROI) with SEM: %s', event_name));
    grid on;

    fig_filename = sprintf('Loc_Bar_GroupTime_%s.png', event_name);
    print(gcf, fullfile(loc_plot_output_dir, fig_filename), '-dpng', '-r300');
end

%% DIFFERENCE MAPS: (Post - Pre) PER GROUP + BETWEEN-GROUP DIFFERENCE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

diff_topo_output_dir = 'analysis_plots_diffmaps';
if ~exist(diff_topo_output_dir, 'dir')
    mkdir(diff_topo_output_dir);
end

% Use same time window as localization
diff_time_window = loc_time_window;  % or roi_time_window
diff_time_idx    = dsearchn(times', diff_time_window');

for evt = 1:length(events)
    event_name = events{evt};

    fprintf('\nCreating Post-Pre difference maps for event: %s\n', event_name);

    % Haptic data
    data_H_pre  = all_tf_data.Haptic.Pre.(event_name);
    data_H_post = all_tf_data.Haptic.Post.(event_name);

    % NonHaptic data
    data_NH_pre  = all_tf_data.NonHaptic.Pre.(event_name);
    data_NH_post = all_tf_data.NonHaptic.Post.(event_name);

    % Δ (Post-Pre) for each subject, then mean over subjects, freq, time
    delta_H = data_H_post - data_H_pre;    % [subj x freq x time x chan]
    delta_NH = data_NH_post - data_NH_pre;

    % Mean topography over subjects, freq band, and time window
    topo_delta_H = squeeze(mean(mean(mean( ...
        delta_H(:, ...
            stat_freq_idx(1):stat_freq_idx(2), ...
            diff_time_idx(1):diff_time_idx(2), :), ...
        1), 2), 3));  % [nChan x 1]

    topo_delta_NH = squeeze(mean(mean(mean( ...
        delta_NH(:, ...
            stat_freq_idx(1):stat_freq_idx(2), ...
            diff_time_idx(1):diff_time_idx(2), :), ...
        1), 2), 3));

    % Between-group difference in training effect
    topo_delta_diff = topo_delta_H - topo_delta_NH;

    % Plot
    figure('color', 'w');
    sgtitle(sprintf('Post-Pre ERD difference maps (%s)\nBand %d–%d Hz, %.2f–%.2f s', ...
        event_name, stat_freq_idx(1), stat_freq_idx(2), ...
        diff_time_window(1), diff_time_window(2)), ...
        'FontSize', 14, 'FontWeight', 'bold');

    % Haptic Post-Pre
    subplot(1,3,1);
    topoplot(topo_delta_H, EEG_chlocs, 'maplimits', diff_clim, ...
        'style', 'map', 'electrodes', 'on');
    title('Haptic: Post - Pre');
    colorbar;

    % NonHaptic Post-Pre
    subplot(1,3,2);
    topoplot(topo_delta_NH, EEG_chlocs, 'maplimits', diff_clim, ...
        'style', 'map', 'electrodes', 'on');
    title('NonHaptic: Post - Pre');
    colorbar;

    % HapticΔ - NonHapticΔ
    subplot(1,3,3);
    topoplot(topo_delta_diff, EEG_chlocs, 'maplimits', diff_clim, ...
        'style', 'map', 'electrodes', 'on');
    title('Between-group: (HapticΔ - NonHapticΔ)');
    colorbar;

    % Optional: overlay ROI
    if isfield(all_stats, event_name) && ...
       isfield(all_stats.(event_name), 'ROI_channels_mask')
        roi_mask = all_stats.(event_name).ROI_channels_mask;
        roi_locs = EEG_chlocs(roi_mask);
        subplot(1,3,1); hold on; plot([roi_locs.X],[roi_locs.Y],'ko','MarkerSize',7,'LineWidth',2); hold off;
        subplot(1,3,2); hold on; plot([roi_locs.X],[roi_locs.Y],'ko','MarkerSize',7,'LineWidth',2); hold off;
        subplot(1,3,3); hold on; plot([roi_locs.X],[roi_locs.Y],'ko','MarkerSize',7,'LineWidth',2); hold off;
    end

    fig_filename = sprintf('DiffMaps_PostMinusPre_%s.png', event_name);
    print(gcf, fullfile(diff_topo_output_dir, fig_filename), '-dpng', '-r300');
end



%% FDR Function
function h = fdr_bh(pvals, q)
    pvals = pvals(~isnan(pvals)); pvals = pvals(:);
    [sorted_pvals, ~] = sort(pvals);
    V = length(sorted_pvals); I = (1:V)';
    p_threshold_idx = find(sorted_pvals <= (I./V)*q, 1, 'last');
    if isempty(p_threshold_idx), crit_p = 0; else, crit_p = sorted_pvals(p_threshold_idx); end
    h = pvals <= crit_p;
end
