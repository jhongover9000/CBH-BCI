%% Plot Individual Subject Topographies (Per-Subject Across Time Blocks)
%
% One figure per subject. Each figure shows the subject's topographies
% over consecutive time blocks (default 200 ms) within the selected window.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
% clear;
% close all;

%% USER-DEFINED PARAMETERS

% --- Data and Reference Files ---
results_file = 'time_frequency_analysis_ALL_RESULTS.mat';
chlocs_file  = 'reference/EEG_chlocs_60.mat';

% --- Plotting Selections ---
event_to_plot     = 'MI';       % 'MI' or 'Rest'
timepoint_to_plot = 'Pre';      % 'Pre' or 'Post'
band_to_plot      = 'Alpha';    % 'Alpha', 'Beta', or 'AB'

% --- Topography Parameters ---
freq_bands = {
    'Alpha', [8 13];
    'Beta',  [13 30];
    'AB',    [8 13]   % <- set to [8 30] if you intend Alpha+Beta
};
topo_time_window = [0 1]; % [sec] window for block segmentation
topo_clim        = [-3 3];    % color limits for all topoplots
electrodes_on    = 'off';     % 'on' or 'off' for electrode markers

% --- NEW: Block segmentation (milliseconds) ---
block_ms = 200;               % default 200 ms per block

% --- Plot Saving Parameters ---
save_plots      = true;
plot_output_dir = 'individual_topo_plots_per_subject';

%% Load Data and Setup
% fprintf('Loading data...\n');
% if ~exist(results_file, 'file')
%     error('Results file not found: %s. Please run the main analysis script first.', results_file);
% end
% load(results_file); % Loads struct 'all_tf_data'
% load(chlocs_file);  % Loads 'EEG_chlocs'
% 
% if save_plots && ~exist(plot_output_dir, 'dir')
%     mkdir(plot_output_dir);
%     fprintf('Created directory: ./%s\n', plot_output_dir);
% end

% Frequency band resolution
band_idx = find(strcmpi(band_to_plot, freq_bands(:,1)));
if isempty(band_idx)
    error('Specified frequency band "%s" not found.', band_to_plot);
end
band_range = freq_bands{band_idx, 2};

% Time & frequency vectors (assumes consistent dims everywhere)
[~, n_frex, n_times, ~] = size(all_tf_data.Haptic.Pre.MI);
frex  = logspace(log10(2), log10(80), n_frex);
times = linspace(-3, 4, n_times);

freq_idx      = dsearchn(frex', band_range');
window_idx    = dsearchn(times', topo_time_window');
win_start_idx = window_idx(1);
win_end_idx   = window_idx(2);

% Block sizing (samples)
dt = mean(diff(times)); % seconds per time bin
if isnan(dt) || dt <= 0
    error('Invalid time vector.');
end
block_samps = max(1, round((block_ms/1000) / dt));

% Build block edges covering the window [inclusive start, exclusive end]
block_edges = win_start_idx : block_samps : (win_end_idx+1);
if block_edges(end) ~= (win_end_idx+1)
    block_edges = [block_edges, (win_end_idx+1)];
end
n_blocks = numel(block_edges) - 1;
if n_blocks < 1
    error('Selected window is shorter than one block (%d ms).', block_ms);
end

fprintf('Event: %s | Timepoint: %s | Band: %s\n', event_to_plot, timepoint_to_plot, band_to_plot);
fprintf('Window: [%.3f %.3f] s | dt=%.1f ms | block=%d ms | blocks=%d\n', ...
    topo_time_window(1), topo_time_window(2), dt*1000, block_ms, n_blocks);

%% Helper for grid size (square-ish layout)
calc_grid = @(N) deal(ceil(N/ceil(sqrt(N))), ceil(sqrt(N))); % returns [rows, cols]

%% Plotting Loop (per subject)
groups = {'Haptic','NonHaptic'};

for g = 1:numel(groups)
    group_name   = groups{g};
    data_to_plot = all_tf_data.(group_name).(timepoint_to_plot).(event_to_plot);
    % data dims: [subjects, frex, time, chan]
    [n_subjects, ~, ~, ~] = size(data_to_plot);

    % Pre-compute grid (same per subject)
    [n_rows, n_cols] = calc_grid(n_blocks);

    for s_idx = 1:n_subjects
        % Subject number per your odd/even mapping
        if strcmp(group_name, 'Haptic')
            subject_num = (s_idx * 2) - 1;
        else
            subject_num = s_idx * 2;
        end

        % Create figure for this subject
        figH = figure('color','w');
        sgtitle(sprintf('%s Group — Subj %d — %s %s — %s Band', ...
            group_name, subject_num, timepoint_to_plot, event_to_plot, band_to_plot), ...
            'FontSize', 16, 'FontWeight', 'bold');

        % Pre-average over frequency band (time x chan)
        band_avg = squeeze(mean(data_to_plot(s_idx, freq_idx(1):freq_idx(2), :, :), 2)); % [time, chan]

        % Plot each time block as a subplot
        for b = 1:n_blocks
            blk_start = block_edges(b);
            blk_end   = block_edges(b+1) - 1;                  % inclusive end
            blk_end   = min(blk_end, n_times);                 % clamp
            if blk_end < blk_start, blk_end = blk_start; end   % guard

            % Average over time within block
            topo_data = mean(band_avg(blk_start:blk_end, :), 1); % [1 x chan]

            % Subplot index
            subplot(n_rows, n_cols, b);

            % Plot topography
            topoplot(topo_data, EEG_chlocs, ...
                'maplimits', topo_clim, ...
                'style', 'map', ...
                'electrodes', electrodes_on);

            % Block time annotation
            t0_ms = round(times(blk_start) * 1000);
            t1_ms = round(times(blk_end)   * 1000);
            title(sprintf('%d–%d ms', t0_ms, t1_ms), 'FontSize', 10);
        end

        % Single colorbar for the whole subject figure
        cb = colorbar('Position', [0.92 0.15 0.02 0.7]);
        ylabel(cb, 'Power (dB vs baseline)');
        caxis(topo_clim);

        % Save the figure
        if save_plots
            fig_filename = sprintf('SubjectTopos_%s_Subj%02d_%s_%s_%s_%dmsBlocks.png', ...
                                   group_name, subject_num, timepoint_to_plot, event_to_plot, ...
                                   band_to_plot, block_ms);
            print(figH, fullfile(plot_output_dir, fig_filename), '-dpng', '-r300');
            fprintf('  Saved: %s\n', fig_filename);
        end
    end
end

fprintf('\nPer-subject block-averaged plotting complete!\n');
