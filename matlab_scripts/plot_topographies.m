%% Plot Individual Subject Topographies
%
% This script loads the results from the main time-frequency analysis and
% plots the individual scalp topographies for all subjects in each group
% for a specified condition.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
% clear;
% close all;

%% USER-DEFINED PARAMETERS

% --- Data and Reference Files ---
results_file = 'time_frequency_analysis_ALL_RESULTS.mat';
chlocs_file = 'reference/EEG_chlocs_60.mat';

% --- Plotting Selections ---
event_to_plot = 'MI';       % Choose: 'MI' or 'Rest'
timepoint_to_plot = 'Pre'; % Choose: 'Pre' or 'Post'
band_to_plot = 'Alpha';      % Choose: 'Alpha' or 'Beta'

% --- Topography Parameters ---
freq_bands = {
    'Alpha', [8 13];
    'Beta', [13 30];
    'AB', [8 13]
};
topo_time_window = [0.5 1.5]; % Time window for averaging (in seconds)
topo_clim = [-3 3];           % Color limits for topoplots

% --- Plot Saving Parameters ---
save_plots = true;
plot_output_dir = 'individual_topo_plots';

%% Load Data and Setup
fprintf('Loading data...\n');
if ~exist(results_file, 'file')
    error('Results file not found: %s. Please run the main analysis script first.', results_file);
end
load(results_file); % Loads 'all_tf_data'
load(chlocs_file);  % Loads 'EEG_chlocs'

if save_plots && ~exist(plot_output_dir, 'dir')
    mkdir(plot_output_dir);
    fprintf('Created directory for saving plots: ./%s\n', plot_output_dir);
end

% Find frequency and time indices from parameters
band_idx = find(strcmpi(band_to_plot, freq_bands(:,1)));
if isempty(band_idx)
    error('Specified frequency band "%s" not found in definitions.', band_to_plot);
end
band_range = freq_bands{band_idx, 2};

% Get time and frequency vectors from the data structure
% (Assumes all data has the same time/freq dimensions)
[~, n_frex, n_times, ~] = size(all_tf_data.Haptic.Pre.MI);
frex = logspace(log10(2), log10(80), n_frex);
times = linspace(-3, 4, n_times);

freq_idx = dsearchn(frex', band_range');
topo_time_idx = dsearchn(times', topo_time_window');

fprintf('Plotting Event: %s, Timepoint: %s, Band: %s\n', event_to_plot, timepoint_to_plot, band_to_plot);

%% Plotting Loop
groups = {'Haptic', 'NonHaptic'};

for g = 1:length(groups)
    group_name = groups{g};
    
    % Get the data for the selected condition
    data_to_plot = all_tf_data.(group_name).(timepoint_to_plot).(event_to_plot);
    n_subjects = size(data_to_plot, 1);
    
    % Determine subplot layout
    n_rows = floor(sqrt(n_subjects));
    n_cols = ceil(n_subjects / n_rows);
    
    % Create figure
    figure('color', 'w');
    sgtitle(sprintf('Individual Topographies: %s Group - %s %s (%s Band)', ...
        group_name, timepoint_to_plot, event_to_plot, band_to_plot), 'FontSize', 16, 'FontWeight', 'bold');
    
    % Loop through subjects in the group
    for s_idx = 1:n_subjects
        subplot(n_rows, n_cols, s_idx);
        
        % Calculate topography for this subject
        % Average over subjects(1), frequencies, and times
        topo_data = squeeze(mean(mean(data_to_plot(s_idx, freq_idx(1):freq_idx(2), topo_time_idx(1):topo_time_idx(2), :), 2), 3));
        
        % Plot the topography
        topoplot(topo_data, EEG_chlocs, 'maplimits', topo_clim, 'style', 'map', 'electrodes', 'off');
        
        % Determine subject number
        if strcmp(group_name, 'Haptic')
            subject_num = (s_idx * 2) - 1;
        else
            subject_num = s_idx * 2;
        end
        title(sprintf('Subj %d', subject_num));
    end
    
    % Add a colorbar for the whole figure
    cb = colorbar('Position', [0.92 0.15 0.02 0.7]);
    ylabel(cb, 'Power (dB vs baseline)');
    caxis(topo_clim);
    
    % Save the figure
    if save_plots
        fig_filename = sprintf('IndividualTopos_%s_%s_%s_%s.png', group_name, timepoint_to_plot, event_to_plot, band_to_plot);
        print(gcf, fullfile(plot_output_dir, fig_filename), '-dpng', '-r300');
        fprintf('  Saved plot: %s\n', fig_filename);
    end
end

fprintf('\nPlotting complete!\n');
