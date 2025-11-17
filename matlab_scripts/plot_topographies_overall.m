%% Setup and Load Data
clc;
% clear;
close all;
%% Load Channels
% Make sure this file is in your MATLAB path
% load('./reference/EEG_chlocs_60.mat'); % Loads 'EEG_chlocs'
%% Load Data
fprintf('Loading analysis data...\n');
load('time_frequency_analysis_ALL_RESULTS_ICA.mat'); % Loads 'all_tf_data'

%% Initialize
% --- Define key parameters from the analysis script
SR = 250; 
min_freq = 2;
max_freq = 80;
num_frex = max_freq - min_freq;
frex = logspace(log10(min_freq), log10(max_freq), num_frex);
epoch_period = [-5 5];
% Create time vector in SECONDS
times_sec = linspace(epoch_period(1), epoch_period(2), diff(epoch_period) * SR);% Create time vector in MILLISECONDS for plotting
times = times_sec * 1000;
nTimes = length(times);
% Extract channel names ('vals') from EEG_chlocs struct
vals = {EEG_chlocs.labels};
fprintf('Data loaded and parameters set.\n');

% --- Definitions needed for new exclusion logic ---
% These must match your analysis script (time_freq_analysis.m)
nSubject_total = 44; % Total number of subjects (e.g., 44)
events = {'MI';
          % 'Rest'
    }; % Events processed
timepoints = {'Pre', 'Post'}; % Timepoints processed
% --------------------------------------------------

% Subject Exclusion
subjects_to_exclude = [1,2,4,33];

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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Extract Data for Plotting
% Select the event type you want to plot (must match your analysis script)
event_to_plot = 'MI'; % Or 'Rest'
fprintf('Extracting data for event: %s\n', event_to_plot);
% Create variables that map to the plotting script's 'tf_mX_db_sub'
% These variables now hold: (subjects, frequencies, timepoints, channels)
data_C1 = all_tf_data.Haptic.Pre.(event_to_plot);    % Condition 1: Haptic Pre
data_C2 = all_tf_data.Haptic.Post.(event_to_plot);   % Condition 2: Haptic Post
data_C3 = all_tf_data.NonHaptic.Pre.(event_to_plot); % Condition 3: NonHaptic Pre
data_C4 = all_tf_data.NonHaptic.Post.(event_to_plot);% Condition 4: NonHaptic Post

%% Topo plots avg over all subjects (Baseline and Time-Series)
fprintf('Generating combined topography plot...\n');
% --- Define Frequencies, Limits, and Colors ---
% Find frequency *indices*
f_range_topo = [8 13];
% f_range_topo = [13 30];
% f_range_topo = [8 30];

f_idx_topo(1) = dsearchn(frex', f_range_topo(1));
f_idx_topo(2) = dsearchn(frex', f_range_topo(2));
fprintf('Topoplot frequency range: %.2f Hz to %.2f Hz (indices %d to %d)\n', ...
        frex(f_idx_topo(1)), frex(f_idx_topo(2)), f_idx_topo(1), f_idx_topo(2));
z1 = -2; % Z-limit min
z2 = 2;  % Z-limit max
c = jet;
% --- Define Time-Series Parameters ---
n_steps = 10;  % Number of time steps to plot 
step_ms = 250; % Step size in milliseconds 
% Convert step_ms to number of time *samples*
step_samples = round(step_ms / (1000/SR));
% --- Create the Combined Figure ---
% We will have 4 rows
% We will have n_steps + 1 columns (1 for baseline, n_steps for time-series)
n_cols = n_steps + 1;
% --- MODIFIED: Figure title is now dynamic based on f_range_topo ---
figure_title = sprintf('Baseline and Time-Series Topoplots (%.0f-%.0f Hz)', f_range_topo(1), f_range_topo(2));
figure('Name', figure_title)
% -----------------------------------------------------------------
set(gcf, 'color', 'w', 'Position', [10 10 1800 800]); % Make a wide figure
% --- 1. Plot Baseline Topographies (Column 1) ---
% Define baseline period (must match analysis script)
baseline_period_sec = [-0.8 -0.2]; % From time_freq_analysis.m
base_idx(1) = dsearchn(times_sec', baseline_period_sec(1));
base_idx(2) = dsearchn(times_sec', baseline_period_sec(2));
fprintf('Plotting baseline topographies for time: %.2f s to %.2f s (indices %d to %d)\n', ...
        times_sec(base_idx(1)), times_sec(base_idx(2)), base_idx(1), base_idx(2));
% Plot 1: Haptic Pre (Row 1, Col 1)
subplot(4, n_cols, 1) % 1st plot
temp_base1 = squeeze(mean(mean(mean(data_C1(:, f_idx_topo(1):f_idx_topo(2), base_idx(1):base_idx(2), :), 1), 2), 3));
topoplot(temp_base1, EEG_chlocs,'maplimits',[z1 z2],'electrodes','off','colormap',c,'style','map' )
title(sprintf('Baseline\nHaptic Pre'))
% Plot 2: Haptic Post (Row 2, Col 1)
subplot(4, n_cols, 1 + n_cols) % 1st plot in 2nd row
temp_base2 = squeeze(mean(mean(mean(data_C2(:, f_idx_topo(1):f_idx_topo(2), base_idx(1):base_idx(2), :), 1), 2), 3));
topoplot(temp_base2, EEG_chlocs,'maplimits',[z1 z2],'electrodes','off','colormap',c,'style','map' )
title('Haptic Post')
% Plot 3: NonHaptic Pre (Row 3, Col 1)
subplot(4, n_cols, 1 + (2 * n_cols)) % 1st plot in 3rd row
temp_base3 = squeeze(mean(mean(mean(data_C3(:, f_idx_topo(1):f_idx_topo(2), base_idx(1):base_idx(2), :), 1), 2), 3));
topoplot(temp_base3, EEG_chlocs,'maplimits',[z1 z2],'electrodes','off','colormap',c,'style','map' )
title('NonHaptic Pre')
% Plot 4: NonHaptic Post (Row 4, Col 1)
subplot(4, n_cols, 1 + (3 * n_cols)) % 1st plot in 4th row
temp_base4 = squeeze(mean(mean(mean(data_C4(:, f_idx_topo(1):f_idx_topo(2), base_idx(1):base_idx(2), :), 1), 2), 3));
topoplot(temp_base4, EEG_chlocs,'maplimits',[z1 z2],'electrodes','off','colormap',c,'style','map' )
title('NonHaptic Post')
% --- 2. Plot Time-Series Topographies (Columns 2 to n_cols) ---
% Find index closest to 0 ms
[~, t0] = min(abs(times - 1));
fprintf('Plotting time-series from %.0f ms...\n', times(t0));
for i = 1:n_steps
    % Calculate time indices for this step
    t_start = t0 + ((i-1) * step_samples);
    t_end   = t0 + (i * step_samples) - 1;
    
    % Ensure indices are within bounds
    if t_end > nTimes
        fprintf('Reached end of time window, stopping topo plot.\n');
        break; 
    end
    
    % --- Calculate plot index ---
    % 'i+1' is the current column number
    plot_idx_r1 = i + 1;                    % Row 1
    plot_idx_r2 = i + 1 + n_cols;           % Row 2
    plot_idx_r3 = i + 1 + (2 * n_cols);     % Row 3
    plot_idx_r4 = i + 1 + (3 * n_cols);     % Row 4
    
    % --- Create title based on ideal time bin size ---
    label_bin_start_ms = (i-1) * step_ms;
    label_bin_end_ms   = i * step_ms;
    time_bin_title     = sprintf('%d-%d ms', label_bin_start_ms, label_bin_end_ms);
    
    % Subplot 1: Haptic Pre (data_C1)
    subplot(4, n_cols, plot_idx_r1)
    temp1 = squeeze(mean(mean(mean(data_C1(:, f_idx_topo(1):f_idx_topo(2), t_start:t_end, :), 1), 2), 3));
    topoplot(temp1, EEG_chlocs,'maplimits',[z1 z2],'electrodes','off','colormap',c,'style','both' )
    title(time_bin_title) % Use the new time bin title
        
    % Subplot 2: Haptic Post (data_C2)
    subplot(4, n_cols, plot_idx_r2)
    temp2 = squeeze(mean(mean(mean(data_C2(:, f_idx_topo(1):f_idx_topo(2), t_start:t_end, :), 1), 2), 3));
    topoplot(temp2, EEG_chlocs,'maplimits',[z1 z2],'electrodes','off','colormap',c,'style','both' )
    
    % Subplot 3: NonHaptic Pre (data_C3)
    subplot(4, n_cols, plot_idx_r3)
    temp3 = squeeze(mean(mean(mean(data_C3(:, f_idx_topo(1):f_idx_topo(2), t_start:t_end, :), 1), 2), 3));
    topoplot(temp3, EEG_chlocs,'maplimits',[z1 z2],'electrodes','off','colormap',c,'style','both' )
    
    % Subplot 4: NonHaptic Post (data_C4)
    subplot(4, n_cols, plot_idx_r4)
    temp4 = squeeze(mean(mean(mean(data_C4(:, f_idx_topo(1):f_idx_topo(2), t_start:t_end, :), 1), 2), 3));
    topoplot(temp4, EEG_chlocs,'maplimits',[z1 z2],'electrodes','off','colormap',c,'style','both' )
end
fprintf('Topography plot complete.\n');




%% Plot ERSP traces for all channels
nCh = 60;
% Find frequency *indices*
f_range_ersp = [8 30]; % Original script used f=[2 8]
f_idx_ersp(1) = dsearchn(frex', f_range_ersp(1));
f_idx_ersp(2) = dsearchn(frex', f_range_ersp(2));
fprintf('ERSP trace frequency range: %.2f Hz to %.2f Hz (indices %d to %d)\n', ...
        frex(f_idx_ersp(1)), frex(f_idx_ersp(2)), f_idx_ersp(1), f_idx_ersp(2));
figure('Name', 'ERSP Traces per Channel') ;
for plotId = 1 : nCh
    
    subplot(6, 10, plotId) ;
    
    % Average over subjects (1) and frequencies (2)
    temp1 = squeeze(mean(mean(data_C1(:, f_idx_ersp(1):f_idx_ersp(2), :, plotId), 1), 2));
    plot(times, temp1, 'b', 'LineWidth', 1) % Haptic Pre (Blue)
    hold on
    
    temp2 = squeeze(mean(mean(data_C2(:, f_idx_ersp(1):f_idx_ersp(2), :, plotId), 1), 2));
    plot(times, temp2, 'c', 'LineWidth', 1) % Haptic Post (Cyan)
    
    temp3 = squeeze(mean(mean(data_C3(:, f_idx_ersp(1):f_idx_ersp(2), :, plotId), 1), 2));
    plot(times, temp3, 'r', 'LineWidth', 1) % NonHaptic Pre (Red)
    
    temp4 = squeeze(mean(mean(data_C4(:, f_idx_ersp(1):f_idx_ersp(2), :, plotId), 1), 2));
    plot(times, temp4, 'm', 'LineWidth', 1) % NonHaptic Post (Magenta)
    
    xlim([-500 1000])
    xline(0, '--k');
    title(vals{plotId})
end
% Add a legend to the figure
lgd = legend('Hap Pre', 'Hap Post', 'NonHap Pre', 'NonHap Post');
lgd.Position = [0.5, 0.01, 0.1, 0.05]; % [x, y, width, height]
lgd.Orientation = 'horizontal';
set(gcf, 'color', 'w');
%% Plotting TF plot averaged over subjects
% Select channels to average over
chs={'CP1','CP3','C3','C5','C1','CP5'};
% chs={'FC1', 'C3', 'CP3', 'C1'};
idx=[];
ctr=1;
for i=chs
  idx(ctr)=find(ismember(vals, i));
  ctr=ctr+1;
end
fprintf('Plotting TF for %d channels: %s\n', length(chs), strjoin(chs, ', '));
% This figure plots the 4 absolute conditions
figure('Name', 'Time-Frequency Plots (Absolute Power)')
colormap(jet)
climdb = [-2 2]; % Color limits
% --- Subplot 1: Haptic Pre (data_C1) ---
subplot(2,2,1)
% Average over subjects (1) and channels (4)
temp1 = squeeze(mean(mean(data_C1(:, :, :, idx), 1), 4));
contourf(times,frex,temp1,40,'linecolor','none')
set(gca,'clim',climdb,'ydir','normal','xlim',[-200 1500],'yscale','log' )
ylim([1 50 ])
xlabel('Time (ms)')
ylabel('Frequencies (Hz)')
set(gca,'FontSize',16, 'FontWeight','Bold')
title('Haptic Pre')
xline(0,'-.r','LineWidth',2)
yticks([ 3 8 13 30 50])
% --- Subplot 2: Haptic Post (data_C2) ---
subplot(2,2,2)
temp2 = squeeze(mean(mean(data_C2(:, :, :, idx), 1), 4));
contourf(times,frex,temp2,40,'linecolor','none')
set(gca,'clim',climdb,'ydir','normal','xlim',[-200 1500] ,'yscale','log' )
ylim([1 50 ])
xlabel('Time (ms)')
ylabel('Frequencies (Hz)')
set(gca,'FontSize',16, 'FontWeight','Bold')
title('Haptic Post')
xline(0,'-.r','LineWidth',2)
yticks([3 8 13 30 50 ])
% --- Subplot 3: NonHaptic Pre (data_C3) ---
subplot(2,2,3)
temp3 = squeeze(mean(mean(data_C3(:, :, :, idx), 1), 4));
contourf(times,frex,temp3,40,'linecolor','none')
set(gca,'clim',climdb,'ydir','normal','xlim',[-200 1500] ,'yscale','log')
ylim([1 50])
xlabel('Time (ms)')
ylabel('Frequencies (Hz)')
set(gca,'FontSize',16, 'FontWeight','Bold')
title('NonHaptic Pre')
xline(0,'-.r','LineWidth',2)
yticks([3 8 13 30 50])
% --- Subplot 4: NonHaptic Post (data_C4) ---
subplot(2,2,4)
temp4 = squeeze(mean(mean(data_C4(:, :, :, idx), 1), 4));
contourf(times,frex,temp4,40,'linecolor','none')
set(gca,'clim',climdb,'ydir','normal','xlim',[-200 1500] ,'yscale','log' )
ylim([1 50 ])
xlabel('Time (ms)')
ylabel('Frequencies (Hz)')
set(gca,'FontSize',16, 'FontWeight','Bold')
title('NonHaptic Post')
xline(0,'-.r','LineWidth',2)
yticks([  3 8 13 30 50 ])
set(gcf, 'color', 'w');
%% BONUS: Plotting TF *Comparisons* (Differences)
% This new figure shows the comparisons you asked for.
% It uses the 'temp' variables calculated in the section above.
figure('Name', 'Time-Frequency Plots (Comparisons/Differences)')
colormap(jet)
climdb_diff = [-1 1]; % Use a different, zero-centered limit for differences
% --- Subplot 1: Haptic (Post - Pre) ---
subplot(2,2,1)
contourf(times, frex, temp2 - temp1, 40, 'linecolor', 'none')
set(gca,'clim',climdb_diff,'ydir','normal','xlim',[-200 1500],'yscale','log' )
ylim([1 50 ])
xlabel('Time (ms)'), ylabel('Frequencies (Hz)')
set(gca,'FontSize',16, 'FontWeight','Bold')
title('Haptic: Post - Pre')
xline(0,'-.r','LineWidth',2)
yticks([ 3 8 13 30 50])
colorbar;
% --- Subplot 2: NonHaptic (Post - Pre) ---
subplot(2,2,2)
contourf(times, frex, temp4 - temp3, 40, 'linecolor', 'none')
set(gca,'clim',climdb_diff,'ydir','normal','xlim',[-200 1500] ,'yscale','log' )
ylim([1 50 ])
xlabel('Time (ms)'), ylabel('Frequencies (Hz)')
set(gca,'FontSize',16, 'FontWeight','Bold')
title('NonHaptic: Post - Pre')
xline(0,'-.r','LineWidth',2)
yticks([3 8 13 30 50 ])
colorbar;
% --- Subplot 3: Pre (Haptic - NonHaptic) ---
subplot(2,2,3)
contourf(times, frex, temp1 - temp3, 40, 'linecolor', 'none')
set(gca,'clim',climdb_diff,'ydir','normal','xlim',[-200 1500] ,'yscale','log')
ylim([1 50])
xlabel('Time (ms)'), ylabel('Frequencies (Hz)')
set(gca,'FontSize',16, 'FontWeight','Bold')
title('Pre-Test: Haptic - NonHaptic')
xline(0,'-.r','LineWidth',2)
yticks([3 8 13 30 50])
colorbar;
% --- Subplot 4: Post (Haptic - NonHaptic) ---
subplot(2,2,4)
contourf(times, frex, temp2 - temp4, 40, 'linecolor', 'none')
set(gca,'clim',climdb_diff,'ydir','normal','xlim',[-200 1500] ,'yscale','log' )
ylim([1 50 ])
xlabel('Time (ms)'), ylabel('Frequencies (Hz)')
set(gca,'FontSize',16, 'FontWeight','Bold')
title('Post-Test: Haptic - NonHaptic')
xline(0,'-.r','LineWidth',2)
yticks([  3 8 13 30 50 ])
colorbar;
set(gcf, 'color', 'w');