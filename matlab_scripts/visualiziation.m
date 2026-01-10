%% STEP 5: Visualization and Reality Check
%  Generates Grand Average plots to inspect the validity of the data.

clear; clc; close all;
load('./data_intermediate/TF_Data_CSD.mat');
load('./data_intermediate/ROI_Mask.mat'); % Need indices

% Define Groups
subs_haptic_idx = find(mod(all_subs, 2) == 1);
subs_non_idx    = find(mod(all_subs, 2) == 0);

% Define Motor Channels for Plotting (Adjust indices!)
chan_left  = 14; % C3
chan_right = 48; % C4
roi_chans  = [14, 48, 15, 49]; 

% Calculate ERD Matrices
ERD_Pre = tf_Pre_MI - tf_Pre_Rest;
ERD_Post = tf_Post_MI - tf_Post_Rest;

%% PLOT 1: Grand Average Time-Frequency (Motor Channels)
% We average over the ROI channels to get a single TF map
GA_Haptic = squeeze(mean(mean(ERD_Pre(subs_haptic_idx, :, :, roi_chans), 1), 4));
GA_Non    = squeeze(mean(mean(ERD_Pre(subs_non_idx, :, :, roi_chans), 1), 4));

figure('Color','w', 'Name', 'Grand Average TF Maps');
subplot(2,1,1);
contourf(times, frex, GA_Haptic, 40, 'linecolor','none');
set(gca, 'clim', [-3 3], 'xlim', [-0.5 2.5]);
colormap(jet); colorbar;
title('Haptic Group (Pre-Training) - Motor Area');
xlabel('Time (s)'); ylabel('Frequency (Hz)');

subplot(2,1,2);
contourf(times, frex, GA_Non, 40, 'linecolor','none');
set(gca, 'clim', [-3 3], 'xlim', [-0.5 2.5]);
colormap(jet); colorbar;
title('Non-Haptic Group (Pre-Training) - Motor Area');
xlabel('Time (s)'); ylabel('Frequency (Hz)');

%% PLOT 2: Temporal Evolution (Mu Band)
% Extract Mu Band Trace (8-13 Hz)
mu_idx = find(frex >= 8 & frex <= 13);
Mu_Haptic = squeeze(mean(mean(ERD_Pre(subs_haptic_idx, mu_idx, :, roi_chans), 2), 4)); % [Sub x Time]
Mu_Non    = squeeze(mean(mean(ERD_Pre(subs_non_idx, mu_idx, :, roi_chans), 2), 4));

% Calculate Mean and SEM (Standard Error)
mean_H = mean(Mu_Haptic, 1);
sem_H  = std(Mu_Haptic, 0, 1) / sqrt(length(subs_haptic_idx));
mean_N = mean(Mu_Non, 1);
sem_N  = std(Mu_Non, 0, 1) / sqrt(length(subs_non_idx));

figure('Color','w', 'Name', 'Mu Band Time Course');
hold on;
% Plot Haptic (Red)
boundedline(times, mean_H, sem_H, 'r', 'alpha');
% Plot Non-Haptic (Blue)
boundedline(times, mean_N, sem_N, 'b', 'alpha');

legend('Haptic (Mean +/- SEM)', 'Non-Haptic');
xlabel('Time (s)'); ylabel('Power (dB)');
title('Mu Rhythm Desynchronization (Pre-Training)');
xlim([-0.5 3]); grid on;
yline(0, 'k--');
xline(0, 'k-');

%% PLOT 3: Topoplots (Spatial Check)
% Requires 'locs.ced' or EEG structure with chanlocs
try
    % Load dummy set for chanlocs
    tmp = pop_loadset(['CBH' sprintf('%04d', all_subs(1)) '_MI_pre_ICA.set'], './epoched/');
    chanlocs = tmp.chanlocs;
    
    % Define Time Window for Topo (e.g., 0.5s to 1.5s)
    t_topo = find(times >= 0.5 & times <= 1.5);
    
    % Get Spatial Data (Avg over Time and Freq)
    Topo_H = squeeze(mean(mean(mean(ERD_Pre(subs_haptic_idx, mu_idx, t_topo, :), 1), 2), 3));
    Topo_N = squeeze(mean(mean(mean(ERD_Pre(subs_non_idx, mu_idx, t_topo, :), 1), 2), 3));

    figure('Color','w', 'Name', 'Mu Topographies');
    subplot(1,2,1);
    topoplot(Topo_H, chanlocs, 'maplimits', [-2 2]);
    title('Haptic Group (Pre)');
    
    subplot(1,2,2);
    topoplot(Topo_N, chanlocs, 'maplimits', [-2 2]);
    title('Non-Haptic Group (Pre)');
    colorbar;
catch
    warning('Could not plot topographies. Check chanlocs path.');
end