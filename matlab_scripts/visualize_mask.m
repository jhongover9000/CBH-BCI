%% VISUALIZATION CHECK: Mask & ROI Verification
% Run this AFTER 'roi_selection.m' to verify your mask aligns with the data.

clear; clc; close all;

% 1. Load Data
% -------------------------------------------------------------------------
fprintf('Loading Data and Mask...\n');
load('./data_intermediate/TF_Data_CSD.mat'); % tf_Pre_MI, tf_Pre_Rest, frex, times
load('./data_intermediate/ROI_Mask.mat');    % mask_refined, times_reduced, t_idx_use

% Load Channel Locations (Needed for Topoplot)
% Adjust path if needed. If you don't have this .mat, you might need to load 
% a raw .set file just to get 'EEG.chanlocs'.
try
    load('reference/EEG_chlocs_60.mat'); % or 'chanlocs' variable
catch
    warning('Channel locations not found. Topography plot will be skipped.');
    chanlocs = [];
end

% 2. Prepare Grand Average Data (Pre-Test Only)
% -------------------------------------------------------------------------
% Calculate ERD (MI - Rest) for visualization
% Dimensions: [Sub x Freq x Time x Chan]
ERD_Full = tf_Pre_MI - tf_Pre_Rest; 

% Align Data dimensions to Mask dimensions (Downsampling)
% The mask uses 'times_reduced', so we must slice the data to match.
ERD_Reduced = ERD_Full(:, :, t_idx_use, :);

% 3. Define What to Plot
% -------------------------------------------------------------------------
% We need to know which Channels and Frequencies are "Significant" 
% based on the mask to average the data correctly.

if sum(mask_refined(:)) == 0
    error('Mask is empty! No significant clusters to visualize.');
end

% Find indices of the mask
[f_ind, t_ind, ch_ind] = ind2sub(size(mask_refined), find(mask_refined));
active_chans = unique(ch_ind);
active_freqs = unique(f_ind);

% Compute Grand Average (Mean across all subjects)
GA_ERD = squeeze(mean(ERD_Reduced, 1)); % [Freq x Time x Chan]

% 4. PLOT 1: Time-Frequency Spectrogram (ROI Channels)
% -------------------------------------------------------------------------
figure('Name', 'ROI Visualization Dashboard', 'Color', 'w', 'Position', [100 100 1200 800]);

subplot(2, 2, 1);
% Average data across the ACTIVE CHANNELS only
plot_data = squeeze(mean(GA_ERD(:, :, active_chans), 3));

% Collapse Mask across channels for 2D contour (Freq x Time)
% If any channel is significant at that pixel, we draw it.
mask_2d = squeeze(max(mask_refined, [], 3)); 

% Plot Spectrogram
imagesc(times_reduced, frex, plot_data);
axis xy; colormap(jet); 
clim([-3 3]); % Adjust contrast limits (dB)
hold on;

% Overlay the Mask as a Contour
% The '1' indicates drawing the level where logic goes 0->1
contour(times_reduced, frex, double(mask_2d), 1, 'k', 'LineWidth', 2);

% Aesthetics
title(sprintf('Grand Average ERD (ROI Channels: %s)', num2str(active_chans')));
xlabel('Time (ms)'); ylabel('Frequency (Hz)');
c = colorbar; c.Label.String = 'Power (dB)';
xline(0, '--k'); xline(1000, '--k', 'End of Motor Task');


% 5. PLOT 2: Time Course of Mu Band (500-1000ms check)
% -------------------------------------------------------------------------
subplot(2, 2, 2);

% Average across Active Frequencies (Mu/Beta) and Active Channels
tc_data = squeeze(mean(mean(ERD_Reduced(:, :, active_chans), 1), 3)); 
% Now [Subjects x Time]

% Calculate Mean and Standard Error
mu_mean = mean(tc_data, 1);
mu_ste  = std(tc_data, 0, 1) / sqrt(size(ERD_Reduced, 1));

% Plot Shaded Error Bar area
fill([times_reduced, fliplr(times_reduced)], ...
     [mu_mean + mu_ste, fliplr(mu_mean - mu_ste)], ...
     [0.8 0.8 0.8], 'EdgeColor', 'none'); hold on;

% Plot Mean Line
plot(times_reduced, mu_mean, 'b', 'LineWidth', 2);

% Highlight the Significant Time points from Mask
% We collapse the mask over Freq and Chan to see "When is it significant?"
mask_time = squeeze(max(max(mask_refined, [], 1), [], 3));
area_h = area(times_reduced, mask_time * min(mu_mean)*1.1); % Draw at bottom
area_h.FaceColor = 'r'; area_h.FaceAlpha = 0.2; area_h.EdgeColor = 'none';

% Aesthetics
yline(0, 'k'); xline(0, 'k');
xlim([-500 2000]);
title('Mu Rhythm Time Course (Masked Region)');
xlabel('Time (ms)'); ylabel('ERD (dB)');
legend('Std Err', 'Mean ERD', 'Significant Time Window');


% 6. PLOT 3: Topography (Spatial Check)
% -------------------------------------------------------------------------
subplot(2, 2, 3);

if ~isempty(chanlocs)
    % Average over the Significant Time and Frequency windows
    topo_data = squeeze(mean(mean(mean(ERD_Reduced(:, active_freqs, mask_time==1, :), 1), 2), 3));
    
    topoplot(topo_data, chanlocs, 'maplimits', [-2 2], 'electrodes', 'on');
    title('Topography of Masked Region');
    
    % Highlight the channels selected by the mask
    hold on;
    % Often we need to find the XY coords of selected channels
    % This is a rough visual indicator
    if isfield(chanlocs, 'X')
        plot([chanlocs(active_chans).Y], [chanlocs(active_chans).X], ...
             'o', 'MarkerSize', 10, 'LineWidth', 2, 'Color', 'k');
    end
else
    text(0.5, 0.5, 'No Channel Locs Found', 'HorizontalAlignment', 'center');
end

% 7. Summary Text
% -------------------------------------------------------------------------
subplot(2, 2, 4);
axis off;
text(0, 0.8, sprintf('Cluster Stats Summary:'), 'FontWeight', 'bold');
text(0, 0.6, sprintf('Freq Range: %.1f - %.1f Hz', min(frex(active_freqs)), max(frex(active_freqs))));
text(0, 0.5, sprintf('Time Range: %.0f - %.0f ms', min(times_reduced(mask_time)), max(times_reduced(mask_time))));
text(0, 0.4, sprintf('Channels Included: %d', length(active_chans)));
text(0, 0.2, 'If the plot (Top-Left) shows the Black Contour covering');
text(0, 0.1, 'the Blue ERD blob, your mask is correct.');