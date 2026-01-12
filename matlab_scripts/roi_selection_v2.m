%% STEP 2: Data-Driven ROI Selection (One-Sample / Baseline Approach)
%  - Ignores 'Rest' file (assumed contaminated).
%  - Tests if MI Data is significantly different from 0 (Baseline).
%  - 0 represents the Pre-Cue interval because data is in dB.

clear; clc;
load('./data_intermediate/TF_Data_CSD.mat'); % Loads tf_*, frex, times

%% 1. DATA REDUCTION
fprintf('Reducing data resolution for Statistics...\n');

% A. Define Time Window (Motor Task Only)
% We focus on the active period (0 to 3s) vs Baseline (implicitly 0)
time_lims = [0 2.5]; 
t_idx_win = find(times >= time_lims(1) & times <= time_lims(2));

ds_factor = 5; 
t_idx_use = t_idx_win(1:ds_factor:end);
times_reduced = times(t_idx_use);

% B. Prepare Data (MI Only)
% Data format: [Subjects x Freqs x Times x Chans]
data_MI = tf_Pre_MI(:, :, t_idx_use, :);

% Permute for statcond: [Freq x Time x Chan x Sub]
data_MI = permute(data_MI, [2, 3, 4, 1]);

%% 2. RUN CLUSTER STATISTICS (One-Sample Test)
fprintf('Running One-Sample Permutation Test (MI vs 0)...\n');

% We pass a SINGLE cell array. statcond will test this against 0.
% Note: 'paired' is 'off' because we don't have two conditions.
[tvals, df, pvals, surrog] = statcond({data_MI}, ...
    'method', 'perm', ...
    'naccu', 1000, ...      
    'cluster', 'on', ...    
    'alpha', 0.05, ...     
    'verbose', 'on');

%% 3. CREATE MASK
% This mask shows where MI is significantly different from Baseline.
mask_raw = pvals < 0.05;

% 4. FILTER FOR MOTOR FEATURES (CRITICAL STEP)
% Since we are testing against 0, you will likely get Visual Cortex activity 
% (Alpha) and Attention activity (Theta). We MUST restrict this to Mu/Beta 
% to avoid selecting visual processing as your ROI.

fprintf('\nRefining Mask to Motor Frequencies (Mu/Beta)...\n');

mu_beta_idx = find(frex >= 8 & frex <= 30); % 8-30 Hz
mask_refined = zeros(size(mask_raw));

% Only keep significant pixels that are within Mu/Beta range
mask_refined(mu_beta_idx, :, :) = mask_raw(mu_beta_idx, :, :);

% Check results
n_sig_pixels = sum(mask_refined(:));

if n_sig_pixels == 0
    warning('No significant pixels found even with One-Sample test.');
    % Fallback logic...
    mu_idx = find(frex >= 8 & frex <= 13);
    motor_chans_idx = [12 13 45 46]; 
else
    fprintf('  FOUND Significant Cluster(s)! Total voxels: %d\n', n_sig_pixels);
    
    [f_ind, t_ind, ch_ind] = ind2sub(size(mask_refined), find(mask_refined));
    mu_idx = unique(f_ind);          
    motor_chans_idx = unique(ch_ind);
    
    found_freqs = frex(mu_idx);
    fprintf('  Significant Freq Range: %.1f Hz - %.1f Hz\n', min(found_freqs), max(found_freqs));
    fprintf('  Significant Channels:   %d found\n', length(motor_chans_idx));
end

%% 5. SAVE
save('./data_intermediate/ROI_Mask.mat', ...
    'mask_refined', 'mu_idx', 'motor_chans_idx', 'times_reduced', 'ds_factor', 't_idx_use');
fprintf('ROI Mask saved.\n');