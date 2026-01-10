%% STEP 2: Data-Driven ROI Selection (Memory Optimized)
%  - Loads High-Res TF Data
%  - CROPS time window (to relevant MI period)
%  - DOWNSAMPLES time (to reduce RAM usage for CBPT)
%  - Runs CBPT to find ROI

clear; clc;
load('./data_intermediate/TF_Data_CSD.mat'); % Loads tf_*, frex, times

%% 1. DATA REDUCTION (CRITICAL FIX)
fprintf('Reducing data resolution for Statistics...\n');

% A. Define Time Window of Interest (e.g., -1s to 3s)
% We don't need the tails (-5s or +5s) for the ROI check
time_lims = [-1 3]; 
t_idx_win = find(times >= time_lims(1) & times <= time_lims(2));

% B. Define Downsampling Factor
% Induced power is slow. 50Hz (20ms) resolution is plenty.
% Current SR = 250Hz. Factor of 5 = 50Hz.
ds_factor = 5; 

% Select indices: Inside window, stepping by ds_factor
t_idx_use = t_idx_win(1:ds_factor:end);
times_reduced = times(t_idx_use);

fprintf('  Original Time Points: %d\n', length(times));
fprintf('  Reduced Time Points:  %d\n', length(times_reduced));

% C. Subset the Data [Sub x Freq x Time x Chan]
% We overwrite the variables to clear RAM
tf_Pre_MI   = tf_Pre_MI(:, :, t_idx_use, :);
tf_Pre_Rest = tf_Pre_Rest(:, :, t_idx_use, :);

%% 2. PREPARE FOR STATCOND
% Permute for statcond: [Freq x Time x Chan x Sub]
data_MI   = permute(tf_Pre_MI,   [2, 3, 4, 1]); 
data_Rest = permute(tf_Pre_Rest, [2, 3, 4, 1]);

% Clear originals to free memory immediately
clear tf_Pre_MI tf_Pre_Rest;

%% 3. RUN CLUSTER PERMUTATION (MI vs Rest)
fprintf('Running CBPT on Reduced Data (N=%d)...\n', size(data_MI, 4));

% Note: 'naccu' determines p-value precision. 
% If 1000 is still slow/heavy, try 500 for a quick check, but 1000 is standard.
[tvals, df, pvals, surrog] = statcond({data_MI, data_Rest}, ...
    'method', 'perm', ...
    'naccu', 1000, ...      % Number of permutations
    'paired', 'on', ...
    'cluster', 'on', ...    % Enable cluster correction
    'verbose', 'on');

% Create Logical Mask (p < 0.05)
mask_raw = pvals < 0.05;

%% 4. REFINE MASK
% Use 'frex' (loaded from .mat)
mu_idx   = find(frex >= 8 & frex <= 13);
beta_idx = find(frex >= 13 & frex <= 30);

% Define Motor Channels (Update these indices based on your cap!)
% E.g., if using 60 chans, roughly C3=14, C4=48, etc.
% You must look at {chanlocs.labels} to be sure.
motor_chans_idx = [12, 13, 14, 15, 16, 44, 45, 46, 47, 48]; % Approx Motor Area

mask_refined = zeros(size(mask_raw));

% Find t=0 in the REDUCED time vector
t_zero_red = find(times_reduced >= 0, 1);

% Keep Mu/Beta clusters in motor channels only
mask_refined(mu_idx, t_zero_red:end, motor_chans_idx) = mask_raw(mu_idx, t_zero_red:end, motor_chans_idx);
mask_refined(beta_idx, t_zero_red:end, motor_chans_idx) = mask_raw(beta_idx, t_zero_red:end, motor_chans_idx);

%% 5. SAVE
% IMPORTANT: We save 'times_reduced' so Step 3 knows the new time axis
save('./data_intermediate/ROI_Mask.mat', 'mask_refined', 'motor_chans_idx', 'mu_idx', 'beta_idx', 'times_reduced');
disp('Step 2 Complete: ROI Mask Generated (Memory Optimized).');