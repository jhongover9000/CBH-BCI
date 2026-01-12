%% STEP 2: Data-Driven ROI Selection (Fully Data-Driven Version)
%  - Loads High-Res TF Data
%  - CROPS time window
%  - DOWNSAMPLES time
%  - Runs CBPT (Cluster Based Permutation Testing)
%  - DEFINES ROI based PURELY on statistical clusters (No manual cropping)

clear; clc;
load('./data_intermediate/TF_Data_CSD.mat'); % Loads tf_*, frex, times

%% 1. DATA REDUCTION
fprintf('Reducing data resolution for Statistics...\n');

% A. Define Time Window (Broad window to catch any effects)
time_lims = [-0.5 2.5]; 
t_idx_win = find(times >= time_lims(1) & times <= time_lims(2));

% B. Downsample (50Hz resolution is sufficient for induced power)
ds_factor = 5; 
t_idx_use = t_idx_win(1:ds_factor:end);
times_reduced = times(t_idx_use);

fprintf('  Original Time Points: %d\n', length(times));
fprintf('  Reduced Time Points:  %d\n', length(times_reduced));

% C. Apply to Data
% Data format: [Subjects x Freqs x Times x Chans]
data_MI   = tf_Pre_MI(:, :, t_idx_use, :);
data_Rest = tf_Pre_Rest(:, :, t_idx_use, :);

% D. Permute for statcond (needs [Freq x Time x Chan x Sub])
data_MI   = permute(data_MI,   [2, 3, 4, 1]);
data_Rest = permute(data_Rest, [2, 3, 4, 1]);

%% 2. RUN CLUSTER STATISTICS (CBPT)
fprintf('Running Permutation Statistics (This may take time)...\n');

% Using 'statcond' from EEGLAB
% 'cluster' = 'on' -> Returns corrected p-values for clusters
[tvals, df, pvals, surrog] = statcond({data_MI, data_Rest}, ...
    'method', 'perm', ...
    'naccu', 1000, ...      % 1000 permutations
    'paired', 'on', ...
    'cluster', 'on', ...    % Cluster correction enabled
    'alpha', 0.1, ...      % Significance level
    'verbose', 'on');

%% 3. CREATE DATA-DRIVEN MASK
% In a fully data-driven approach, we TRUST the cluster correction.
% We do not crop by frequency or channel manually.

mask_refined = pvals < 0.05;

% Check if we found anything
n_sig_pixels = sum(mask_refined(:));
fprintf('\n--- STATISTICS RESULTS ---\n');
if n_sig_pixels == 0
    warning('NO SIGNIFICANT CLUSTERS FOUND at p < 0.05.');
    fprintf('  Try relaxing alpha in statcond to 0.1 to see trends.\n');
    
    % FALLBACK: To prevent crashes, define a dummy manual ROI
    % (Standard Mu / Motor) just so variables exist.
    fprintf('  -> Defaulting to Standard Mu/Motor ROI for compatibility.\n');
    mu_idx = find(frex >= 8 & frex <= 13);
    % Standard C3/C4 indices (Approximation, check your cap!)
    motor_chans_idx = [12 13 45 46]; 
else
    fprintf('  FOUND Significant Cluster(s)! Total voxels: %d\n', n_sig_pixels);
    
    % 4. EXTRACT FEATURES FROM THE CLUSTER
    % Instead of defining "Mu" and "Motor" manually, we ask:
    % "What frequencies and channels are in this cluster?"
    
    % Find indices of all true pixels in the 3D mask [Freq, Time, Chan]
    [f_ind, t_ind, ch_ind] = ind2sub(size(mask_refined), find(mask_refined));
    
    % Update the indices variables to match the DATA
    mu_idx = unique(f_ind);          % The significant frequency bins
    motor_chans_idx = unique(ch_ind);% The significant channels
    
    % Display what was found
    found_freqs = frex(mu_idx);
    fprintf('  Significant Freq Range: %.1f Hz - %.1f Hz\n', min(found_freqs), max(found_freqs));
    fprintf('  Significant Channels:   %d found (Indices: %s)\n', length(motor_chans_idx), num2str(motor_chans_idx'));
end

%% 5. SAVE
% We save the mask AND the indices derived from it.
% This ensures 'stat_test.m' extracts data from the *actual* active region.

save('./data_intermediate/ROI_Mask.mat', ...
    'mask_refined', ...
    'mu_idx', ...           % Now represents the ACTIVE frequencies
    'motor_chans_idx', ...  % Now represents the ACTIVE channels
    'times_reduced', ...
    'ds_factor', ...
    't_idx_use');

fprintf('ROI Mask saved. Ready for Step 3 (stat_test.m).\n');