%% STEP 3: Statistical Extraction & Temporal Dynamics
%  - Loads processed Time-Freq data (from Step 1)
%  - REDUCES dimensions to match the ROI Mask (Critical Fix)
%  - Applies Odd/Even Group Logic
%  - Calculates ERD Contrast (MI - Rest)
%  - Computes Temporal Metrics (Onset, Duration, Slope)
%  - Exports to CSV

clear; clc;
load('./data_intermediate/TF_Data_CSD.mat'); % Loads tf_*, all_subs, frex, times
load('./data_intermediate/ROI_Mask.mat');    % Loads mask_refined, motor_chans_idx, mu_idx, times_reduced

%% 1. DATA ALIGNMENT (CRITICAL FIX)
% The mask was created on DOWNSAMPLED data. We must apply the same 
% downsampling to the data here so dimensions match [Freq x Time x Chan].

fprintf('Aligning data dimensions to ROI Mask...\n');

% Parameters must match "roi_selection.m" exactly
time_lims = [-0.5 2.5]; 
ds_factor = 5; 

% Re-calculate indices to ensure perfect alignment
t_idx_win = find(times >= time_lims(1) & times <= time_lims(2));
t_idx_use = t_idx_win(1:ds_factor:end);

% Check consistency
if length(t_idx_use) ~= size(mask_refined, 2)
    error('Dimension Mismatch! The mask time points (%d) do not match the downsampling (%d). Check ds_factor.', ...
        size(mask_refined, 2), length(t_idx_use));
end

% Apply Reduction to all conditions
tf_Pre_MI    = tf_Pre_MI(:, :, t_idx_use, :);
tf_Pre_Rest  = tf_Pre_Rest(:, :, t_idx_use, :);
tf_Post_MI   = tf_Post_MI(:, :, t_idx_use, :);
tf_Post_Rest = tf_Post_Rest(:, :, t_idx_use, :);

% Update the time vector to the reduced version
times = times(t_idx_use); 

%% 2. CONFIGURATION (Group Assignment)
subjects_haptic = all_subs(mod(all_subs, 2) == 1); % Odd IDs = Haptic
subjects_non    = all_subs(mod(all_subs, 2) == 0); % Even IDs = Non-Haptic

fprintf('Stats Configuration:\n');
fprintf('  Haptic Group (Odd): N=%d\n', length(subjects_haptic));
fprintf('  Non-Haptic Group (Even): N=%d\n', length(subjects_non));

%% 3. CALCULATE ERD MAPS (MI - Rest)
% Subtracting Log Power (dB) gives the contrast in dB.
ERD_Pre  = tf_Pre_MI - tf_Pre_Rest;   
ERD_Post = tf_Post_MI - tf_Post_Rest;

results = table();
temporal = table();

% Check if mask is empty (No significance found in Step 2)
if sum(mask_refined(:)) == 0
    warning('The ROI Mask is empty (all zeros). CBPT found no significant clusters.');
    warning('Switching to MANUAL ROI (Mu band, Motor Chans, 0.5-1.5s) for extraction.');
    
    % --- FALLBACK: Create a manual mask based on your hypothesis ---
    manual_mask = zeros(size(mask_refined));
    t_man_idx = find(times >= 0.5 & times <= 1.5); % 500ms to 1500ms
    manual_mask(mu_idx, t_man_idx, motor_chans_idx) = 1;
    
    use_mask = manual_mask;
    mask_name = 'Manual_Fallback';
else
    use_mask = mask_refined;
    mask_name = 'CBPT_Refined';
end

fprintf('Using Mask: %s with %d active bins.\n', mask_name, sum(use_mask(:)));

%% 4. LOOP THROUGH SUBJECTS
for sub_idx = 1:length(all_subs)
    subID = all_subs(sub_idx);
    
    if ismember(subID, subjects_haptic), grp = 'Haptic'; else, grp = 'NonHaptic'; end
    
    % --- A. POWER ANALYSIS (Magnitude) ---
    dat_pre  = squeeze(ERD_Pre(sub_idx,:,:,:));
    dat_post = squeeze(ERD_Post(sub_idx,:,:,:));
    
    % Mean ERD inside the Mask
    % Use 'omitnan' to handle any edge artifacts from wavelets
    val_pre  = mean(dat_pre(use_mask==1), 'omitnan'); 
    val_post = mean(dat_post(use_mask==1), 'omitnan');
    
    % --- B. TEMPORAL DYNAMICS (Onset/Duration) ---
    % Average over freq (Mu) and Channels first to get a 1D Time Course
    mu_tc_pre  = squeeze(mean(mean(dat_pre(mu_idx, :, motor_chans_idx), 1, 'omitnan'), 3, 'omitnan'));
    mu_tc_post = squeeze(mean(mean(dat_post(mu_idx, :, motor_chans_idx), 1, 'omitnan'), 3, 'omitnan'));
    
    % Threshold: Mean Baseline - 2*SD
    base_idx   = find(times < 0);
    search_idx = find(times >= 0 & times <= 2.0); 
    
    % Protect against baseline NaNs
    if all(isnan(mu_tc_pre(base_idx)))
        thresh = NaN;
    else
        thresh = mean(mu_tc_pre(base_idx), 'omitnan') - (2 * std(mu_tc_pre(base_idx), 'omitnan'));
    end
    
    [on_pre, dur_pre, slp_pre]     = calc_temporal(mu_tc_pre, times, thresh, search_idx);
    [on_post, dur_post, slp_post]  = calc_temporal(mu_tc_post, times, thresh, search_idx);
    
    % --- C. POPULATE TABLES ---
    results.SubID(sub_idx)      = subID; 
    results.Group{sub_idx}      = grp;
    results.Pre_ERD(sub_idx)    = val_pre; 
    results.Post_ERD(sub_idx)   = val_post;
    results.Change_ERD(sub_idx) = val_post - val_pre;
    
    temporal.SubID(sub_idx)      = subID; 
    temporal.Group{sub_idx}      = grp;
    temporal.Pre_Onset(sub_idx)  = on_pre; 
    temporal.Post_Onset(sub_idx) = on_post;
    temporal.Pre_Dur(sub_idx)    = dur_pre; 
    temporal.Post_Dur(sub_idx)   = dur_post;
    temporal.Pre_Slope(sub_idx)  = slp_pre;
end

%% 5. EXPORT
writetable(results, './results/Main_Analysis_Data.csv');
writetable(temporal, './results/Temporal_Analysis_Data.csv');
fprintf('Step 3 Complete. CSV files exported to ./results/\n');

% --- LOCAL FUNCTION ---
function [onset, dur, slope] = calc_temporal(sig, t, thresh, idxs)
    if isnan(thresh), onset=NaN; dur=NaN; slope=NaN; return; end

    rel_sig = sig(idxs); 
    rel_t   = t(idxs);
    
    is_active = rel_sig < thresh; 
    onset_idx = find(is_active, 1, 'first');
    
    if isempty(onset_idx)
        onset = NaN; dur = 0; slope = 0;
    else
        onset = rel_t(onset_idx);
        off_idx = find(~is_active(onset_idx:end), 1, 'first');
        if isempty(off_idx)
            dur = rel_t(end) - onset;
        else
            dur = rel_t(onset_idx + off_idx - 1) - onset; 
        end
        
        [min_v, min_i] = min(rel_sig);
        time_to_peak = rel_t(min_i) - onset;
        if time_to_peak > 0
            slope = (min_v - thresh) / time_to_peak;
        else
            slope = 0; 
        end
    end
end