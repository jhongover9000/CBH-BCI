%% STEP 3: Statistical Extraction & Temporal Dynamics
%  - Loads processed Time-Freq data (from Step 1)
%  - Applies Odd/Even Group Logic
%  - Calculates ERD Contrast (MI - Rest)
%  - Computes Temporal Metrics (Onset, Duration, Slope)
%  - Exports to CSV

clear; clc;
load('./data_intermediate/TF_Data_CSD.mat'); % Loads tf_*, all_subs, frex, times
load('./data_intermediate/ROI_Mask.mat');    % Loads mask_refined, motor_chans_idx, mu_idx

%% 1. CONFIGURATION (Group Assignment)
% Re-define groups here to be safe, based on Odd/Even logic
subjects_haptic = all_subs(mod(all_subs, 2) == 1); % Odd IDs = Haptic
subjects_non    = all_subs(mod(all_subs, 2) == 0); % Even IDs = Non-Haptic

fprintf('Stats Configuration:\n');
fprintf('  Haptic Group (Odd): N=%d\n', length(subjects_haptic));
fprintf('  Non-Haptic Group (Even): N=%d\n', length(subjects_non));

%% 2. CALCULATE ERD MAPS (MI - Rest)
% tf_Pre_MI is already in dB (Log ratio to baseline).
% Subtracting them gives the contrast in dB.
ERD_Pre  = tf_Pre_MI - tf_Pre_Rest;   
ERD_Post = tf_Post_MI - tf_Post_Rest;

results = table();
temporal = table();

%% 3. LOOP THROUGH SUBJECTS
for sub_idx = 1:length(all_subs)
    subID = all_subs(sub_idx);
    
    % Determine Group Label
    if ismember(subID, subjects_haptic)
        grp = 'Haptic'; 
    else
        grp = 'NonHaptic'; 
    end
    
    % --- A. POWER ANALYSIS (Magnitude) ---
    dat_pre  = squeeze(ERD_Pre(sub_idx,:,:,:));
    dat_post = squeeze(ERD_Post(sub_idx,:,:,:));
    
    % Mean ERD inside the Refined Mask
    % (Average over Freq, Time, and Channels where Mask == 1)
    val_pre  = mean(dat_pre(mask_refined==1)); 
    val_post = mean(dat_post(mask_refined==1));
    
    % --- B. TEMPORAL DYNAMICS (Onset/Duration) ---
    % 1. Isolate Motor Channels & Mu Band
    % Average over freq (Mu) and Channels first to get a 1D Time Course
    mu_tc_pre  = squeeze(mean(mean(dat_pre(mu_idx, :, motor_chans_idx), 1), 3));
    mu_tc_post = squeeze(mean(mean(dat_post(mu_idx, :, motor_chans_idx), 1), 3));
    
    % 2. Define Threshold (Mean Baseline - 2*SD)
    base_idx   = find(times < 0);
    search_idx = find(times >= 0 & times <= 2.0); % Look for onset in first 2s
    
    thresh = mean(mu_tc_pre(base_idx)) - (2 * std(mu_tc_pre(base_idx)));
    
    % 3. Calculate Metrics
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

%% 4. EXPORT
writetable(results, './results/Main_Analysis_Data.csv');
writetable(temporal, './results/Temporal_Analysis_Data.csv');
fprintf('Step 3 Complete. CSV files exported to ./results/\n');

% --- LOCAL FUNCTION: TEMPORAL METRICS ---
function [onset, dur, slope] = calc_temporal(sig, t, thresh, idxs)
    rel_sig = sig(idxs); 
    rel_t   = t(idxs);
    
    % Active = Signal drops BELOW threshold (more negative)
    is_active = rel_sig < thresh; 
    onset_idx = find(is_active, 1, 'first');
    
    if isempty(onset_idx)
        onset = NaN; dur = 0; slope = 0;
    else
        onset = rel_t(onset_idx);
        
        % Duration: Time until it rises back above threshold
        off_idx = find(~is_active(onset_idx:end), 1, 'first');
        if isempty(off_idx)
            % Stays active until end of window
            dur = rel_t(end) - onset;
        else
            dur = rel_t(onset_idx + off_idx - 1) - onset; 
        end
        
        % Slope: (Peak_Val - Thresh) / Time_to_Peak
        [min_v, min_i] = min(rel_sig);
        time_to_peak = rel_t(min_i) - onset;
        
        if time_to_peak > 0
            slope = (min_v - thresh) / time_to_peak;
        else
            slope = 0; % Immediate peak
        end
    end
end