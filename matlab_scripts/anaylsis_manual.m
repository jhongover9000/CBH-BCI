%% MASTER ANALYSIS: Manual ROI Approach
%  - Skips Cluster Permutation Testing (CBPT).
%  - Defines ROI based on Literature (Mu Band, Motor Cortex, 0-2s).
%  - Extracts Mean ERD for ANCOVA.
%  - Extracts Temporal Dynamics (Slope, Onset).

clear; clc; close all;

% 1. CONFIGURATION
% =========================================================================
data_path = './data_intermediate/TF_Data_CSD.mat'; % Ensure this matches your file
save_path = './results/';
if ~exist(save_path, 'dir'), mkdir(save_path); end

% A. ROI Definition (Literature Based)
% -------------------------------------------------------------------------
roi_freqs  = [8 13];    % Mu Band (Standard for Motor Imagery)
% roi_freqs = [8 30];   % Alternative: Mu + Beta (Broad motor)

roi_time   = [0 2.0];   % 0s (Cue) to 2.0s (End of strong imagery)
                        % Averaging this window gives the "Magnitude"

% B. Channel Indices (Motor Area)
% -------------------------------------------------------------------------
% CRITICAL: Verify these match your cap! 
% Common 64-chan indices: C3=13, C4=49 (approx). 
% Using the indices you listed previously:
roi_chans = [7, 35, 36, 37, 39]; 

% C. Analysis Settings
% -------------------------------------------------------------------------
use_rest_subtraction = false; % Set TRUE to use (MI - Rest), FALSE for (MI - Baseline)
                              % Recommended: FALSE (if Rest is noisy/contaminated)

% 2. LOAD DATA
% =========================================================================
fprintf('Loading Data (%s)...\n', data_path);
load(data_path); % Expected: tf_Pre_MI, tf_Post_MI, (tf_Pre_Rest...), frex, times, all_subs

% 3. PREPARE ROI INDICES
% =========================================================================
% Find the indices in the data matrices corresponding to our real-world units

f_idx = find(frex >= roi_freqs(1) & frex <= roi_freqs(2));
t_idx = find(times >= roi_time(1) & times <= roi_time(2));
c_idx = roi_chans;

fprintf('\n--- ROI DEFINITION ---\n');
fprintf('  Frequency: %.1f - %.1f Hz (%d bins)\n', frex(f_idx(1)), frex(f_idx(end)), length(f_idx));
fprintf('  Time:      %.2f - %.2f s  (%d points)\n', times(t_idx(1)), times(t_idx(end)), length(t_idx));
fprintf('  Channels:  %d channels selected\n', length(c_idx));
fprintf('----------------------\n');

% 4. EXTRACTION LOOP
% =========================================================================
% Initialize Output Tables
tbl_mag  = table(); % For ANCOVA (Magnitude)
tbl_temp = table(); % For Temporal Dynamics

fprintf('Extracting Data for %d subjects...\n', length(all_subs));

for s = 1:length(all_subs)
    subID = all_subs(s);
    
    % A. Define Group
    if mod(subID, 2) == 1
        grp = 'Haptic';
    else
        grp = 'NonHaptic';
    end
    
    % B. Get Subject Data (Squeeze to [Freq x Time x Chan])
    % ---------------------------------------------------------------------
    pre_data  = squeeze(tf_Pre_MI(s, :, :, :));
    post_data = squeeze(tf_Post_MI(s, :, :, :));
    
    % Optional: Subtract Rest if requested
    if use_rest_subtraction
        pre_data  = pre_data  - squeeze(tf_Pre_Rest(s, :, :, :));
        post_data = post_data - squeeze(tf_Post_Rest(s, :, :, :));
    end
    
    % C. METRIC 1: MEAN MAGNITUDE (For ANCOVA)
    % ---------------------------------------------------------------------
    % Average over Freq, Time, and Channels inside the ROI box
    % Use 'omitnan' to handle any edge artifacts
    
    val_pre  = mean(mean(mean(pre_data(f_idx, t_idx, c_idx), 'omitnan'), 'omitnan'), 'omitnan');
    val_post = mean(mean(mean(post_data(f_idx, t_idx, c_idx), 'omitnan'), 'omitnan'), 'omitnan');
    
    % Store Magnitude
    tbl_mag.SubID(s)    = subID;
    tbl_mag.Group{s}    = grp;
    tbl_mag.Pre_ERD(s)  = val_pre;
    tbl_mag.Post_ERD(s) = val_post;
    tbl_mag.Change(s)   = val_post - val_pre;
    
    % D. METRIC 2: TEMPORAL DYNAMICS (Onset/Slope)
    % ---------------------------------------------------------------------
    % To find onset, we need a 1D Time Course (Average Freq & Chans first)
    % We use the FULL time vector here to find when it starts, not just the ROI window.
    
    tc_pre  = squeeze(mean(mean(pre_data(f_idx, :, c_idx), 1, 'omitnan'), 3, 'omitnan'));
    tc_post = squeeze(mean(mean(post_data(f_idx, :, c_idx), 1, 'omitnan'), 3, 'omitnan'));
    
    % Calculate Metrics (using local function below)
    % Threshold: Baseline Mean - 2*SD
    % Search Window: 0 to 3s
    [on_pre, dur_pre, slp_pre] = calc_temporal(tc_pre, times, [-0.5 0], [0 3]);
    [on_post, dur_post, slp_post] = calc_temporal(tc_post, times, [-0.5 0], [0 3]);
    
    % Store Temporal
    tbl_temp.SubID(s)       = subID;
    tbl_temp.Group{s}       = grp;
    tbl_temp.Pre_Onset(s)   = on_pre;
    tbl_temp.Post_Onset(s)  = on_post;
    tbl_temp.Pre_Slope(s)   = slp_pre;
    tbl_temp.Post_Slope(s)  = slp_post;
end

% 5. STATISTICS & EXPORT
% =========================================================================

% Remove empty rows if any
tbl_mag(tbl_mag.SubID == 0, :) = [];
tbl_temp(tbl_temp.SubID == 0, :) = [];

% Save to CSV
writetable(tbl_mag, fullfile(save_path, 'ROI_Magnitude_ANCOVA.csv'));
writetable(tbl_temp, fullfile(save_path, 'ROI_Temporal_Metrics.csv'));

fprintf('\nAnalysis Complete.\n');
fprintf('  Magnitude Data saved to: %s\n', fullfile(save_path, 'ROI_Magnitude_ANCOVA.csv'));
fprintf('  Temporal Data saved to:  %s\n', fullfile(save_path, 'ROI_Temporal_Metrics.csv'));

% --- QUICK LOOK: ANCOVA RESULTS ---
fprintf('\n--- QUICK ANCOVA PREVIEW (ERD Magnitude) ---\n');
% Model: Post_ERD ~ Pre_ERD + Group
% This controls for baseline differences (Pre_ERD)
mdl = fitlm(tbl_mag, 'Post_ERD ~ Pre_ERD + Group');
disp(mdl);

fprintf('Interpreting the "Group_Haptic" p-value:\n');
fprintf('  p < 0.05 means Haptic group is significantly different from NonHaptic\n');
fprintf('  after controlling for their starting point (Pre).\n');


%% --- LOCAL FUNCTIONS ---

function [onset, dur, slope] = calc_temporal(sig, t_vec, base_win, search_win)
    % 1. Define Threshold (Mean Baseline - 2*SD)
    base_idx = find(t_vec >= base_win(1) & t_vec <= base_win(2));
    if isempty(base_idx) || all(isnan(sig(base_idx)))
        onset=NaN; dur=NaN; slope=NaN; return; 
    end
    
    mu_base = mean(sig(base_idx), 'omitnan');
    sd_base = std(sig(base_idx), 'omitnan');
    thresh  = mu_base - (2 * sd_base); % ERD is negative, so we look below this
    
    % 2. Search for Onset
    search_idx = find(t_vec >= search_win(1) & t_vec <= search_win(2));
    rel_sig    = sig(search_idx);
    rel_t      = t_vec(search_idx);
    
    % Find first point that drops below threshold
    is_active = rel_sig < thresh;
    first_idx = find(is_active, 1, 'first');
    
    if isempty(first_idx)
        onset = NaN; dur = 0; slope = 0;
    else
        onset = rel_t(first_idx);
        
        % Duration (how long does it stay down?)
        % Find next point where it goes back UP above threshold
        off_idx = find(~is_active(first_idx:end), 1, 'first');
        if isempty(off_idx)
            dur = rel_t(end) - onset; % Stays down until end
        else
            dur = rel_t(first_idx + off_idx - 1) - onset;
        end
        
        % Slope (Time to Peak)
        % Find the minimum value (peak ERD)
        [min_val, min_i] = min(rel_sig);
        time_to_peak = rel_t(min_i) - onset;
        
        if time_to_peak > 0
            slope = (min_val - thresh) / time_to_peak; % dB per second
        else
            slope = 0; % Immediate peak
        end
    end
end