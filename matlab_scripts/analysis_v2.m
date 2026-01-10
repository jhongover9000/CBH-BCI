%% MASTER ANALYSIS SCRIPT: Haptic vs Non-Haptic MI Training
% Covers: CSD, Time-Freq, CBPT (Cluster Correction), and ANCOVA Export
%
% Modified to accept 'preprocess.m' outputs: 
% Files: [ID]_MI_pre_ICA.set, [ID]_Rest_pre_ICA.set, etc.

clear; clc; close all;

%% 1. CONFIGURATION
% =========================================================================
base_path = './epoched/';
save_path = './results/';
if ~exist(save_path, 'dir'), mkdir(save_path); end

% Groups
subjects_haptic = [1:20];       % IDs of Haptic Group
subjects_nonhaptic = [21:40];   % IDs of Non-Haptic Group
excluded_subs = [4, 12];        % Example excluded subjects

% Concatenate for looping
all_subjects = [subjects_haptic, subjects_nonhaptic];
all_subjects = setdiff(all_subjects, excluded_subs);

% Time-Frequency Parameters
freqs = 3:60;              % 3 to 60 Hz
cycles = [3 10];           % Wavelet cycles (3 at low freq, 10 at high)
times_out = -1000:20:2500; % Output time points (ms)
baseline = NaN;            % No baseline correction at this stage (we compare conditions)

% Initialize Waitbar
h = waitbar(0, 'Initializing Analysis...');

%% 2. DATA PROCESSING LOOP (CSD + TFR)
% =========================================================================
disp('Starting Data Processing Loop...');

% Pre-allocate storage
% Structure: tf_data.Condition(Subject, Freq, Time, Channel)
tf_data = []; 
group_labels = {};

for s = 1:length(all_subjects)
    subID = all_subjects(s);
    waitbar(s/length(all_subjects), h, sprintf('Processing Subject %d...', subID));
    fprintf('\n--- Processing Subject %d ---\n', subID);
    
    % --- GENERATE FILENAME (MATCHING PREPROCESS.M) ---
    % Preprocess.m uses 4 digits (CBH0001)
    if subID < 10
        sub_str = ['CBH000' num2str(subID)];
    else
        sub_str = ['CBH00' num2str(subID)];
    end
    
    % Define file paths for the 4 conditions
    files = struct();
    files.MI_Pre    = [sub_str '_MI_pre_ICA.set'];
    files.Rest_Pre  = [sub_str '_Rest_pre_ICA.set'];
    files.MI_Post   = [sub_str '_MI_post_ICA.set'];
    files.Rest_Post = [sub_str '_Rest_post_ICA.set'];
    
    % Define Condition Labels for the loop
    cond_names = fieldnames(files);
    
    for c = 1:length(cond_names)
        cond = cond_names{c};
        fname = files.(cond);
        
        % 1. LOAD DATA
        if ~exist(fullfile(base_path, fname), 'file')
            fprintf('  [!] File not found: %s. Skipping condition.\n', fname);
            continue; 
        end
        EEG = pop_loadset('filename', fname, 'filepath', base_path);
        
        % 2. APPLY CSD (Surface Laplacian)
        % Using the 'pop_current' plugin as discussed
        if exist('pop_current', 'file')
            EEG = pop_current(EEG, 'transform', 'surface_laplacian', ...
                                   'm', 4, 'lambda', 1e-5, 'head_radius', 10);
        else
            warning('CSD Plugin (pop_current) not found! Skipping CSD.');
        end
        
        % 3. TIME-FREQUENCY DECOMPOSITION
        % We iterate through channels to compute power
        
        % On first run, initialize the matrix sizes based on real data
        if isempty(tf_data)
            nFreq = length(freqs); 
            nTime = length(times_out); 
            nChan = EEG.nbchan;
            nSub  = length(all_subjects);
            
            tf_data.MI_Pre    = zeros(nSub, nFreq, nTime, nChan);
            tf_data.Rest_Pre  = zeros(nSub, nFreq, nTime, nChan);
            tf_data.MI_Post   = zeros(nSub, nFreq, nTime, nChan);
            tf_data.Rest_Post = zeros(nSub, nFreq, nTime, nChan);
        end
        
        % Loop through channels (using 'newtimef' purely for calculation, no plotting)
        % Note: Using parfor here would speed it up if you have the toolbox
        for ch = 1:EEG.nbchan
            
            % Check if channel is flat/empty (artifact of interpolation/removal?)
            if all(EEG.data(ch,:) == 0)
                continue; 
            end
            
            % Run Wavelet Transform
            % Output: [ersp, itc, powbase, times, freqs, erspboot, itcboot]
            [ersp, ~, ~, ~, ~] = newtimef(EEG.data(ch, :, :), ...
                EEG.pnts, [EEG.xmin EEG.xmax]*1000, EEG.srate, ...
                cycles, ...
                'freqs', [freqs(1) freqs(end)], ...
                'nfreqs', length(freqs), ...
                'timesout', times_out, ...
                'baseline', baseline, ... % No baseline subtraction here
                'plotersp', 'off', 'plotitc', 'off', 'verbose', 'off');
                
            % Store Data (ERSP is in dB usually, or Log Power)
            % Ensure dimensions match [Freq x Time]
            tf_data.(cond)(s, :, :, ch) = ersp; 
        end
    end
    
    % Store Group Label
    if ismember(subID, subjects_haptic)
        group_labels{s} = 'Haptic';
    else
        group_labels{s} = 'NonHaptic';
    end
end
close(h);

% Save intermediate processing
save([save_path 'TF_Data_Processed.mat'], 'tf_data', 'group_labels', 'times_out', 'freqs', '-v7.3');
disp('Time-Frequency Processing Complete. Data Saved.');


