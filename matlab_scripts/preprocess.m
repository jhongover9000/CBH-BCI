%%%% EEG Preprocessing Pipeline
%%%% Adapted for MI, Rest, and Tapping conditions

% clear all;
% close all;
% clc;

%% Set paths
path_to_rawdata = './data/CBH/';
path_to_epoched = './epoched/';

% Create output directory if it doesn't exist
if ~exist(path_to_epoched, 'dir')
    mkdir(path_to_epoched);
end

%% Define parameters
nSubject = 44;  % Number of subjects
subjectNum = 0;
filename = "";
coinType = "gold";  % Filter for specific coin events
toggle_skip = true; % Filter out self-reported unsuccessful MI/Rest trials


% Epoch parameters
epoch_period = [-5 5];          % -3 to 4 seconds

% Load channel locations
load ./reference/NewEasyCap63.mat  % Assuming this file exists

%% Initialize summary storage
% Create a cell array to store the counts of skipped trials for each subject.
load skipped_counts_summary.mat
skipped_counts_summary = cell(nSubject + 1, 5);
skipped_counts_summary(1,:) = {'SubjectID', 'Skipped_MI_Pre', 'Skipped_MI_Post', 'Skipped_Rest_Pre', 'Skipped_Rest_Post'};


%% Main preprocessing loop
for sub = 1:nSubject
    
    % Initialize counters for the current subject
    skipped_mi_count_pre = 0;
    skipped_mi_count_post = 0;
    skipped_rest_count_pre = 0;
    skipped_rest_count_post = 0;

    fprintf('Processing Subject %d/%d\n', sub, nSubject);
    
    % Load raw data
    if (sub < 10)
        filename = ['CBH000' int2str(sub)];
    else
        filename = ['CBH00' int2str(sub)];
    end
    set_file = [path_to_rawdata '' filename '.vhdr'];
    
    % Set condition markers based on subject number
    if mod(sub,2) == 1
        markerStartLetter = 'S';
    else
        markerStartLetter = 'R';
    end
    
    % Redefine events with the correct marker letter for the subject
    events = {
              [markerStartLetter '  3'], 'MI';
              [markerStartLetter '  4'], 'Rest';
              [markerStartLetter '  9'], 'TapStart';
              [markerStartLetter ' 11'], 'TapEnd'
              };

    disp(set_file)

    if ~exist(set_file, 'file')
        fprintf('Warning: File not found for subject %d\n', sub);
        continue;
    end
    
    fprintf('Reading %s...\n', set_file);
    
    % Load EEG data
    EEG = pop_fileio(set_file);
    
    % Set channel locations
    EEG.chanlocs = struct(chanlocsEasyCapNoRef);
    EEG = eeg_checkset(EEG);
    
    % Add reference channel FCz as channel 64
    EEG = pop_chanedit(EEG, 'append', 63, ...
        'changefield', {64 'labels' 'FCz'}, ...
        'changefield', {64 'X' '0.383'}, ...
        'changefield', {64 'Y' '0'}, ...
        'changefield', {64 'Z' '0.923'}, ...
        'convert', {'cart2all'});
    EEG = eeg_checkset(EEG);
    
    %% Channel removal
    % Remove bad channels (FT9, FT10, TP9, TP10)
    ex_channels = [5 10 21 27];
    EEG = pop_select(EEG, 'nochannel', ex_channels);
    nCh = EEG.nbchan;
    
    %% Filtering
    fprintf('Applying filters...\n');
    
    % Bandpass filter [0.1-80 Hz]
    EEG = pop_eegfiltnew(EEG, [], 80);   % LP
    EEG = pop_eegfiltnew(EEG, 1,  []);   % HP
    EEG = eeg_checkset(EEG);
    
    % Notch filter at 50Hz
    EEG = pop_eegfiltnew(EEG, 49.5, 50.5, 8250, 1, [], 1);
    EEG = eeg_checkset(EEG);
    close all;
    
    %% Artifact Subspace Reconstruction (ASR)
    fprintf('Running ASR...\n');
    
    % Store channel info before ASR
    infoCh = {EEG.chanlocs.labels};
    saveEEG = struct(EEG);
    
    % Apply ASR
    EEG = clean_artifacts(EEG, ...
      'FlatlineCriterion',   10, ...
      'Highpass',            'off', ...
      'ChannelCriterion',    0.8, ...
      'LineNoiseCriterion',  'off', ...
      'BurstCriterion',      30, ...   
      'WindowCriterion',     'off', ... 
      'BurstRejection',      'off', ... 
      'Distance',            'Euclidian', ...
      'WindowCriterionTolerances', [-Inf 7], ...
      'fusechanrej',         1);
    
    nChMiss = EEG.nbchan;
    [M, N] = size(EEG.data);
    
    %% Find and interpolate missing channels
    % Expand data matrix to original size
    EEG.data = [EEG.data(:,:); zeros(nCh-nChMiss, N)];
    
    missCh = {};
    missChNum = [];
    
    for i = 1:nCh
        temp = max(strcmp(infoCh(i), {EEG.chanlocs.labels}));
        if temp == 0
            missCh = [missCh infoCh(i)];
            missChNum = [missChNum i];
            EEG.data(i+1:end,:) = EEG.data(i:end-1,:);
            EEG.data(i,:) = zeros(1, N);
        end
    end
    
    % Restore original channel structure
    EEG.nbchan = saveEEG.nbchan;
    EEG.chanlocs = struct(saveEEG.chanlocs);
    EEG = eeg_checkset(EEG);
    
    % Interpolate missing channels
    if ~isempty(missChNum)
        fprintf('Interpolating %d channels...\n', length(missChNum));
        EEG = pop_interp(EEG, missChNum, 'spherical');
        EEG = eeg_checkset(EEG);
    end
    
    %% Common Average Reference (CAR)
    fprintf('Applying CAR...\n');
    
    % Re-reference to average with FCz retained
    EEG = pop_reref(EEG, [], 'refloc', struct(...
        'theta', {0}, ...
        'radius', {0.1252}, ...
        'labels', {'FCz'}, ...
        'sph_theta', {0}, ...
        'sph_phi', {67.4639}, ...
        'X', {0.383}, ...
        'Y', {0}, ...
        'Z', {0.923}, ...
        'sph_radius', {0.99931}, ...
        'type', {''}, ...
        'ref', {''}, ...
        'urchan', {[]}, ...
        'datachan', {0}));
    EEG = eeg_checkset(EEG);
    
    %% Find boundary marker before cleaning events
    % This is needed to correctly assign skipped trials to pre/post bins for MI/Rest.
    tap_start_boundary_marker = [markerStartLetter '  9'];
    all_event_types_for_boundary = {EEG.event.type};
    boundary_event_index = find(strcmp(all_event_types_for_boundary, tap_start_boundary_marker), 1, 'first');

    if isempty(boundary_event_index)
        fprintf('Warning: No TapStart marker ("%s") found for subject %s. Cannot separate pre/post NO counts or MI/Rest epochs.\n', tap_start_boundary_marker, filename);
    end

    %% Clean events and count "NO" evaluations
    % Remove empty markers, fix broken ones
    del = 0;
    for j = 1:size(EEG.event, 2)
        if strcmp(EEG.event(j-del).type, 'empty')
            EEG.event(j-del) = [];
            del = del + 1;
        end
    end
    
    for j = 1:size(EEG.event, 2)
        % Issue in initial Unity code sent S3 instead of S7 for gold coins
        if strcmp(EEG.event(j).type, [markerStartLetter '  3'])
            if(j > 1 && strcmp(EEG.event(j - 1).type, [markerStartLetter '  2']) == 0)
                EEG.event(j).type = [markerStartLetter '  7']; % Re-label as Gold MI
            end
        end
        % If tapzone enter, check cointype to skip
        if (coinType ~= "") && (strcmp(EEG.event(j).type, [markerStartLetter '  9']))
            if(coinType == "gold")
                coinMarker = [markerStartLetter '  7'];
            elseif (coinType == "wooden")
                coinMarker = [markerStartLetter '  8'];
            end

            if(j > 1 && strcmp(EEG.event(j - 1).type, coinMarker))
                EEG.event(j).type = 'skip';
            end
        end

        % If EVAL NO, then toggle skip for Rest/MI marker before
        if (strcmp(EEG.event(j).type, [markerStartLetter ' 15']) && sub ~= 14 && sub ~= 33)
            if(toggle_skip)
                if (j > 2) % Ensure we don't go out of bounds
                    is_pre_trial = ~isempty(boundary_event_index) && (j - 2 < boundary_event_index);

                    if (strcmp(EEG.event(j - 2).type, [markerStartLetter '  3'])) % It's an MI trial
                        if is_pre_trial
                            skipped_mi_count_pre = skipped_mi_count_pre + 1;
                        else
                            skipped_mi_count_post = skipped_mi_count_post + 1;
                        end
                        EEG.event(j-2).type = 'skip';
                    elseif (strcmp(EEG.event(j - 2).type, [markerStartLetter '  4'])) % It's a Rest trial
                        if is_pre_trial
                            skipped_rest_count_pre = skipped_rest_count_pre + 1;
                        else
                            skipped_rest_count_post = skipped_rest_count_post + 1;
                        end
                        EEG.event(j-2).type = 'skip';
                    end
                end
            end
        end
    end
    
    % Store the final counts for the current subject in the summary array
    skipped_counts_summary{sub + 1, 1} = filename;
    skipped_counts_summary{sub + 1, 2} = skipped_mi_count_pre;
    skipped_counts_summary{sub + 1, 3} = skipped_mi_count_post;
    skipped_counts_summary{sub + 1, 4} = skipped_rest_count_pre;
    skipped_counts_summary{sub + 1, 5} = skipped_rest_count_post;
    
    fprintf('Subject %s -> Skipped MI (Pre/Post): %d/%d, Skipped Rest (Pre/Post): %d/%d\n', ...
        filename, skipped_mi_count_pre, skipped_mi_count_post, skipped_rest_count_pre, skipped_rest_count_post);

    % ===== ICA (continuous, before epoching) =====
    % Train ICA on a 1 Hz high-pass copy (recommended for stable unmixing),
    % then apply weights to the original 0.1–85 Hz data and reject artifact ICs.
    
    fprintf('Preparing ICA training copy (1 Hz high-pass)...\n');
    EEG_ica = pop_eegfiltnew(EEG, 1, []);     % 1 Hz HP *for training only*
    EEG_ica = eeg_checkset(EEG_ica);
    
    % --- Decide PCA dimensionality (rank) ---
    % Average reference reduces rank by ~1; channel interpolation also lowers rank.
    if exist('missChNum','var'), nInterp = numel(missChNum); else, nInterp = 0; end
    approxRank = max(1, EEG_ica.nbchan - 1 - nInterp);
    
    fprintf('Running ICA (pca=%d)...\n', approxRank);
    % If PICARD plugin is available, it’s faster; otherwise use runica (extended).
    if exist('picard', 'file')
        EEG_ica = pop_runica(EEG_ica, 'icatype','picard', 'pca', approxRank);
    else
        EEG_ica = pop_runica(EEG_ica, 'icatype','runica', ...
            'extended', 1, 'stop', 1e-7, 'maxsteps', 512, 'pca', approxRank);
    end
    EEG_ica = eeg_checkset(EEG_ica);
    
    % --- Label ICs (ICLabel) and pick artifact components ---
    badICs = [];
    if exist('pop_iclabel','file')
        EEG_ica = pop_iclabel(EEG_ica, 'default');
        cls = EEG_ica.etc.ic_classification.ICLabel.classifications;
        % ICLabel order: [Brain, Muscle, Eye, Heart, LineNoise, ChannelNoise, Other]
        % Conservative auto-reject for MI: remove strong Eye/Muscle/Line/ChanNoise.
        badICs = find(cls(:,2) >= 0.90 | ...  % Muscle
                      cls(:,3) >= 0.90 | ...  % Eye (blinks/saccades)
                      cls(:,5) >= 0.90 | ...  % Line Noise
                      cls(:,6) >= 0.90);      % Channel Noise
        fprintf('ICs flagged for removal: %s\n', mat2str(badICs));
    else
        fprintf('ICLabel not found. Proceeding without auto-labels.\n');
    end
    
    % --- Transfer ICA to the original bandpass data and remove bad ICs ---
    EEG.icaweights  = EEG_ica.icaweights;
    EEG.icasphere   = EEG_ica.icasphere;
    EEG.icawinv     = EEG_ica.icawinv;
    EEG.icachansind = EEG_ica.icachansind;
    if isfield(EEG_ica, 'etc') && isfield(EEG_ica.etc, 'ic_classification')
        EEG.etc.ic_classification = EEG_ica.etc.ic_classification; % keep labels
    end
    
    if ~isempty(badICs)
        fprintf('Removing %d artifact IC(s) and back-projecting clean data...\n', numel(badICs));
        EEG = pop_subcomp(EEG, badICs, 0);   % 0 = do not permanently remove IC matrices
    end
    EEG = eeg_checkset(EEG);

    %% Epoching for different conditions
    fprintf('Epoching data...\n');

    % Get all event types *after* cleaning and marking skips
    all_event_types = {EEG.event.type};

    for evt = 1:size(events, 1)
        event_marker = events{evt, 1};
        event_name = events{evt, 2};
        fprintf('Processing %s condition...\n', event_name);
        
        events_only_indices = find(strcmp(all_event_types, event_marker));

        % Loop to create three sets of epochs: pre, post, and all
        for s = 1:3
            selected = [];
            epoch_label = '';

            % Determine the epoch label (pre, post, all)
            if s == 1, epoch_label = 'pre'; elseif s == 2, epoch_label = 'post'; else, epoch_label = 'all'; end
            fprintf('Processing %s %s...\n', event_name, epoch_label);

            % --- Logic to define 'selected' indices based on event type ---
            if strcmp(event_name, 'MI') || strcmp(event_name, 'Rest')
                % Use boundary marker logic for MI/Rest
                if s == 1 % Pre
                    if ~isempty(boundary_event_index), selected = events_only_indices(events_only_indices < boundary_event_index); end
                elseif s == 2 % Post
                    if ~isempty(boundary_event_index), selected = events_only_indices(events_only_indices > boundary_event_index); end
                else % All
                    selected = events_only_indices;
                end
            else
                % Use count-from-end logic for other events (TapStart/End)
                if s == 1 % Pre (last 40 to last 21 trials)
                    if length(events_only_indices) >= 40
                        selected = events_only_indices(end-39:end-20);
                    end
                elseif s == 2 % Post (last 20 trials)
                    if length(events_only_indices) >= 20
                        selected = events_only_indices(end-19:end);
                    end
                else % All
                    selected = events_only_indices;
                end
            end
            
            epoch_file = sprintf('%s_%s_%s_ICA.set', filename, event_name, epoch_label);

            if isempty(selected)
                fprintf('  - No events found for this condition/%s. Skipping.\n', epoch_label);
                continue;
            end
            
            fprintf('  - Found %d events for %s.\n', length(selected), epoch_label);

            % Temporarily modify event types for selected events to create unique identifiers.
            temp_event_type = sprintf('TEMP_%s_%s', event_name, epoch_label);
            
            for i = 1:length(selected)
                EEG.event(selected(i)).type = temp_event_type;
            end
            
            % Epoch around the temporarily renamed events
            epoch = pop_epoch(EEG, {temp_event_type}, epoch_period, 'epochinfo', 'yes');
            
            % Restore original event types in the main EEG structure
            for i = 1:length(selected)
                EEG.event(selected(i)).type = event_marker;
            end
            
            epoch = eeg_checkset(epoch);
            
            % Save epoched data
            epoch = pop_saveset(epoch, ...
                'filename', epoch_file, ...
                'filepath', path_to_epoched, ...
                'savemode', 'onefile');
            
            fprintf('  - Saved: %s\n', fullfile(path_to_epoched, epoch_file));
       end
    end
    
    fprintf('Subject %d completed!\n\n', sub);
end

fprintf('Preprocessing completed for all subjects!\n\n');

% Display the summary of skipped counts in the command window
disp('Summary of Self-Reported "NO" Evaluations:');
disp(skipped_counts_summary);

% Optionally, save the results to a file for later analysis
save('skipped_counts_summary.mat', 'skipped_counts_summary');
writecell(skipped_counts_summary, 'skipped_counts_summary.csv');
fprintf('Skipped counts summary has been generated.\n');
