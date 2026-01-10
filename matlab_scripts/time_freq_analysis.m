%% STEP 1: CSD and Time-Frequency Processing
%  - Loads Epoched Data
%  - Applies CSD (using your specific working method)
%  - Computes Induced Power using FFT-Wavelets
%  - Saves data for Stats

clear; clc; close all;

%% CONFIGURATION (Corrected for Alternating Groups)
base_path = './epoched/';
save_dir  = './data_intermediate/';
if ~exist(save_dir, 'dir'), mkdir(save_dir); end

% 1. Define All Potential Subjects (e.g., 1 to 44)
all_potential_subs = 1:44;

% 2. Assign Groups based on Odd/Even
% Assumption: Odds = Haptic, Evens = Non-Haptic (SWAP IF NEEDED)
subjects_haptic = all_potential_subs(mod(all_potential_subs, 2) == 1); % [1, 3, 5...]
subjects_non    = all_potential_subs(mod(all_potential_subs, 2) == 0); % [2, 4, 6...]

% 3. Remove Excluded Subjects
excluded_subs   = []; 

subjects_haptic = setdiff(subjects_haptic, excluded_subs);
subjects_non    = setdiff(subjects_non, excluded_subs);
all_subs        = sort([subjects_haptic, subjects_non]); % Combined valid list

% -------------------------------------------------------------------------
% Double Check Output in Command Window
fprintf('Haptic Group (N=%d): %s\n', length(subjects_haptic), num2str(subjects_haptic));
fprintf('Non-Haptic Group (N=%d): %s\n', length(subjects_non), num2str(subjects_non));
% -------------------------------------------------------------------------

% Wavelet & Freq Parameters
SR = 250; 
min_freq = 2; max_freq = 80;
num_frex = max_freq - min_freq;
frex = logspace(log10(min_freq), log10(max_freq), num_frex);

range_cycles = [3 15]; 
% RENAMED 's' to 'wav_s' to avoid conflict with subject loop
wav_s = logspace(log10(range_cycles(1)), log10(range_cycles(2)), num_frex) ./ (2*pi*frex);
wavtime = -2:1/SR:2;
half_wave = (length(wavtime)-1)/2;

% Time vector (Assuming -5 to 5 epoch)
epoch_period = [-5 5];
times = linspace(epoch_period(1), epoch_period(2), diff(epoch_period) * SR);
baseline_window = [-0.5 0]; 
baseidx = dsearchn(times', baseline_window');

% Initialize Storage
tf_Pre_MI   = []; tf_Pre_Rest = [];
tf_Post_MI  = []; tf_Post_Rest= [];

%% PROCESSING LOOP
% Changed loop variable from 's' to 'sub_idx'
for sub_idx = 1:length(all_subs)
    subID = all_subs(sub_idx);
    fprintf('\nProcessing Subject %d (CBH%04d)...\n', subID, subID);
    fname_base = sprintf('CBH%04d', subID); 
    
    % Loop Conditions
    conditions = {'MI', 'Rest'};
    phases     = {'pre', 'post'};
    
    for ph = 1:2
        for c = 1:2
            cond = conditions{c};
            phase = phases{ph};
            
            f_name = sprintf('%s_%s_%s_ICA.set', fname_base, cond, phase);
            if ~exist(fullfile(base_path, f_name), 'file')
                fprintf('  Skipping %s (Not found)\n', f_name); continue;
            end
            
            % 1. Load Data
            EEG = pop_loadset(f_name, base_path);
            [nCh, nPnts, nTrials] = size(EEG.data);
            
            % ----------------- CSD BLOCK (YOUR WORKING CODE) -----------------
            ConvertLocations('locs.ced', 'locs.csd'); 
            E = textread('E60.asc','%s'); 
            M = ExtractMontage('locs.csd', E);
            [G,H] = GetGH(M);
            
            data_csd = single(repmat(NaN, size(EEG.data))); 
            for ne = 1:nTrials
                myEEG = single(EEG.data(:,:,ne));
                MyResults = CSD(myEEG, G, H, 1.0e-5, 10);
                data_csd(:,:,ne) = MyResults;
            end
            EEG.data = double(data_csd);
            % -----------------------------------------------------------------

            % 2. WAVELET CONVOLUTION
            nWave = length(wavtime);
            nData = nPnts * nTrials;
            nConv = nWave + nData - 1;
            
            temp_pow = zeros(num_frex, nPnts, nCh, 'single');
            
            for ch = 1:nCh
                alldata = reshape(EEG.data(ch, :, :), 1, []);
                dataX   = fft(alldata, nConv);
                
                for fi = 1:num_frex
                    % Fixed: using 'wav_s(fi)' instead of 's(fi)'
                    wavelet  = exp(2*1i*pi*frex(fi).*wavtime) .* exp(-wavtime.^2./(2*wav_s(fi)^2));
                    waveletX = fft(wavelet, nConv);
                    waveletX = waveletX ./ max(waveletX);
                    
                    as = ifft(waveletX .* dataX);
                    as = as(half_wave+1:end-half_wave);
                    as = reshape(as, nPnts, nTrials);
                    
                    % Induced Power (Mean of Squares)
                    temp_pow(fi, :, ch) = mean(abs(as).^2, 2);
                end
            end
            
            % 3. DB CONVERSION
            temp_db = zeros(size(temp_pow), 'single');
            for ch = 1:nCh
                base_pow = mean(temp_pow(:, baseidx(1):baseidx(2), ch), 2);
                temp_db(:,:,ch) = 10 * log10(bsxfun(@rdivide, temp_pow(:,:,ch), base_pow));
            end
            
            % 4. STORE
            if sub_idx==1 
                dims = [length(all_subs), num_frex, nPnts, nCh];
                if isempty(tf_Pre_MI), tf_Pre_MI = zeros(dims,'single'); end
                if isempty(tf_Pre_Rest), tf_Pre_Rest = zeros(dims,'single'); end
                if isempty(tf_Post_MI), tf_Post_MI = zeros(dims,'single'); end
                if isempty(tf_Post_Rest), tf_Post_Rest = zeros(dims,'single'); end
            end
            
            if strcmp(cond, 'MI') && strcmp(phase, 'pre')
                tf_Pre_MI(sub_idx,:,:,:) = temp_db;
            elseif strcmp(cond, 'Rest') && strcmp(phase, 'pre')
                tf_Pre_Rest(sub_idx,:,:,:) = temp_db;
            elseif strcmp(cond, 'MI') && strcmp(phase, 'post')
                tf_Post_MI(sub_idx,:,:,:) = temp_db;
            elseif strcmp(cond, 'Rest') && strcmp(phase, 'post')
                tf_Post_Rest(sub_idx,:,:,:) = temp_db;
            end
            
        end 
    end 
end 

save(fullfile(save_dir, 'TF_Data_CSD.mat'), 'tf_*', 'all_subs', 'subjects_haptic', 'frex', 'times');
disp('Step 1 Complete.');