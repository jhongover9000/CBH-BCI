%% STEP 4: VR Simulation Analysis (Updated)
%  - Uses TapStart files (Simulation Data)
%  - Applies Odd/Even Group Logic
%  - Uses EXACT CSD & Wavelet code from Step 1
%  - Calculates "Induced Power" for Gold vs Wood trials

clear; clc;
base_path = './epoched/';
save_dir  = './data_intermediate/';

%% 1. CONFIGURATION
% A. Define Subjects (Odd=Haptic, Even=Non)
all_potential_subs = 1:44;
excluded_subs      = [4, 12]; % Add your exclusions
all_subs           = setdiff(all_potential_subs, excluded_subs);

subjects_haptic = all_subs(mod(all_subs, 2) == 1);
subjects_non    = all_subs(mod(all_subs, 2) == 0);

% B. Wavelet Parameters (Matched to Step 1)
SR = 250; 
min_freq = 2; max_freq = 80;
num_frex = max_freq - min_freq;
frex = logspace(log10(min_freq), log10(max_freq), num_frex);

range_cycles = [3 15]; 
% Fixed: 'wav_s' to avoid variable conflict
wav_s = logspace(log10(range_cycles(1)), log10(range_cycles(2)), num_frex) ./ (2*pi*frex);
wavtime = -2:1/SR:2;
half_wave = (length(wavtime)-1)/2;

% Time Vector for VR Epochs (Assuming -5 to 5s)
epoch_period = [-5 5];
times = linspace(epoch_period(1), epoch_period(2), diff(epoch_period) * SR);

% Window to average for "Engagement" check (0 to 1.5s after coin interaction)
t_idx_win = find(times >= 0 & times <= 1.5); 

% Output Storage
VR_Power_Gold = []; % [Sub x Freq] (Averaged over Time/Chan)
VR_Power_Wood = []; 

%% 2. PROCESSING LOOP
for sub_idx = 1:length(all_subs)
    subID = all_subs(sub_idx);
    fname_base = sprintf('CBH%04d', subID);
    
    % Load VR Data
    f_name = sprintf('%s_TapStart_all_ICA.set', fname_base);
    if ~exist(fullfile(base_path, f_name), 'file')
        fprintf('Skipping Sub %d (VR file missing)\n', subID);
        continue; 
    end
    
    fprintf('Processing VR Sub %d...\n', subID);
    EEG = pop_loadset(f_name, base_path);
    [nCh, nPnts, nTrials] = size(EEG.data);
    
    % -------------------------------------------------------------
    % CSD BLOCK (EXACT WORKING CODE)
    % -------------------------------------------------------------
    % Ensure locs.ced and E60.asc are in your path
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
    % -------------------------------------------------------------
    
    % Identify Gold vs Wood Trials
    % Assumption: S7 = Gold, S8 = Wood. 
    % We check if the epoch contains these event markers.
    gold_indices = []; 
    wood_indices = [];
    
    for ep = 1:nTrials
        evts = EEG.epoch(ep).eventtype;
        % Handle cell array of strings or simple strings
        if iscell(evts)
            % Check all events in this epoch
            if any(contains(evts, '7')), gold_indices(end+1) = ep; end
            if any(contains(evts, '8')), wood_indices(end+1) = ep; end
        else
            % Single event case
            if contains(evts, '7'), gold_indices(end+1) = ep; end
            if contains(evts, '8'), wood_indices(end+1) = ep; end
        end
    end
    
    % -------------------------------------------------------------
    % WAVELET & POWER CALCULATION
    % -------------------------------------------------------------
    % Defined as a local function handle to handle subsetting
    calc_pow = @(idxs) compute_power_subset(EEG.data(:,:,idxs), ...
                       nCh, nPnts, num_frex, wavtime, frex, wav_s, half_wave);

    % Process Gold
    if ~isempty(gold_indices)
        pow = calc_pow(gold_indices); % [Freq x Time x Chan]
        % Average over Time Window & Channels for simple Stats
        % (Or keep channels if you want Topoplots later)
        VR_Power_Gold(sub_idx, :, :) = mean(pow(:, t_idx_win, :), 2); 
    else
        VR_Power_Gold(sub_idx, :, :) = NaN; % Handle missing data
    end
    
    % Process Wood
    if ~isempty(wood_indices)
        pow = calc_pow(wood_indices);
        VR_Power_Wood(sub_idx, :, :) = mean(pow(:, t_idx_win, :), 2);
    else
        VR_Power_Wood(sub_idx, :, :) = NaN;
    end
end

%% 3. SAVE
save('./data_intermediate/VR_PSD.mat', 'VR_Power_Gold', 'VR_Power_Wood', 'all_subs', 'frex');
disp('Step 4 Complete. VR Data Saved.');

%% --- LOCAL FUNCTION (Copied from Step 1) ---
function pow_out = compute_power_subset(data_in, nCh, nPnts, nF, wavtime, frex, wav_s, half_wave)
    nTrials = size(data_in, 3);
    nWave = length(wavtime);
    nConv = nWave + (nPnts * nTrials) - 1;
    pow_out = zeros(nF, nPnts, nCh, 'single');
    
    for ch = 1:nCh
        alldata = reshape(data_in(ch, :, :), 1, []);
        dataX   = fft(alldata, nConv);
        
        for fi = 1:nF
            % Use 'wav_s' (updated variable name)
            wavelet = exp(2*1i*pi*frex(fi).*wavtime) .* exp(-wavtime.^2./(2*wav_s(fi)^2));
            waveletX = fft(wavelet, nConv);
            waveletX = waveletX ./ max(waveletX);
            
            as = ifft(waveletX .* dataX);
            as = as(half_wave+1:end-half_wave);
            as = reshape(as, nPnts, nTrials);
            
            % Induced Power: Mean of Squares
            pow_out(fi, :, ch) = mean(abs(as).^2, 2);
        end
    end
end