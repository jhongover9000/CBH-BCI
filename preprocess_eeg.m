%%%% Author: Haneen
%%%% Date: Sep 2023
%%%% Pre-processing for the MIT data.
% clear all;
% close all;
clc;

path_to_rawdata = './';
path_to_epoched      = './epoched/';

              
% NT: Natural Thumb
% ST: Supernumerary Thumb
% CT: Concurrent 
% O: Observe
% E: Eexecute
% I: Imagine
% A: Actual

nSubject        = 2;
events          = {'motor_execution', 'ONT';
                   'rest', 'OST';
                   };
baseline_period = [-600 -200];
baseline_period_ex = [-2500 -2100];

epoch_period_ob=[-5 5];
epoch_period_ex=[-5 5];
epoch_period_im=[-3 4];
imagine_triggers = ['S  9','S 10','S 11'];

load chanlocs_gtec.mat

num_discard=0;
total_discard=[];
for sub = 1:nSubject
        
        set_file    = [path_to_rawdata 'gtec_' int2str(sub) '.set'];
        disp(['reading...' set_file]);
        
        EEG = pop_loadset(set_file);   %EEG file reading

        EEG.chanlocs  = struct(chanlocs_gtec); %Loading channel locations
        
        EEG = eeg_checkset( EEG );  %Confirm no errors
        
        
        %eeglab redraw; %Switch from script to GUI
        %Adding the reference channel as #64 FCz
        EEG=pop_chanedit(EEG, 'append',63,'changefield',{64 'labels' 'FCz'},'changefield',{64 'X' '0.383'},'changefield',{64 'Y' '0'},'changefield',{64 'Z' '0.923'},'convert',{'cart2all'});
        EEG = eeg_checkset( EEG );

       

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Band Pass Filter [0.1-85Hz], 33000 order
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % cutoff is 80 so we choose 85. Regarding the order 33000, try with
        % small number and check what the command suggest for you. It is
        % cutoff Frequency dependant 
        EEG = pop_eegfiltnew(EEG, 0.1, 85, 33000, 0, [], 1);  
        %EEG = pop_eegfiltnew(EEG, Lpass, Hpass, BP_filter_order, true, [], 1);
        EEG = eeg_checkset(EEG);
        close all;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Notch filter 50Hz
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
         EEG = pop_eegfiltnew(EEG, 49.5, 50.5, 8250, 1, [], 1); %min 4126, better 8250
         EEG = eeg_checkset(EEG);
         close all;

        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Artifact Subspace Reconstruction 
        % to reject bad channels and correct continuous data using Artifact Subspace Reconstruction (ASR)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        infoCh   = {EEG.chanlocs.labels};
        saveEEG  = struct(EEG);
    
        EEG = clean_rawdata(EEG, 10, [0.25 0.75], 0.8, 4, 20, 0.5);

        nChMiss  = EEG.nbchan;
        [M,N]   = size(EEG.data);

        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Find Miss Channels 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        EEG.data    = [EEG.data(:,:);
                        zeros(nCh-nChMiss,N);
                        ];
    
        missCh      = {};
        missChNum   = [];
        for i = 1:nCh
            temp  = max(strcmp(infoCh(i),{EEG.chanlocs.labels}));
            if temp == 0
                missCh    = [missCh infoCh(i)];
                missChNum = [missChNum i];
                EEG.data(i+1:end,:) = EEG.data(i:end-1,:);
                EEG.data(i,:)       = zeros(1,N);
            end
        end
    
        EEG.nbchan    = saveEEG.nbchan;
        EEG.chanlocs  = struct(saveEEG.chanlocs);
        EEG           = eeg_checkset(EEG);
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % interpolation Channel
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        EEG = pop_interp(EEG, missChNum , 'spherical');
        EEG = eeg_checkset(EEG);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % CAR: 59 -> 60 channels
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %retain reference channel
        EEG = pop_reref( EEG, [],'refloc',struct('theta',{0},'radius',{0.1252},'labels',{'FCz'},'sph_theta',{0},'sph_phi',{67.4639},'X',{0.383},'Y',{0},'Z',{0.923},'sph_radius',{0.99931},'type',{''},'ref',{''},'urchan',{[]},'datachan',{0}));
        EEG = eeg_checkset( EEG );  %Confirm no errors
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Signal Processing - done once to reduce computations
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % can be skipped and EEG be altered directly for memory saving
        subEEG = struct(EEG);
        % remove empty markers
        del = 0;
        for j = 1:size(subEEG.event,2)
            if strcmp(subEEG.event(j-del).type, 'empty')
                subEEG.event(j-del) = [];
                del = del + 1;      
            end
        end
        num_discard = 0;
        % Discard trials self-reported as no
        for j = 1:size(subEEG.event,2)
            if ismember(subEEG.event(j).type, imagine_triggers) 
                if strcmp(subEEG.event(j+1).type, 'S 13')
                    subEEG.event(j).type = 'skip';
                    num_discard = num_discard +1;
                end
            end
        end
        total_discard= [total_discard num_discard];
    
        del=0;
        for j = 1:size(subEEG.event,2)
            if strcmp(subEEG.event(j-del).type, 'skip')
                subEEG.event(j-del) = [];
                del = del + 1;      
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Epoching 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   

        for evt = 7:9
            epoch_file  = ['MIT' int2str(sub) '_' events{evt,2}];
            
            if evt < 4
                disp(evt)
                epoch = pop_epoch(subEEG, events(evt,1), epoch_period_ob, 'epochinfo', 'yes'); 
                epoch = pop_rmbase(epoch, baseline_period);
                epoch = eeg_checkset(epoch);
            elseif evt < 7
                disp(evt)
                epoch = pop_epoch(subEEG, events(evt,1), epoch_period_ex, 'epochinfo', 'yes'); 
                epoch = pop_rmbase(epoch, baseline_period_ex);
                epoch = eeg_checkset(epoch);
            else
                disp(evt)
                epoch = pop_epoch(subEEG, events(evt,1), epoch_period_im, 'epochinfo', 'yes'); 
                epoch = pop_rmbase(epoch, baseline_period);
                epoch = eeg_checkset(epoch);
            end


            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % save epoched set
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %save epochs
            epoch= pop_saveset(epoch, 'filename', epoch_file,'filepath', path_to_epoched, 'savemode', 'onefile');

        end

end


% save('total_discard.mat','total_discard')