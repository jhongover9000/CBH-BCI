%%%% Author: Haneen
%%%% Date: Sep 2023
%%%% Pre-processing for the MIT data.
%NICE PAPER: The Artifact Subspace Reconstruction (ASR) for EEG Signal Correction. A Comparative Study
clear all;
close all;
clc; 

path_to_rawdata = '../data/rawdata/eegrawdata';
path_to_epoched      = '../data/ME_NT_epoched/';

              
% NT: Natural Thumb
% ST: Supernumerary Thumb
% CT: Concurrent 
% O: Observe
% E: Execute
% I: Imagine
% A: Actual

nSubject        = 33;

% 
% events          = {'S  2', 'ONT';
%                    'S  3', 'OST';
%                    'S  4', 'OCT'; 
%                    }; 
% 
events          = {'S  5', 'ENT';  
                   % 'S  6', 'EST';
                   % 'S  7', 'ECT'; 
                   }; 

% events          = {
%                    'S  9', 'INT';  
%                    };



baseline_period = [-500 -200];

epoch_period_ob=[-3 4];
epoch_period_ex=[-8 5];
epoch_period_im=[-3 4];
imagine_triggers = ['S  9'];


load NewEasyCap63
total_discard=[];
nt_discard=[];
st_discard=[];
ct_discard=[];

for sub = 1:nSubject
        num_discard=0;
        num_nt_discard=0;
        num_st_discard=0;
        num_ct_discard=0;

        set_file    = [path_to_rawdata 'MIT' int2str(sub) '.vhdr'];
        disp(['reading...' set_file]);
        
        EEG = pop_fileio(set_file);   %EEG file reading

        EEG.chanlocs  = struct(chanlocsEasyCapNoRef); %Loading channel locations
        
        EEG = eeg_checkset( EEG );  %Confirm no errors
        
        
        %eeglab redraw; %Switch from script to GUI
        %Adding the reference channel as #64 FCz
        EEG=pop_chanedit(EEG, 'append',63,'changefield',{64 'labels' 'FCz'},'changefield',{64 'X' '0.383'},'changefield',{64 'Y' '0'},'changefield',{64 'Z' '0.923'},'convert',{'cart2all'});
        EEG = eeg_checkset( EEG );

       
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Bad Electrode Removal: 64 -> 56 channels
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        ex_channels     = [5 10 21 27];
        EEG = pop_select(EEG, 'nochannel', ex_channels);    
        nCh = EEG.nbchan;
                
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Band Pass Filter [0.1-85Hz], 33000 order
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % cutoff is 80 so we choose 85. Regarding the order 33000, try with
        % small number and check what the command suggest for you. It is
        % cutoff Frequency dependant 
        EEG = pop_eegfiltnew(EEG, 0.5, 85, 33000, 0, [], 1);  
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
        % CAR [38, 51]
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

        % Discard trials self-reported as no
        for j = 1:size(subEEG.event,2)
            if ismember(subEEG.event(j).type, imagine_triggers) 
                if strcmp(subEEG.event(j+1).type, 'S 13')
                    subEEG.event(j).label = 'skip';
                    num_discard = num_discard +1;

                    if strcmp(subEEG.event(j).type, 'S  9')
                    num_nt_discard= num_nt_discard+1;

                    elseif strcmp(subEEG.event(j).type, 'S 10')
                        num_st_discard= num_st_discard+1;
    
                    elseif strcmp(subEEG.event(j).type, 'S 11')
                        num_ct_discard= num_ct_discard+1;
                    end

                end

            end
        end
        total_discard= [total_discard num_discard];
        nt_discard= [nt_discard num_nt_discard];
        st_discard= [st_discard num_st_discard];
        ct_discard= [ct_discard num_ct_discard];


 
        %This line is to create a field called label.
        subEEG.event(1).label='start';
        del=0;
        for j = 1:size(subEEG.event,2)
            if strcmp(subEEG.event(j-del).label, 'skip')
                subEEG.event(j-del) = [];
                del = del + 1;      
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Epoching 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   

        for evt = 1
            epoch_file  = ['MIT' int2str(sub) '_' events{evt,2}];

            disp(evt)
            epoch = pop_epoch(subEEG, events(evt,1), epoch_period_im, 'epochinfo', 'yes'); 
            epoch = pop_rmbase(epoch, baseline_period);
            epoch = eeg_checkset(epoch);


            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % save epoched set
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %save epochs
            epoch= pop_saveset(epoch, 'filename', epoch_file,'filepath', path_to_epoched, 'savemode', 'onefile');

        end
end


% save('discarded_trials.mat','total_discard','nt_discard','st_discard','ct_discard')