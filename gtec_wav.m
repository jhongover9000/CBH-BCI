
%%%%Author: Haneen Alsuradi
%%%%Date: June 2020
%%%%Function: Creates a matrix for TF data using morlet wavelet tranform.
%%%%Also, visualization of topography, PSD vs time and TF plots.

%% TF Analysis using morlet wavelet
%clear all;
close all;
clc;


path_to_rawdata = './';
path_to_epoched      = './epoched/';

nSubject        = 2;
events          = {'motor_execution', 'ONT';
                   'rest', 'OST';
                   };
baseline_period = [-500 0];

epoch_period = [-2 3];

nEvents = 2;
nChannels       = 60;
eegset_dir      = './epoched/';
SR = 250;      % EEG rate

% frequency parameters
min_freq =  2;
max_freq = 80;
num_frex = max_freq - min_freq;
frex = logspace(log10(min_freq),log10(max_freq), num_frex);


% other wavelet parameters
range_cycles = [ 2 12 ];
s    = logspace(log10(range_cycles(1)),log10(range_cycles(2)),num_frex)./(2*pi*frex);
wavtime = -2:1/SR:2;
half_wave = (length(wavtime)-1)/2



nTimes = (2+3)*1000;
phase_name = 'gtec';

% initialize output time-frequency data
tf = zeros(nSubject,length(frex),nTimes,nChannels,nEvents, 'single');

for sub = 1:nSubject

    % iterate over events, NT ST CT
    for evt = 1:2
        
        % subtract 1 from sub to match datasets (can be fixed ground up later)
        eeg_file   = ['gtec_' int2str(sub) '_'  events{evt,2} '.set'];
    
        EEG     = pop_loadset('filename', eeg_file, 'filepath', eegset_dir);
    
        % FFT parameters
        nWave = length(wavtime);
        nData = EEG.pnts * EEG.trials;
        nConv = nWave + nData - 1;
        
        for ch=1:nChannels
    
            channel2use = EEG.chanlocs(ch).labels;
    
            % now compute the FFT of all trials concatenated
            alldata = reshape( EEG.data(strcmpi(channel2use,{EEG.chanlocs.labels}),:,:) ,1,[]);
            dataX   = fft( alldata ,nConv );
            
            % initialize output time-frequency data
            % tf_temp = zeros(length(frex), EEG.pnts);
    
            % loop over frequencies
            for fi=1:length(frex)
    
                % create wavelet and get its FFT
                % the wavelet doesn't change on each trial...
                wavelet  = exp(2*1i*pi*frex(fi).*wavtime) .* exp(-wavtime.^2./(2*s(fi)^2));
                waveletX = fft(wavelet,nConv);
                waveletX = waveletX ./ max(waveletX);
    
                % run convolution
                as = ifft(waveletX .* dataX);
                as = as(half_wave+1:end-half_wave);
    
                as = reshape(as, EEG.pnts, EEG.trials);

                % compute power and average over trials
                tf(sub,fi,:,ch, mod(evt,nEvents)+1) = mean(abs(as).^2 ,2);
            end
        end
    end       
    tf=single(tf);
end
tf=single(tf);
save([phase_name '_tf_morlet_original.mat'] , 'tf', '-v7.3');
% clear tf;


%% db matrix including all subjects: tf=%sub x freq x time x ch x event

nSubject= 33;
num_frex= 78;

% 500 ms (based on the preprocessing baselines)
baseline_windows = [-2500 -2100];

times = -5000:1:(nTimes-5001);

% 3 Conditions for each phase - NT, ST, CT
% indicies instead of values in ms (convert)
baseidx = reshape( dsearchn(times',baseline_windows(:)), [],2);

% iterate over categories of same nTimes - Rest, PoI, Begin
for phase = 3

    % MI: 3 + 1 trial, 1.25 + 1 baseline
    if phase == 1
        nTimes = (5 + 5)*1000;
        phase_name = 'MO';

    % ME: 4 + 1 trial, 1.25 + 1 baseline
    elseif phase == 2
        nTimes = (5 + 5)*1000;
        phase_name = 'ME';

    % MI: 3 + 1 trial, 1.25 + 1 baseline
    else
        nTimes = (5 + 5)*1000;
        phase_name = 'MI';
    end

    % load corresponding convoluted mat
    % load([phase_name '_tf_morlet_original.mat']);

    % splitting the tf into variables (for each condition)
    tf_sub_1=tf(:,:,:,:,1);
    tf_sub_2=tf(:,:,:,:,2);
    tf_sub_3=tf(:,:,:,:,3);
    
    clear tf;
    
    % create new matrix for db 
    tf_m1_db_sub=zeros(nSubject,num_frex,nTimes,nChannels);
    tf_m2_db_sub=zeros(nSubject,num_frex,nTimes,nChannels);

    for sub=1:nSubject
        for ch=1:nChannels
        % get activity
        activity_m1 = squeeze(tf_sub_1(sub,:,:,ch));
        activity_m2 = squeeze(tf_sub_2(sub,:,:,ch));
        % get baselines
        baseline1 = mean( tf_sub_1(sub,:,baseidx(1):baseidx(2),ch) ,3)';
        baseline2 = mean( tf_sub_2(sub,:,baseidx(1):baseidx(2),ch) ,3)';
        baseline = (baseline1 + baseline2 )/2;
        % baseline=(baseline1+baseline2+baseline3)/3;
        % normalize baseline
        tf_m1_db_sub(sub,:,:,ch) = 10*log10( bsxfun(@rdivide, activity_m1, baseline1) );
        tf_m2_db_sub(sub,:,:,ch) = 10*log10( bsxfun(@rdivide, activity_m2, baseline2) );
        end
    end
    
    % decrease size
    tf_m1_db_sub = single(tf_m1_db_sub);
    tf_m2_db_sub = single(tf_m2_db_sub);

    %save dbs
    save([phase_name '_tf_m1_db.mat'] , 'tf_m1_db_sub', '-v7.3');
    save([phase_name '_tf_m2_db.mat'] , 'tf_m2_db_sub', '-v7.3');

end


%% Topo plots avg over all subjects after normalization

load EEG_chlocs_60.mat

times = linspace(-5000,4999,nTimes);

f=[6 11];
z1=-1.5;
z2=1.5;
t0= find(times == 0); % select a time in ms
figure()
n=20;  
step=200;
%c=parula;
c=jet;
for i=1:n
    subplot(4,n,i)
    temp1=mean(mean(squeeze(mean(tf_m1_db_sub(:,f(1):f(2),t0+((i-1)*step):t0+(i*step)-1,:),1))));
    
    topoplot(squeeze(temp1), EEG_chlocs,'maplimits',[z1 z2],'electrodes','off','colormap',c,'style','map' ) %select a condition
    %topoplot(temp1, EEG_chlocs,'maplimits',[z1 z2],'electrodes','off','colormap',c,'style','map','emarker2', {idx,'o','k'}) %select a condition

    subplot(4,n,i+n)   
    temp2=mean(mean(squeeze(mean(tf_m2_db_sub(:,f(1):f(2),t0+((i-1)*step):t0+(i*step)-1,:),1))));

    topoplot(temp2, EEG_chlocs,'maplimits',[z1 z2],'electrodes','off','colormap',c,'style','map' ) %select a condition
    
    subplot(4,n,i+n+n)   
    temp3=mean(mean(squeeze(mean(tf_m3_db_sub(:,f(1):f(2),t0+((i-1)*step):t0+(i*step)-1,:),1))));

    topoplot(temp3, EEG_chlocs,'maplimits',[z1 z2],'electrodes','off','colormap',c,'style','map' ) %select a condition
    % 
    % subplot(4,n,i+n+n+n)   
    % temp4=mean(mean(squeeze(mean(tf_m2_db_sub(:,f(1):f(2),t0+((i-1)*step):t0+(i*step)-1,:),1))));
    % 
    % topoplot(temp4, EEG_chlocs,'maplimits',[z1 z2],'electrodes','off','colormap',c,'style','map' ) %select a condition
end
set(gcf, 'color', 'w');  % Set the background color of the figure to white


%%
nCh=60;
f=[2 8];

figure() ;
 for plotId = 1 : nCh
      
    subplot(6, 10, plotId) ;
    temp=squeeze(mean(mean(tf_m1_db_sub(:,f(1):f(2),:,plotId))));
    plot(times,temp)

    hold on
     temp=squeeze(mean(mean(tf_m2_db_sub(:,f(1):f(2),:,plotId))));
     plot(times,temp)

    
     hold on
     temp=squeeze(mean(mean(tf_m3_db_sub(:,f(1):f(2),:,plotId))));
     plot(times,temp)
     
     hold on
    temp=squeeze(mean(mean(tf_m4_db_sub(:,f(1):f(2),:,plotId))));
    plot(times,temp)
    xlim([-500 1000])

    title(vals{plotId})

 end

%% Plotting TF plot averaged over subjects
%ch=[11,12,13,17,18,40,41,42,43,44,45,46,47,48];
%chs={'CP1','CP3','C3','C5','C1','CP5'};
%chs={'FC1', 'C3', 'CP3', 'C1'};
chs={'F1','F2','Fz','FC1','FC2','FCz'};
%chs={'FC1','FC2','FCz','Cz'};
%chs={'C1','C3'};
%chs={'P2'};
%chs={'Fp1','Fp2'};
%chs={'Oz', 'O1','O2'};
%chs={'CP2','CP4','P2','P4'}
%chs={'P2','P4','PO4'};
%chs={'P1','P2','Pz'};
%chs={'AFz','F1','F2','Fz'};
%chs={'FC1', 'FC3','CP1', 'C3', 'CP3', 'C1'};
idx=[];
ctr=1;
for i=chs
  idx(ctr)=find(ismember(vals, i));
  ctr=ctr+1;
end

% need to plot for each of the conditions
figure(4)
colormap(jet)
climdb  = [-2 2];
climpct = [-90 90];
subplot(2,2,1)
% selecting specific channel (idx)
% two means (first, third; subject, channels) - outputs time&freq
temp1=mean(squeeze(mean(tf_m1_db_sub(:,:,:,idx),1)),3);
%temp1(1:5,:)=temp1(1:5,:)-0.4;
contourf(times,frex,temp1,40,'linecolor','none') % 
%set(gca,'clim',climdb,'ydir','normal','xlim',[-1000 800])
% without log scale all will be plotted normally
set(gca,'clim',climdb,'ydir','normal','xlim',[-200 1500],'yscale','log' )
ylim([1 50 ])
xlabel('Time (ms)')
ylabel('Frequencies (Hz)')
set(gca,'FontSize',16)
set(gca,'FontWeight','Bold')
% title('No Delay')
% xline(0,'LineWidth',2)
yticks([ 3 8 13 30 50])
xline(0,'-.r','LineWidth',2)


subplot(2,2,2)
temp2=mean(squeeze(mean(tf_m2_db_sub(:,:,:,idx),1)),3);
contourf(times,frex,circshift(temp2,1,1)-circshift(temp1,1,1),40,'linecolor','none')
%set(gca,'clim',climdb,'ydir','normal','xlim',[-1000 800])
set(gca,'clim',climdb,'ydir','normal','xlim',[-200 1500] ,'yscale','log' )
ylim([1 50 ])
xlabel('Time (ms)')
ylabel('Frequencies (Hz)')
set(gca,'FontSize',16)
set(gca,'FontWeight','Bold')
xline(120,'-.r','LineWidth',2)
yticks([3 8 13 30 50 ])

subplot(2,2,3)
temp3=mean(squeeze(mean(tf_m3_db_sub(:,:,:,idx),1)),3);
contourf(times,frex,circshift(temp3,1,1)-circshift(temp1,1,1),40,'linecolor','none')
%set(gca,'clim',climdb,'ydir','normal','xlim',[-1000 800])
set(gca,'clim',climdb,'ydir','normal','xlim',[-200 1500] ,'yscale','log')
ylim([1 50])
xlabel('Time (ms)')
ylabel('Frequencies (Hz)')
set(gca,'FontSize',16)
set(gca,'FontWeight','Bold')
xline(250,'-.r','LineWidth',2)
yticks([3 8 13 30 50])

subplot(2,2,4)
temp4=mean(squeeze(mean(tf_m4_db_sub(:,:,:,idx),1)),3);
contourf(times,frex,circshift(temp4,1,1)-circshift(temp1,1,1),40,'linecolor','none')
%set(gca,'clim',climdb,'ydir','normal','xlim',[-1000 800])
set(gca,'clim',climdb,'ydir','normal','xlim',[-200 1500] ,'yscale','log' )
%set(gca,'clim',[-2 2],'xlim',[-200 1000],'yscale','log','ytick',logspace(log10(min_freq),log10(max_freq),6),'yticklabel',round(logspace(log10(min_freq),log10(max_freq),6)*10)/10)

ylim([1 50 ])
xlabel('Time (ms)')
ylabel('Frequencies (Hz)')
set(gca,'FontSize',16)
set(gca,'FontWeight','Bold')
xline(400,'-.r','LineWidth',2)
yticks([  3 8 13 30 50 ])