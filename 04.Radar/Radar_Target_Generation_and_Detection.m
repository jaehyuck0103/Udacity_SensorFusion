clear all
close all;
clc;

%% Radar Specifications 
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Frequency of operation = 77GHz
% Max Range = 200m
% Range Resolution = 1 m
% Max Velocity = 100 m/s
%%%%%%%%%%%%%%%%%%%%%%%%%%%

f_c = 77e9;
max_range = 200;
range_res = 1;
c = 3e8;  % speed of light = 3e8
%% User Defined Range and Velocity of target
% *%TODO* :
% define the target's initial position and velocity. Note : Velocity remains contant
inital_target_pos = 110;
target_vel = -20;
 


%% FMCW Waveform Generation

% *%TODO* :
%Design the FMCW waveform by giving the specs of each of its parameters.
% Calculate the Bandwidth (B), Chirp Time (Tchirp) and Slope (slope) of the FMCW chirp using the requirements above.
Bchirp = c / (2 * range_res);
Tchirp = 5.5 * (2 * max_range) / c;
chrip_slope = Bchirp / Tchirp;

%Operating carrier frequency of Radar 
fc= 77e9;             %carrier freq

                                                          
%The number of chirps in one sequence. 
% Its ideal to have 2^ value for the ease of running the FFT for Doppler Estimation. 
Nd=128;                   % #of doppler cells OR #of sent periods % number of chirps

%The number of samples on each chirp. 
Nr=1024;                  %for length of time OR # of range cells

% Timestamp for running the displacement scenario for every sample on each chirp
t=linspace(0,Nd*Tchirp,Nr*Nd); %total time for samples


%Creating the vectors for Tx, Rx and Mix based on the total samples input.
Tx=zeros(1,length(t)); %transmitted signal
Rx=zeros(1,length(t)); %received signal
Mix = zeros(1,length(t)); %beat signal

%Similar vectors for range_covered and time delay.
r_t=zeros(1,length(t));
td=zeros(1,length(t));


%% Signal generation and Moving Target simulation
% Running the radar scenario over the time. 

for i=1:length(t)         
    
    % *%TODO* :
    %For each time stamp update the Range of the Target for constant velocity. 
    r_t(i) = inital_target_pos + t(i) * target_vel;
    td(i) = 2 * r_t(i) / c;
    
    % *%TODO* :
    %For each time sample we need update the transmitted and received signal. 
    Tx(i) = cos(2 * pi * (f_c * t(i) + chrip_slope * t(i)^2 / 2));
    Rx(i) = cos(2 * pi * (f_c * (t(i) - td(i)) + chrip_slope * (t(i) - td(i))^2 / 2));
    
    % *%TODO* :
    %Now by mixing the Transmit and Receive generate the beat signal
    %This is done by element wise matrix multiplication of Transmit and Receiver Signal
    Mix(i) = Tx(i) * Rx(i);
    
end

%% RANGE MEASUREMENT


 % *%TODO* :
%reshape the vector into Nr*Nd array. 
% Nr and Nd here would also define the size of Range and Doppler FFT respectively.
Mix = reshape(Mix, [Nr, Nd]);

 % *%TODO* :
%run the FFT on the beat signal along the range bins dimension (Nr) and normalize.
sig_fft1 = fft(Mix) / Nr;

 % *%TODO* :
% Take the absolute value of FFT output
sig_fft1 = abs(sig_fft1);

 % *%TODO* :
% Output of FFT is double sided signal, but we are interested in only one side of the spectrum.
% Hence we throw out half of the samples.
sig_fft1 = sig_fft1(1:Nr/2 + 1);


%plotting the range
figure ('Name','Range from First FFT')
subplot(2,1,1)

 % *%TODO* :
 % plot FFT output 
 plot(sig_fft1)
 xlabel('range (m)')

axis ([0 200 0 1]);



%% RANGE DOPPLER RESPONSE
% The 2D FFT implementation is already provided here. This will run a 2DFFT
% on the mixed signal (beat signal) output and generate a range doppler
% map.You will implement CFAR on the generated RDM


% Range Doppler Map Generation.

% The output of the 2D FFT is an image that has reponse in the range and
% doppler FFT bins. So, it is important to convert the axis from bin sizes
% to range and doppler based on their Max values.

Mix=reshape(Mix,[Nr,Nd]);

% 2D FFT using the FFT size for both dimensions.
sig_fft2 = fft2(Mix,Nr,Nd);

% Taking just one side of signal from Range dimension.
sig_fft2 = sig_fft2(1:Nr/2,1:Nd);
sig_fft2 = fftshift (sig_fft2);
RDM = abs(sig_fft2);
RDM = 10*log10(RDM) ;

%use the surf function to plot the output of 2DFFT and to show axis in both
%dimensions
doppler_axis = linspace(-100,100,Nd);
range_axis = linspace(-200,200,Nr/2)*((Nr/2)/400);
figure,surf(doppler_axis,range_axis,RDM);

%% CFAR implementation

%Slide Window through the complete Range Doppler Map

% *%TODO* :
%Select the number of Training Cells in both the dimensions.
Tr = 8;
Td = 8;

% *%TODO* :
%Select the number of Guard Cells in both dimensions around the Cell under 
%test (CUT) for accurate estimation
Gr = 4;
Gd = 4;

% *%TODO* :
% offset the threshold by SNR value in dB
offset = 6;

% *%TODO* :
%Create a vector to store noise_level for each iteration on training cells

% *%TODO* :
%design a loop such that it slides the CUT across range doppler map by
%giving margins at the edges for Training and Guard Cells.
%For every iteration sum the signal level within all the training
%cells. To sum convert the value from logarithmic to linear using db2pow
%function. Average the summed values for all of the training
%cells used. After averaging convert it back to logarithimic using pow2db.
%Further add the offset to it to determine the threshold. Next, compare the
%signal under CUT with this threshold. If the CUT level > threshold assign
%it a value of 1, else equate it to 0.

% Use RDM[x,y] as the matrix from the output of 2D FFT for implementing CFAR

train_cols = 2 * (Tr + Gr) + 1;
train_rows = 2 * (Td + Gd) + 1;
guard_cols = 2 * Gr + 1;
guard_rows = 2 * Gd + 1;
num_train_cells = train_cols * train_rows - guard_cols * guard_rows;

RDM_pow = db2pow(RDM);
RDM_pow_integral = integralImage(RDM_pow);

% To efficiently generate the sum of values in a rectangular areas, I use the integral image.
% https://en.wikipedia.org/wiki/Summed-area_table
sum_trains = RDM_pow_integral(train_cols+1:end, train_rows+1:end) + RDM_pow_integral(1:end-train_cols, 1:end-train_rows) ...
             - RDM_pow_integral(train_cols+1:end, 1:end-train_rows) - RDM_pow_integral(1:end-train_cols, train_rows+1:end);
sum_guards = RDM_pow_integral(guard_cols+1:end, guard_rows+1:end) + RDM_pow_integral(1:end-guard_cols, 1:end-guard_rows) ...
            - RDM_pow_integral(guard_cols+1:end, 1:end-guard_rows) - RDM_pow_integral(1:end-guard_cols, guard_rows+1:end);
sum_guards = sum_guards(1+Tr:end-Tr, 1+Td:end-Td);

noise_level = (sum_trains - sum_guards) / num_train_cells;
noise_level = pow2db(noise_level) + offset;


% *%TODO* :
% The process above will generate a thresholded block, which is smaller 
%than the Range Doppler Map as the CUT cannot be located at the edges of
%matrix. Hence,few cells will not be thresholded. To keep the map size same
% set those values to 0. 

% I give infinite threshold to the edges in order to suppress signal_cfar at the edges.
noise_level_padded = padarray(noise_level, [Tr+Gr, Td+Gd], inf, 'both');
signal_cfar = double(RDM > noise_level_padded);


% *%TODO* :
%display the CFAR output using the Surf function like we did for Range
%Doppler Response output.
figure, surf(doppler_axis, range_axis, signal_cfar);
colorbar;
