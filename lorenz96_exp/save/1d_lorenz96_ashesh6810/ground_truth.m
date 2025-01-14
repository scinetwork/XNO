%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MATLAB Script for Plotting Ground Truth and Multiple Predictions
% 
% - Loads 5 .mat files, each with variables: x, y, pred
% - x: (Time x Variables) ground truth states
% - y, pred: (Time x Variables) predicted delta_x in each time step
% 
% The ground truth for time step t is x(t,:).
% The predicted state for time step t is computed as:
%    t_pred(1,:) = x(1,:)
%    t_pred(k,:) = t_pred(k-1,:) + pred(k-1,:)
% for k = 2,...,T
%
% Author: ChatGPT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;

%% ---------------- User Parameters ---------------- %%
% List of .mat files (each contains x, y, pred)
fileNames = {
    '/Users/sina/Desktop/data/data/1d_lorenz96_ashesh6810/1d_lorenz96_ashesh6810_fno.mat',
    '/Users/sina/Desktop/data/data/1d_lorenz96_ashesh6810/1d_lorenz96_ashesh6810_hno.mat',
    '/Users/sina/Desktop/data/data/1d_lorenz96_ashesh6810/1d_lorenz96_ashesh6810_lno.mat',
    '/Users/sina/Desktop/data/data/1d_lorenz96_ashesh6810/1d_lorenz96_ashesh6810_wno.mat',
};

% Modes (or model names) for legend labeling
modes = {
    'FNO',
    'HNO',
    'LNO',
    'WNO',
};

% Which variables to plot (subset of the 8 variables)
varIndices = 1:1;  % Example: plot variables #1 and #2

% Which time range to plot (subset of the 5000 timesteps)
timeRange = 1:100;  % Example: all 5000 timesteps

%% ---------------- Load Data and Prepare Plot ---------------- %%
% Load the first file (to retrieve ground truth 'x' and check size)
dataRef = load(fileNames{1});
xRef = dataRef.x;  % Ground truth reference
% yRef = dataRef.y; % If you need it for something else

% For convenience
numFiles = numel(fileNames);
numVars  = size(xRef, 3);

% Safety checks (optional)
if any(varIndices > numVars)
    error('One of the requested varIndices exceeds the number of variables (%d).', numVars);
end
if any(timeRange > size(xRef,1))
    error('One of the requested time steps exceeds the length of x (%d).', size(xRef,1));
end
if numFiles ~= numel(modes)
    error('Number of modes (%d) must match number of files (%d).', numel(modes), numFiles);
end

%% ---------------- Main Loop: One Figure per Variable ---------------- %%
for v = varIndices
    
    % Create figure for this variable
    figure('Name', sprintf('Variable #%d Comparison', v), 'Color', 'w');
    hold on;
    
    x_true = dataRef.x(timeRange, v);
    y_true = dataRef.y(timeRange, v);
    
    % We'll rebuild x from y, starting at x_true(1)...
    x_rebuilt = zeros(length(timeRange),1);
    x_rebuilt(1) = x_true(1);
    
    % Because x(t+1) = x(t) + y(t), we do:
    for t = 2:length(timeRange)
        x_rebuilt(t) = x_rebuilt(t-1) + y_true(t-1);
    end

    plot(timeRange, x_rebuilt, 'k-', 'LineWidth', 2, 'DisplayName', 'Ground Truth');
    
    % Loop over each .mat file, load predictions, and plot
    for iFile = 1:numFiles
        % Load file
        data = load(fileNames{iFile});
        x_gt = data.x(timeRange, v);     % ground truth portion (same as xRef ideally)
        pred = data.pred(timeRange, v);  % predicted delta_x portion
        
        % Build predicted trajectory by cumulative summation:
        % t_pred(1) = x_gt(1)
        % t_pred(k) = t_pred(k-1) + pred(k-1)
        t_pred = zeros(length(timeRange), 1);
        t_pred(1) = x_gt(1);
        for t = 2:length(timeRange)
            t_pred(t) = t_pred(t-1) + pred(t-1);
        end
        
        % Plot predicted evolution
        plot(timeRange, t_pred, 'LineWidth', 1.5, 'DisplayName', modes{iFile});
    end
    
    % Configure the plot
    xlabel('Time Step', 'FontWeight', 'bold');
    ylabel(sprintf('Variable #%d Value', v), 'FontWeight', 'bold');
    legend('Location', 'best');
    title(sprintf('Ground Truth vs. Predictions for Variable #%d', v), 'FontWeight', 'bold');
    grid on;
    hold off;
    
end