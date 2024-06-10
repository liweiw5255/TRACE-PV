% Define the Simulink model name
model = 'myModel';

% Load the Simulink model
load_system(model);

% % Define different sets of parameters
% paramSets = [
%     struct('Gain', 1, 'Integrator_IC', 0);
%     struct('Gain', 2, 'Integrator_IC', 1);
%     struct('Gain', 3, 'Integrator_IC', 2)
% ];
% 
% % Number of simulation cases
numCases = 100; %length(paramSets);

% Initialize a cell array to store simulation results
simResults = cell(numCases, 1);

% Start parallel pool (if not already started)
if isempty(gcp('nocreate'))
    parpool;
end

% Use parfor to run simulations in parallel
parfor i = 1:numCases
    % Create a unique copy of the model for this worker
    modelCopy = [model '_copy' num2str(i)];
    save_system(model, modelCopy);
    load_system(modelCopy);
    
    % Set parameters in the model workspace
    set_param([modelCopy '/Gain'], 'Gain', num2str(paramSets(i).Gain));
    set_param([modelCopy '/Integrator'], 'InitialCondition', num2str(paramSets(i).Integrator_IC));
    
    % Run the simulation
    simOut = sim(modelCopy);
    
    % Store the simulation results
    simResults{i} = simOut;
    
    % Optionally save results to a file
    save(['simResult_' num2str(i) '.mat'], 'simOut');
    
    % Close the copied model
    close_system(modelCopy, 0);
    delete([modelCopy '.slx']); % Delete the copied model file
end

% Close the original Simulink model
close_system(model, 0);

% Process or visualize the results as needed
for i = 1:numCases
    disp(['Results for case ' num2str(i) ':']);
    disp(simResults{i});
end

% Shut down the parallel pool
delete(gcp('nocreate'));