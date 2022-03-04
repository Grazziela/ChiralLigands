clear all;
close all;

%% LASSO FEATURE SELECTION 
% Developed by Grazziela P Figueredo
% Last modified 20/05/2019


% If you need to generate feature selection for multiple files for multiple
% descriptors, add their name here. These names need to match the excell
% sheet names
descriptorsInvestigated = {'LigandSubstrateBoronDragonDescriptors'};
%descriptorsInvestigated = {'DragonDescriptors', 'AllFragmentDescriptors','scan1', 'scan2', 'scan3', 'scan4'};

for d = 1:size(descriptorsInvestigated,2)
    %%
    file_name = '';
    filename = [file_name, descriptorsInvestigated{d},''];
    [data,txt,Main] = xlsread([filename,'.xlsx']);
    
    % Remove columns where NaN
    index = find(any( isnan( data ), 1 )==1)
    data(:,index) = [];
    Main(:,index) = [];
       
    X = data(:, 1:end-1);
    y = data(:, end);
    
    % LASSO feature selection starts
    [B,FitInfo] = lassoglm(X,y,'gamma','CV',2, 'PredictorNames', Main(1, 1:end-1));
    
    % The features that provide the min square error are selected
    idxLambdaMinMSE = FitInfo.IndexMinDeviance;
    minMSEModelPredictors = FitInfo.PredictorNames(B(:,idxLambdaMinMSE)~=0);
    
    idxLambda1SE = FitInfo.Index1SE;
    sparseModelPredictors = FitInfo.PredictorNames(B(:,idxLambda1SE)~=0);
    
    lassoPlot(B,FitInfo,'plottype','CV');
    legend('show')
    
    % Creating a new array with the main descriptors
    
    vectorDescriptors = [];
    %%
    
    number_predictors = size(minMSEModelPredictors,2);
    
    for i = 1:number_predictors
        % Assigning the descriptors to the monomers and saving them on files
        index = find(strcmp(Main(1,:), minMSEModelPredictors(i)));
        vectorDescriptors = [vectorDescriptors, Main(:,index)];
    end
    
    % Saving output file
    output_file = [file_name,descriptorsInvestigated{d},'_LASSO.xlsx'];
    vectorOutput = [vectorDescriptors, Main(:, end)];
    xlswrite(output_file, vectorOutput);
end