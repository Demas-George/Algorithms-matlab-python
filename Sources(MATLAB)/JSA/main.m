%-----------------------------------------------------------------------------------------------------------------%
%  Jellyfish Search Optimizer (JS) source codes demo version 1.0  Developed in MATLAB R2016a                      %
%  Author and programmer:                                                                                         %
%         Professor        Jui-Sheng Chou                                                                         %
%         Ph.D. Candidate  Dinh- Nhat Truong                                                                      %
%  Paper: A Novel Metaheuristic Optimizer Inspired By Behavior of Jellyfish in Ocean,                             %
%         Applied Mathematics and Computation. Computation, Volume 389, 15 January 2021, 125535.                  %
%  DOI:   https://doi.org/10.1016/j.amc.2020.125535                                                               %
%                                     PiM Lab, NTUST, Taipei, Taiwan, July-2020                                   %
%-----------------------------------------------------------------------------------------------------------------%
function main
clear all
clc
%% Select function
fnumber=1;            % Select function
[lb ub dim]=boundcondition(fnumber);
%% Set the parameters
Npop=50;              % Number of jellyfish
Max_iteration=10000;   % Maximum numbef of iterations
para=[Max_iteration Npop];
%% Run JS optimizer
tic;
[u,fval,NumEval,fbestvl]=js(@fobj,fnumber,lb,ub,dim,para);
time=toc;
%% Display optimal results
display(['-------------------------------------------------------------------------']);
display(['  Jellyfish Search Optimizer (JS) for mathematical benchmark problems    ']);
display(['-------------------------------------------------------------------------']);
display(['The best solution obtained by JS is : ', num2str(u)]);
display(['The best optimal value of the objective function found by JS is : ', num2str(fval)]);
%% Save optimal results
save('result.mat','time','u','fval','NumEval','fbestvl');
end





