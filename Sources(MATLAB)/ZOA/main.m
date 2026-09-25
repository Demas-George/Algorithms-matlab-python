


%% 
% Zebra Optimization Algorithm: A New Bio-inspired Optimization Algorithm for Solving Optimization Algorithm
% IEEE access
% Eva Trojovská1, Mohammad Dehghani1, and Pavel Trojovský1*
% 1Department of Mathematics, Faculty of Science, University of Hradec Králové, 50003 Hradec Králové, Czech Republic

% " Optimizer"
%%
clc
clear
close all
%%

%%

Fun_name='F1'; % number of test functions: 'F1' to 'F23'
SearchAgents=30;                      % number of Zebras (population members) 
Max_iterations=1000;                  % maximum number of iterations
[lowerbound,upperbound,dimension,fitness]=fun_info(Fun_name); % Object function information
[Best_score,Best_pos,ZOA_curve]=ZOA(SearchAgents,Max_iterations,lowerbound,upperbound,dimension,fitness);  % Calculating the solution of the given problem using ZOA 

%%

display(['The best optimal value of the objective funciton found by ZOA  for ' num2str(Fun_name),'  is : ', num2str(Best_score)]);
%%     