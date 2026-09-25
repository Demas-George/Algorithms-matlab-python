% Written by Dr. Seyedali Mirjalili
% To wach videos on this algorithm, enrol to the course with 95% discount using the following links: 

% ************************************************************************************************************************************************* 
%  A course on "Optimization Problems and Algorithms: how to understand, formulation, and solve optimization problems": 
%  https://www.udemy.com/optimisation/?couponCode=MATHWORKSREF
% ************************************************************************************************************************************************* 
%  "Introduction to Genetic Algorithms: Theory and Applications" 
%  https://www.udemy.com/geneticalgorithm/?couponCode=MATHWORKSREF
% ************************************************************************************************************************************************* 

clear 
close all
clc

% Problem preparation 
problem.nVar = 2;
problem.ub = 50 * ones(1, 2);
problem.lb = -50 * ones(1, 2);
problem.fobj = @ObjectiveFunction;

% PSO parameters 
noP = 4;
maxIter = 500;
visFlag = 1; % set this to 0 if you do not want visualization

RunNo  = 30; 
BestSolutions_PSO = zeros(1 , RunNo);


 [ GBEST  , cgcurve ] = PSO( noP , maxIter, problem , visFlag ) ;
 
 disp('Best solution found')
 GBEST.X
 disp('Best objective value')
 GBEST.O
