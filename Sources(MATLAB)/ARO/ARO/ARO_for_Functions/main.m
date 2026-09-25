%--------------------------------------------------------------------------
%%% Artificial Rabbits Optimization (ARO) for 23 functions %%%
% ARO code v1.0.
% Developed in MATLAB R2011b
% --------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BestX:The best solution                  %
% BestF:The best fitness                   %
% HisBestF:History of the best fitness     %
% FunIndex：Index of functions             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
clear;

MaxIteration=1000;
PopSize=50;
FunIndex=1;
    [BestX,BestF,HisBestF]=ARO(FunIndex,MaxIteration,PopSize);
    % display(['FunIndex=', num2str(FunIndex)]);
    display(['The best fitness of F',num2str(FunIndex),' is: ', num2str(BestF)]);
    %display(['The best solution is: ', num2str(BestX)]);

    if BestF>0
        semilogy(HisBestF,'r','LineWidth',2);
    else
        plot(HisBestF,'r','LineWidth',2);
    end

    xlabel('Iterations');
    ylabel('Fitness');
    title(['F',num2str(FunIndex)]);


