%--------------------------------------------------------------------------
%%% Artificial Rabbits Optimization (ARO) for 5 engineering designs %%%
% ARO code v1.0.
% Developed in MATLAB R2011b
% --------------------------------------------------------------------------
    %%%%%%%%%%% Engineering lists %%%%%%%%%%%%%%
    %  1.Pressure vessel design                %
    %  2.Rolling element bearing design        %
    %  3.Tension/compression spring design     %
    %  4.Cantilever beam design                %
    %  5.Gear train design                     %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % BestX:The best solution                  %
    % BestF:The best fitness                   %
    % HisBestF:History of the best fitness     %
    % EngIndex：Index of engineerings          %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    clc;
    clear;
    EngList=[{'Pressure vessel design'},{'Rolling element bearing design'},...
        {'Tension/compression spring design'},{'Cantilever beam design'},...
        {'Gear train design'}];
    MaxIteration=1000;
    PopSize=50;
    
  EngIndex=1; %Index of engineerings can be chanaged according to engineering lists.
    
    [BestX,BestF,HisBestF]=ARO(EngIndex,MaxIteration,PopSize);
    if EngIndex==2 || EngIndex==3
        BestX(3)=round(BestX(3));
    elseif  EngIndex==5
        BestX=round(BestX);
    end
    
    display(['The best fitness of ',EngList{EngIndex},' is: ', num2str(BestF)]);
    display(['The best solution is: ', regexprep(num2str(BestX),'\s*',',')]);
    figure;
    if BestF>0
        semilogy(HisBestF,'r','LineWidth',2);
    else
        plot(HisBestF,'r','LineWidth',2);
    end
    
    xlabel('Iterations');
    ylabel('Fitness');
    title([EngList{EngIndex}]);

    
    
