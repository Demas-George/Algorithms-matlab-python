close all;
clc;clear;
rng(100)

format long;   
% parameter setting
fun_nums=12;     % CEC-2022 includes 12 functions
runs=10;         % run times
iter_max=1000;   % max iterationn
pop_size=100;    % population size
dim=20;          % dimension
Xmin=-100;       % lower bound
Xmax=100;        % upper bound

% xbest 
xbest(runs,dim) = inf;
xbest(:) = inf;
% fbest 
fbest(fun_nums,runs) = inf;
fbest(:) = inf;
% f_mean 
f_mean(fun_nums) = inf;
f_mean(:) = inf;
% f_median 
f_median(fun_nums) = inf;
f_median(:) = inf;
% f_std 
f_std(fun_nums) = inf;
f_std(:) = inf;
% f_time
f_time(fun_nums,runs) = inf;
f_time(:) = inf;

targetbest = [300;400;600;800;900;1800;2000;2200;2300;2400;2600;2700]; % values refer to CEC2022

fhd=str2func('cec22_test_func'); 
fname = ['APO_',num2str(dim),'D.txt'];
f_out_name = fopen(fname,'wt');
fname_b_m_std_time = ['record_APO_b_m_std_time_',num2str(dim),'D.txt'];% record best, mean,std and time (runs)
f_out_b_m_std_time = fopen(fname_b_m_std_time,'wt');
ftime = ['APO_Time_',num2str(dim),'D.txt'];
f_out_time = fopen(ftime,'wt');

for i=1:fun_nums
    fun_num=i;
    disp(['Fid:',num2str(fun_num)]);
    fprintf(f_out_name,'Fid:%d\n',fun_num);
    fprintf(f_out_time,'Fid:%d\n',fun_num);
    for j=1:runs
    [gbest,gbestval,recordtime] = APO_func(fhd,dim,pop_size,iter_max,Xmin,Xmax,fun_num,j);
        xbest(j,:)=gbest;   
        fbest(i,j)=gbestval-targetbest(i); % tranfer to the same fitness value 0
        disp(['x[',num2str(gbest),']=',num2str(gbestval-targetbest(i),15)]);
        fprintf(f_out_name,'x%s\t[%s]=%s\n',num2str(j),num2str(gbest),num2str(gbestval-targetbest(i)));
        f_time(i,j) = recordtime;
        fprintf(f_out_time,'Time[%s]=\t%.15f\n',num2str(j),f_time(i,j));
    end
    [bestval,bestindex] = min(fbest(i,:));
    locb = xbest(bestindex,:); % best solution in runs
    disp(['Best[',num2str(locb),']=',num2str(bestval,15)]);
    fprintf(f_out_name,'Best[%s]=%s\n',num2str(locb),num2str(bestval));
    [worstval,worstindex] = max(fbest(i,:)); 
    locw = xbest(worstindex,:); % worst solution in runs
    disp(['Worst[',num2str(locw),']=',num2str(worstval,15)]);
    fprintf(f_out_name,'Worst[%s]=%s\n',num2str(locw),num2str(worstval));
    f_mean(i)=mean(fbest(i,:));
    f_median(i) = median(fbest(i,:));
    f_std(i) = std(fbest(i,:));
    disp(['mean[',num2str(i),']=',num2str(f_mean(i),15)]);
    fprintf(f_out_name,'mean[%s]=%s\n',num2str(i),num2str(f_mean(i)));
    disp(['median[',num2str(i),']=',num2str(f_median(i),15)]);
    fprintf(f_out_name,'median[%s]=%s\n',num2str(i),num2str(f_median(i)));
    disp(['std[',num2str(i),']=',num2str(f_std(i),15)]);
    fprintf(f_out_name,'std[%s]=%s\n',num2str(i),num2str(f_std(i)));
    
    MeanT=mean(f_time(i,:));
    disp(['MeanTime[',num2str(i),']=',num2str(MeanT,15)]);
    fprintf(f_out_time,'MeanTime[%s]=\t%.15f\n',num2str(i),MeanT);
    
    best_mean_std(i,1)=i; 
    best_mean_std(i,2)=bestval;
    best_mean_std(i,3)=f_mean(i);
    best_mean_std(i,4)=f_std(i);
    best_mean_std(i,5)=MeanT;
    best_mean_std(i,6)=bestindex;
    fprintf(f_out_b_m_std_time,'%s\n',num2str(best_mean_std(i,:)));
end
fclose(f_out_name);
fclose(f_out_time);
fclose(f_out_b_m_std_time);
clear all;