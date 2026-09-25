
 
function [Low,Up,Dim]=EngRange(FunIndex)
 
    
    switch FunIndex
        
        case 1
          
            Low=[0 0 10 10];
            Up=[99 99 200 200];
            Dim=length(Low);
            
        case 2
            D=160;
            d=90;
            Low=[0.5*(D+d) 0.15*(D-d) 4 0.515 0.515 0.4 0.6 0.3 0.02 0.6];
            Up=[0.6*(D+d) 0.45*(D-d) 50 0.6 0.6 0.5 0.7 0.4 0.1 0.85];
            Dim=length(Low);
            
        case 3
            Low=[0.05 0.25 2];
            Up=[2.0  1.3  15];
            Dim=length(Low);
            
            
        case 4
            Low=0.01;Up=100;
            Dim=5;
            
                    
        otherwise
             Low=[12 12 12 12];
            Up=[60 60 60 60];
            Dim=length(Low);
    
    end

