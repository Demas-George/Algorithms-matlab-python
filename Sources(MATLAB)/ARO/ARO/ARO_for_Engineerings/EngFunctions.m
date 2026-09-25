
function Fit=EngFunctions(x,FunIndex)




switch FunIndex
    
        %%%%%%%%%%%%%%%%%%%%%%%%%Pressure vessel design%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    case 1
        lambda=10^20;
        g(1)=-x(1)+0.0193*x(3);
        g(2)=-x(2)+0.00954*x(3);
        g(3)= -pi*x(3)^2*x(4)-(4/3)*pi*x(3)^3+1296000;
        g(4)= x(4)-240;
        f=0.6224*x(1)*x(3)*x(4)+1.7781*x(2)*x(3)^2+3.1661*x(1)^2*x(4)+19.84*x(1)^2*x(3);
        Penalty=0;
        for i=1:length(g)
            Penalty= Penalty+ lambda*g(i)^2*GetInequality(g(i));
        end
        Fit=f+Penalty;
        %%%%%%%%%%%%%%%%%%%%%%Rolling element bearing design%%%%%%%%%%%%%%%%%%%%%%%%%%%
    case 2
        x(3)=round(x(3));
        lambda=10^20;
        D=160;
        d=90;
        Bw=30;
        T=D-d-2*x(2);
        phio=2*pi-2*acos(((((D-d)/2)-3*(T/4))^2+(D/2-T/4-x(2))^2-(d/2+T/4)^2)...
            /(2*((D-d)/2-3*(T/4))*(D/2-T/4-x(2))));
        g(1)=-phio/(2*asin(x(2)/x(1)))+x(3)-1;
        g(2)=-2*x(2)+x(6)*(D-d);
        g(3)= -x(7)*(D-d)+2*x(2);
        g(4)=(0.5-x(9))*(D+d)-x(1);
        g(5)=-(0.5+x(9))*(D+d)+x(1);
        g(6)=-x(1)+0.5*(D+d);
        g(7)= -0.5*(D-x(1)-x(2))+x(8)*x(2);
        g(8)=x(10)*Bw-x(2);
        g(9)=0.515-x(4);
        g(10)=0.515-x(5);
        Penalty=0;
        for i=1:length(g)
            Penalty= Penalty+ lambda*g(i)^2*GetInequality(g(i));
        end
        gama=x(2)/x(1);
        fc=37.91*((1+(1.04*((1-gama/1+gama)^1.72)*((x(4)*(2*x(5)-1)/x(5)*...
            (2*x(4)-1))^0.41))^(10/3))^-0.3)*((gama^0.3*(1-gama)^1.39)/...
            (1+gama)^(1/3))*(2*x(4)/(2*x(4)-1))^0.41;
        if x(2)<=25.4
            f=-fc*x(3)^(2/3)*x(2)^1.8;
        else
            f=-3.647*fc*x(3)^(2/3)*x(2)^1.4;
        end
        Fit=f+Penalty;
        %%%%%%%%%%%%%%%%%%%%Tension/compression spring design%%%%%%%%%%%%%%%%%
    case 3
        x(3)=round(x(3));
        lambda=10^20;
        g(1)=1-x(3)*x(2)^3/(71785*x(1)^4);
        g(2)=(4*x(2)^2-x(1)*x(2))/(12566*(x(2)*x(1)^3-x(1)^4))+1/(5108*x(1)^2)-1;
        g(3)=1-140.45*x(1)/(x(2)^2*x(3));
        g(4)=(x(1)+x(2))/1.5-1;
        f=(2+x(3))*x(1)^2*x(2);
        Penalty=0;
        for i=1:length(g)
            Penalty= Penalty+ lambda*g(i)^2*GetInequality(g(i));
        end
        Fit=f+Penalty;
        %%%%%%%%%%%%%%%%%%%%%%%%%Cantilever beam design%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    case 4
        lambda=10^20;
        g(1)=61/x(1)^3+37/x(2)^3+19/x(3)^3+7/x(4)^3+1/x(5)^3-1;
        f=0.0624*(x(1)+x(2)+x(3)+x(4)+x(5));
        Penalty=0;
        for i=1:length(g)
            Penalty= Penalty+ lambda*g(i)^2*GetInequality(g(i));
        end
        Fit=f+Penalty;        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%Gear train design%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    otherwise
        x=round(x);
        Fit=(1/6.931-(x(3)*x(2)/(x(1)*x(4))))^2;
end

    function R=GetInequality(equation)
        if equation<=0
            R=0;
        else
            R=1;
        end
    end
end

