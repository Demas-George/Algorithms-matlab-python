function [BestX,BestF,HisBestF]=ARO_func(nPop,MaxIt,Low,Up,Dim,fobj)
if isscalar(Low), Low = Low * ones(1, Dim); end
if isscalar(Up), Up = Up * ones(1, Dim); end
PopPos=zeros(nPop,Dim);
PopFit=zeros(nPop,1);

for i=1:nPop
    PopPos(i,:)=rand(1,Dim).*(Up-Low)+Low;
    PopFit(i)=fobj(PopPos(i,:));
end

BestF=inf;
BestX=[];

for i=1:nPop
    if PopFit(i)<=BestF
        BestF=PopFit(i);
        BestX=PopPos(i,:);
    end
end

HisBestF=zeros(MaxIt,1);

for It=1:MaxIt
    Direct1=zeros(nPop,Dim);
    Direct2=zeros(nPop,Dim);
    theta=2*(1-It/MaxIt);
    for i=1:nPop
        L=(exp(1)-exp(((It-1)/MaxIt)^2))*(sin(2*pi*rand));
        rd=ceil(rand*(Dim));
        Direct1(i,randperm(Dim,rd))=1;
        c=randperm(Dim,rd);
        Direct2(i,c(randi(rd)))=1;
        R=L.*Direct1(i,:);
        
        A=2*log(1/rand)*theta;
        
        if A>1
            K=[1:i-1 i+1:nPop];
            RandInd=K(randi(nPop-1));
            newPopPos=PopPos(RandInd,:)+R.*(PopPos(i,:)-PopPos(RandInd,:))...
                +round(0.5*(0.05+rand))*randn;
        else
            Direct2(i,ceil(rand*Dim))=1;
            Gr=Direct2(i,:);
            H=((MaxIt-It+1)/MaxIt)*randn;
            b=PopPos(i,:)+H*Gr.*PopPos(i,:);
            newPopPos=PopPos(i,:)+ R.*(rand*b-PopPos(i,:));
        end
        newPopPos=SpaceBound(newPopPos,Up,Low);
        newPopFit=fobj(newPopPos);
        if newPopFit<PopFit(i)
            PopFit(i)=newPopFit;
            PopPos(i,:)=newPopPos;
        end
        
        if PopFit(i)<BestF
            BestF=PopFit(i);
            BestX=PopPos(i,:);
        end
    end
    HisBestF(It)=BestF;
end
end
