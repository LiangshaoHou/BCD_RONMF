function [gap,Z]=OptGap1(A,U,V)
% This gap for EM-ONMF and BCD-ONMF with a preprocessing of zero columns.
Z=0;
NV=sqrt(sum(V.*V,2));
Ind=find(NV==0);
if ~isempty(Ind)
    U(:,Ind)=[];
    V(Ind,:)=[];
    NV(Ind)=[];
    Z=1;
end

%
U=U*diag(NV);
V=diag(NV)^(-1)*V;


tmp1=min(U,-2*A*V'+2*U*(V*V'));
% N1=max(sum(abs(tmp1)));
N1=norm(tmp1,1);

tmp2=-2*U'*A+2*U'*U*V;
N2=sum(abs(tmp2(V>0)));

K=find(sum(abs(V))==0);
N3=sum(sum(abs(min(0,tmp2(:,K)))));

gap=N1+N2+N3;


end