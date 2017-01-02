function [Priors,Mu,Sigma]=EM_init_kmeans(Data,nbStates)
[nbVar,nbData]=size(Data);
[Data_id,Centers]=kmeans(Data',nbStates);
Mu=Centers;
for i=1:nbStates
    idtmp=find(Data_id==i);
    Priors(i)=length(idtmp);
    Sigma(:,:,i)=cov([Data(:,idtmp) Data(:,idtmp)]');
    Sigma(:,:,i)=Sigma(:,:,i)+1E-5.*diag(ones(nbVar,1));%Add a tiny variance to avoid numberical instability
    Sigma(:,:,i)=diag(diag(Sigma(:,:,i)));%将向量放到相应的主对角线上
end
Priors=Priors./sum(Priors);