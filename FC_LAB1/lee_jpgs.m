clear
dirname='./retratos/';
lista = dir([dirname 'ret*']); L = length(lista);
fprintf('Encontradas %d imagenes en el directorio\n',L);
fprintf('Leyendo:\n'),

N=288; M=192;  imags=uint8(zeros(N,M,L)); 
for k=1:L  
  org = [dirname lista(k).name]; 
  im=imread(org);  
  if (size(im,3)>1), im=rgb2gray(im); end 
  imags(:,:,k)=im;   
  fprintf('%3d ',k); if mod(k,20)==0, fprintf('\n'); end
end
imags=im2double(imags);

