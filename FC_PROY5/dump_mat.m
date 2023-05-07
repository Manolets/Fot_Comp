function dump_mat(A)
  for k=1:size(A,1)
    fprintf('%9.4f ',A(k,:)); fprintf('\n');  
  end
end