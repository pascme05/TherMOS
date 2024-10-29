% function dydt = odefnc(t, y, dt, F, GC)
%     dydt = zeros(size(GC));
%     for i = 1:size(GC, 1)
%         for ii = 1:size(GC, 2)
%             dydt(i, ii) = (F(ii, ceil(t / dt)) - y(i)*GC(i, ii));
%         end
%     end
%     dydt = sum(dydt, 1)';
% end

% function dy = odefnc(t,y,B,q,qt,N)
%     for i = 1:N
%         q1(i) = interp1(qt,q(:,i),t);
%         for j = 1:N
%             c(i,j) = B(i,j)*y(j);
%         end
%         dy = q1(i) - sum(c,2);
%     end
% end

function dydt = odefnc(t, y, GC, q, qt, ~)
    q1 = interp1(qt,q,t);
    dydt = q1 - y'*GC;
    dydt = dydt';
end

% function dydt = odefnc(t, y, GC, q, qt, N)
%     dydt = zeros(N, N);
%     for i = 1:N
%         for ii = 1:N
%             q1 = interp1(qt ,q(:, ii), t);
%             dydt(i, ii) = (q1 - y(i)*GC(i, ii));
%         end
%     end
%     dydt = sum(dydt, 1)';
% end