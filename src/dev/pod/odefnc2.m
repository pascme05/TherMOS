% function f = odefnc2(t, u, Gth, q, qt)
%     q1 = interp1(qt,q,t);
%     f = -Gth*u + q1';
% end
function f = odefnc2(t, u, Gth, q, qt)
    if t < qt(1)
        q1 = q(1); % Use first value if t is before qt range
    elseif t > qt(end)
        q1 = q(end); % Use last value if t is after qt range
    else
        q1 = interp1(qt, q, t); % Interpolate within range
    end
    f = -Gth * u + q1'; % Main equation
end


