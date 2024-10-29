function f = odefnc2(t, u, Gth, q, qt)
    q1 = interp1(qt,q,t);
    f = -Gth*u + q1';
end

