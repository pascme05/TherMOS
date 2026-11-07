function smoothed = smooth_internal(matrix)
    % Copy original
    smoothed = padarray(matrix,[1 1],'replicate','both');

    % Define convolution kernel (2x2 averaging)
    kernel = ones(2) / 4;

    % Apply convolution only to internal region
    core = conv2(smoothed, kernel, 'same');

    % Replace only internal part with smoothed version
    smoothed = core(2:end-1, 2:end-1);
end
