function comprehensive_model_comparison_analyzer
% comprehensive_model_comparison_analyzer - 综合模型对比分析工具
% 用于详细对比原始数据模型和迭代后生成的模型之间的空间与形态特征匹配度
% 主要功能：
% 1. 加载两个二值化raw模型文件
% 2. 计算全面的空间和形态特征
% 3. 生成详细的对比分析报告
% 4. 可视化匹配度评估结果
%% 1. 用户界面和文件加载
fprintf('\n===== 综合模型对比分析工具 =====\n');
fprintf('本程序用于评估迭代后模型与原始模型的匹配度\n\n');
% 获取原始模型文件
fprintf('请选择原始数据模型文件（二值化.raw文件）：\n');
[origFile, origPath] = uigetfile('*.raw', '选择原始模型文件');
if isequal(origFile, 0)
    error('未选择原始模型文件');
end
origFullPath = fullfile(origPath, origFile);
% 获取迭代后模型文件
fprintf('请选择迭代后的模型文件（二值化.raw文件）：\n');
[iterFile, iterPath] = uigetfile('*.raw', '选择迭代后模型文件');
if isequal(iterFile, 0)
    error('未选择迭代后模型文件');
end
iterFullPath = fullfile(iterPath, iterFile);
% 获取模型尺寸
fprintf('\n请输入模型尺寸：\n');
dims(1) = input('  X维度: ');
dims(2) = input('  Y维度: ');
dims(3) = input('  Z维度: ');
% 验证输入
if any(dims <= 0)
    error('无效的模型尺寸');
end
fprintf('\n正在加载模型文件...\n');
% 加载模型
origModel = loadBinaryModel(origFullPath, dims);
iterModel = loadBinaryModel(iterFullPath, dims);
fprintf('模型加载完成\n');
fprintf('  原始模型: %s\n', origFile);
fprintf('  迭代模型: %s\n', iterFile);
fprintf('  模型尺寸: %d × %d × %d\n', dims(1), dims(2), dims(3));
%% 2. 基本统计信息
fprintf('\n===== 基本统计信息 =====\n');
origStats = computeBasicStats(origModel);
iterStats = computeBasicStats(iterModel);
fprintf('\n原始模型:\n');
displayBasicStats(origStats);
fprintf('\n迭代后模型:\n');
displayBasicStats(iterStats);
%% 3. 计算综合特征
fprintf('\n===== 计算综合特征 =====\n');
% 计算空间特征
fprintf('计算空间特征...\n');
origSpatial = computeCompleteSpatialFeatures(origModel);
iterSpatial = computeCompleteSpatialFeatures(iterModel);
% 计算形态特征
fprintf('计算形态特征...\n');
origMorph = computeCompleteMorphologyFeatures(origModel);
iterMorph = computeCompleteMorphologyFeatures(iterModel);
% 计算簇特征
fprintf('计算簇分布特征...\n');
origCluster = computeClusterFeatures(origModel);
iterCluster = computeClusterFeatures(iterModel);
% 计算多尺度特征
fprintf('计算多尺度特征...\n');
origMultiScale = computeMultiScaleFeatures(origModel);
iterMultiScale = computeMultiScaleFeatures(iterModel);
%% 4. 匹配度评估
fprintf('\n===== 匹配度评估 =====\n');
matchingResults = evaluateMatching(origSpatial, iterSpatial, ...
    origMorph, iterMorph, origCluster, iterCluster, ...
    origMultiScale, iterMultiScale);
displayMatchingResults(matchingResults);
%% 5. 生成可视化报告
fprintf('\n===== 生成可视化报告 =====\n');
generateComprehensiveReport(origModel, iterModel, ...
    origSpatial, iterSpatial, origMorph, iterMorph, ...
    origCluster, iterCluster, origMultiScale, iterMultiScale, ...
    matchingResults);
% 保存分析结果
saveResults(matchingResults, origStats, iterStats);
fprintf('\n分析完成！\n');
end
%% ========== 文件I/O函数 ==========
function model = loadBinaryModel(filepath, dims)
% 加载二值化模型
fid = fopen(filepath, 'rb');
if fid == -1
    error('无法打开文件 %s', filepath);
end
% 读取数据
data = fread(fid, prod(dims), 'uint8');
fclose(fid);
% 重塑为3D数组
model = reshape(data, dims);
% 转换为二值模型
model = logical(model);
end
%% ========== 基本统计函数 ==========
function stats = computeBasicStats(model)
% 计算基本统计信息
stats = struct();
stats.porosity = mean(model(:));
stats.solidFraction = 1 - stats.porosity;
stats.totalPoreVoxels = sum(model(:));
stats.totalSolidVoxels = sum(~model(:));
stats.modelSize = size(model);
% 连通组分分析
CC = bwconncomp(model, 26);
stats.numClusters = CC.NumObjects;
if CC.NumObjects > 0
    stats.clusterSizes = cellfun(@numel, CC.PixelIdxList);
    stats.minClusterSize = min(stats.clusterSizes);
    stats.maxClusterSize = max(stats.clusterSizes);
    stats.meanClusterSize = mean(stats.clusterSizes);
    stats.stdClusterSize = std(stats.clusterSizes);
else
    stats.clusterSizes = [];
    stats.minClusterSize = 0;
    stats.maxClusterSize = 0;
    stats.meanClusterSize = 0;
    stats.stdClusterSize = 0;
end
% 表面积估算
boundary = bwperim(model, 26);
stats.surfaceArea = sum(boundary(:));
stats.surfaceToVolumeRatio = stats.surfaceArea / (stats.totalPoreVoxels + eps);
end
function displayBasicStats(stats)
% 显示基本统计信息
fprintf('  孔隙率: %.4f\n', stats.porosity);
fprintf('  孔隙体素数: %d\n', stats.totalPoreVoxels);
fprintf('  簇数量: %d\n', stats.numClusters);
if stats.numClusters > 0
    fprintf('  簇大小: 最小=%d, 最大=%d, 平均=%.1f, 标准差=%.1f\n', ...
        stats.minClusterSize, stats.maxClusterSize, ...
        stats.meanClusterSize, stats.stdClusterSize);
end
fprintf('  表面积/体积比: %.4f\n', stats.surfaceToVolumeRatio);
end
%% ========== 空间特征计算 ==========
function spatial = computeCompleteSpatialFeatures(model)
% 计算完整的空间特征
spatial = struct();
% 1. 两点相关函数
fprintf('  计算两点相关函数...\n');
keyDistances = [1, 3, 5, 10, 15, 20, 30, 40, 50, 70];
spatial.twoPointCorr = computeTwoPointCorrelation(model, keyDistances);
spatial.keyDistances = keyDistances;
% 2. 各向异性
fprintf('  计算各向异性...\n');
spatial.anisotropy = computeAnisotropy(model);
% 3. 孔隙率梯度
fprintf('  计算孔隙率梯度...\n');
spatial.porosityGradient = computePorosityGradient(model);
% 4. 连通性指标
fprintf('  计算连通性...\n');
spatial.connectivity = computeConnectivityMetrics(model);
% 5. 迂曲度
fprintf('  计算迂曲度...\n');
spatial.tortuosity = estimateTortuosity(model);
% 6. 空间自相关
fprintf('  计算空间自相关...\n');
spatial.spatialAutocorrelation = computeSpatialAutocorrelation(model);
% 7. 径向分布函数
fprintf('  计算径向分布函数...\n');
spatial.radialDistribution = computeRadialDistribution(model);
end
function tpc = computeTwoPointCorrelation(model, distances)
% 计算两点相关函数
nDistances = length(distances);
tpc = zeros(nDistances, 3); % 三个方向
for i = 1:nDistances
    lag = distances(i);
    
    % 采样以加速计算
    sampleStep = max(1, round(lag / 5));
    sampledModel = model(1:sampleStep:end, 1:sampleStep:end, 1:sampleStep:end);
    
    for dir = 1:3
        shiftVec = [0, 0, 0];
        shiftVec(dir) = min(lag, size(sampledModel, dir) - 1);
        shifted = circshift(sampledModel, shiftVec);
        tpc(i, dir) = mean(sampledModel(:) .* shifted(:));
    end
end
end
function aniso = computeAnisotropy(model)
% 计算各向异性
% 投影面积方法
projXY = sum(model, 3);
projXZ = squeeze(sum(model, 2));
projYZ = squeeze(sum(model, 1));
areaXY = sum(projXY(:) > 0);
areaXZ = sum(projXZ(:) > 0);
areaYZ = sum(projYZ(:) > 0);
areas = [areaXY, areaXZ, areaYZ];
aniso = std(areas) / (mean(areas) + eps);
end
function grad = computePorosityGradient(model)
% 计算孔隙率梯度
windowSize = 10;
stride = 5;
[nx, ny, nz] = size(model);
% 计算局部孔隙率
nWindowsX = floor((nx - windowSize) / stride) + 1;
nWindowsY = floor((ny - windowSize) / stride) + 1;
nWindowsZ = floor((nz - windowSize) / stride) + 1;
localPorosity = zeros(nWindowsX, nWindowsY, nWindowsZ);
for i = 1:nWindowsX
    for j = 1:nWindowsY
        for k = 1:nWindowsZ
            x1 = (i-1)*stride + 1;
            y1 = (j-1)*stride + 1;
            z1 = (k-1)*stride + 1;
            x2 = min(x1+windowSize-1, nx);
            y2 = min(y1+windowSize-1, ny);
            z2 = min(z1+windowSize-1, nz);
            
            window = model(x1:x2, y1:y2, z1:z2);
            localPorosity(i,j,k) = mean(window(:));
        end
    end
end
% 计算梯度
[gx, gy, gz] = gradient(localPorosity);
grad = mean(sqrt(gx(:).^2 + gy(:).^2 + gz(:).^2));
end
function connectivity = computeConnectivityMetrics(model)
% 计算连通性指标
connectivity = struct();
% 连通组分分析
CC = bwconncomp(model, 26);
connectivity.numComponents = CC.NumObjects;
if CC.NumObjects > 0
    sizes = cellfun(@numel, CC.PixelIdxList);
    connectivity.largestComponentRatio = max(sizes) / sum(sizes);
    connectivity.connectivityDensity = CC.NumObjects / numel(model);
else
    connectivity.largestComponentRatio = 0;
    connectivity.connectivityDensity = 0;
end
% 欧拉数（使用切片平均）
[nx, ny, nz] = size(model);
eulerNumbers = zeros(nz, 1);
for z = 1:nz
    eulerNumbers(z) = bweuler(model(:,:,z), 8);
end
connectivity.eulerNumber = mean(eulerNumbers);
% 孔隙网络连接度
connectivity.coordination = computeCoordinationNumber(model);
end
function coord = computeCoordinationNumber(model)
% 计算平均配位数
CC = bwconncomp(model, 26);
if CC.NumObjects <= 1
    coord = 0;
    return;
end
% 计算簇之间的接触
nAnalyze = min(20, CC.NumObjects);
coordinationNumbers = zeros(nAnalyze, 1);
for i = 1:nAnalyze
    clusterMask = false(size(model));
    clusterMask(CC.PixelIdxList{i}) = true;
    
    % 膨胀找到相邻簇
    dilated = imdilate(clusterMask, ones(3,3,3));
    contactingClusters = 0;
    
    for j = 1:CC.NumObjects
        if j ~= i && any(dilated(CC.PixelIdxList{j}))
            contactingClusters = contactingClusters + 1;
        end
    end
    coordinationNumbers(i) = contactingClusters;
end
coord = mean(coordinationNumbers);
end
function tort = estimateTortuosity(model)
% 估计迂曲度
[nx, ny, nz] = size(model);
tort = 0;
validPaths = 0;
% 采样路径
nSamples = min(20, nx*ny/100);
for s = 1:nSamples
    startX = randi(nx);
    startY = randi(ny);
    
    % 检查Z方向的连通性
    if model(startX, startY, 1) && model(startX, startY, nz)
        % 简化：使用直线距离比
        pathLength = nz;
        straightDistance = nz - 1;
        if straightDistance > 0
            tort = tort + pathLength / straightDistance;
            validPaths = validPaths + 1;
        end
    end
end
if validPaths > 0
    tort = tort / validPaths;
else
    tort = 1.5; % 默认值
end
tort = max(1, min(tort, 3));
end
function autocorr = computeSpatialAutocorrelation(model)
% 计算空间自相关
data = double(model);
meanVal = mean(data(:));
deviation = data - meanVal;
% 采样计算
nSamples = min(1000, numel(data));
sampleIdx = randperm(numel(data), nSamples);
sumNum = 0;
sumDenom = sum(deviation(:).^2);
nPairs = 0;
% 计算不同距离的相关性
distanceScales = [1, 3, 5, 10, 20];
for i = 1:nSamples
    [x, y, z] = ind2sub(size(data), sampleIdx(i));
    
    for d = 1:length(distanceScales)
        scale = distanceScales(d);
        % 6邻域
        neighbors = [x+scale, y, z; x-scale, y, z; ...
                    x, y+scale, z; x, y-scale, z; ...
                    x, y, z+scale; x, y, z-scale];
        
        for n = 1:size(neighbors, 1)
            nx = neighbors(n,1); ny = neighbors(n,2); nz = neighbors(n,3);
            if nx >= 1 && nx <= size(data,1) && ...
               ny >= 1 && ny <= size(data,2) && ...
               nz >= 1 && nz <= size(data,3)
                sumNum = sumNum + deviation(x,y,z) * deviation(nx,ny,nz);
                nPairs = nPairs + 1;
            end
        end
    end
end
if nPairs > 0 && sumDenom > 0
    autocorr = (sumNum / nPairs) / (sumDenom / numel(data));
else
    autocorr = 0;
end
end
function rdf = computeRadialDistribution(model)
% 计算径向分布函数
% 获取孔隙中心点
CC = bwconncomp(model, 26);
if CC.NumObjects == 0
    rdf = struct('distances', 0, 'values', 0);
    return;
end
% 计算质心
nCentroids = min(100, CC.NumObjects); % 限制计算量
centroids = zeros(nCentroids, 3);
for i = 1:nCentroids
    [x, y, z] = ind2sub(size(model), CC.PixelIdxList{i});
    centroids(i, :) = [mean(x), mean(y), mean(z)];
end
% 计算距离分布
if nCentroids > 1
    distances = pdist(centroids);
    maxDist = max(distances);
    nBins = 20;
    edges = linspace(0, maxDist, nBins+1);
    [counts, ~] = histcounts(distances, edges);
    
    % 归一化
    binCenters = (edges(1:end-1) + edges(2:end)) / 2;
    shellVolumes = 4 * pi * binCenters.^2 .* diff(edges);
    rdf.distances = binCenters;
    rdf.values = counts ./ (shellVolumes + eps);
else
    rdf.distances = 0;
    rdf.values = 0;
end
end
%% ========== 形态特征计算 ==========
function morph = computeCompleteMorphologyFeatures(model)
% 计算完整的形态特征
morph = struct();
CC = bwconncomp(model, 26);
if CC.NumObjects == 0
    morph = createEmptyMorphologyFeatures();
    return;
end
% 限制分析的簇数量
nAnalyze = min(CC.NumObjects, 50);
sizes = cellfun(@numel, CC.PixelIdxList);
[~, sortIdx] = sort(sizes, 'descend');
analyzeIdx = sortIdx(1:nAnalyze);
% 初始化特征数组
morph.sphericity = zeros(nAnalyze, 1);
morph.elongation = zeros(nAnalyze, 1);
morph.convexity = zeros(nAnalyze, 1);
morph.solidity = zeros(nAnalyze, 1);
morph.compactness = zeros(nAnalyze, 1);
fprintf('  分析形态特征（%d个簇）...\n', nAnalyze);
for i = 1:nAnalyze
    idx = analyzeIdx(i);
    [x, y, z] = ind2sub(size(model), CC.PixelIdxList{idx});
    
    if length(x) > 10  % 只对足够大的簇计算
        % 主成分分析
        coords = [x - mean(x), y - mean(y), z - mean(z)];
        try
            [V, D] = eig(cov(coords));
            eigenvalues = sort(diag(D), 'descend');
            
            % 伸长率
            if eigenvalues(3) > 0
                morph.elongation(i) = sqrt(eigenvalues(1) / eigenvalues(3));
            else
                morph.elongation(i) = 1;
            end
            
            % 球形度（使用椭球体近似）
            volume = length(x);
            a = sqrt(eigenvalues(1));
            b = sqrt(eigenvalues(2));
            c = sqrt(eigenvalues(3));
            
            if a > 0 && b > 0 && c > 0
                ellipsoidVolume = (4/3) * pi * a * b * c;
                morph.sphericity(i) = (pi^(1/3) * (6*volume)^(2/3)) / ...
                    (ellipsoidVolume^(2/3));
                morph.sphericity(i) = min(1, morph.sphericity(i));
            else
                morph.sphericity(i) = 0.5;
            end
            
            % 紧凑度和凸性
            rangeX = max(x) - min(x) + 1;
            rangeY = max(y) - min(y) + 1;
            rangeZ = max(z) - min(z) + 1;
            boundingBoxVolume = rangeX * rangeY * rangeZ;
            
            morph.convexity(i) = volume / boundingBoxVolume;
            morph.compactness(i) = volume / (rangeX * rangeY * rangeZ);
            morph.solidity(i) = morph.convexity(i);
            
        catch
            % 如果PCA失败，使用默认值
            morph.elongation(i) = 1;
            morph.sphericity(i) = 0.5;
            morph.convexity(i) = 0.5;
            morph.solidity(i) = 0.5;
            morph.compactness(i) = 0.5;
        end
    else
        % 小簇使用默认值
        morph.elongation(i) = 1;
        morph.sphericity(i) = 0.8;
        morph.convexity(i) = 0.8;
        morph.solidity(i) = 0.8;
        morph.compactness(i) = 0.8;
    end
end
% 计算其他形态特征
morph.poreNetworkDensity = computePoreNetworkDensity(model);
morph.lacunarity = computeLacunarity(model);
morph.fractalDimension = computeFractalDimension(model);
end
function morph = createEmptyMorphologyFeatures()
% 创建空的形态特征结构
morph = struct();
morph.sphericity = [];
morph.elongation = [];
morph.convexity = [];
morph.solidity = [];
morph.compactness = [];
morph.poreNetworkDensity = 0;
morph.lacunarity = 0;
morph.fractalDimension = 2.5;
end
function density = computePoreNetworkDensity(model)
% 计算孔隙网络密度
CC = bwconncomp(model, 26);
if CC.NumObjects == 0
    density = 0;
    return;
end
% 使用簇数量与总体积的比值
density = CC.NumObjects / sum(model(:));
end
function lac = computeLacunarity(model)
% 计算空隙率
boxSizes = [5, 10, 15, 20];
lacValues = zeros(length(boxSizes), 1);
for b = 1:length(boxSizes)
    boxSize = boxSizes(b);
    
    % 采样计算
    nSamples = 50;
    masses = zeros(nSamples, 1);
    
    for s = 1:nSamples
        % 随机选择盒子位置
        x = randi([1, max(1, size(model,1)-boxSize+1)]);
        y = randi([1, max(1, size(model,2)-boxSize+1)]);
        z = randi([1, max(1, size(model,3)-boxSize+1)]);
        
        % 计算盒子内的质量
        box = model(x:min(x+boxSize-1, end), ...
                   y:min(y+boxSize-1, end), ...
                   z:min(z+boxSize-1, end));
        masses(s) = sum(box(:));
    end
    
    % 计算空隙率
    if mean(masses) > 0
        lacValues(b) = var(masses) / mean(masses)^2 + 1;
    else
        lacValues(b) = 1;
    end
end
lac = mean(lacValues);
end
function fd = computeFractalDimension(model)
% 计算分形维数（盒计数法）
boxSizes = [2, 4, 8, 16, 32];
counts = zeros(length(boxSizes), 1);
for i = 1:length(boxSizes)
    boxSize = boxSizes(i);
    count = 0;
    
    for x = 1:boxSize:size(model, 1)
        for y = 1:boxSize:size(model, 2)
            for z = 1:boxSize:size(model, 3)
                box = model(x:min(x+boxSize-1, end), ...
                           y:min(y+boxSize-1, end), ...
                           z:min(z+boxSize-1, end));
                if any(box(:))
                    count = count + 1;
                end
            end
        end
    end
    counts(i) = count;
end
% 线性拟合
validIdx = counts > 0;
if sum(validIdx) > 1
    p = polyfit(log(1./boxSizes(validIdx)), log(counts(validIdx)), 1);
    fd = p(1);
else
    fd = 2.5; % 默认值
end
fd = max(2, min(3, fd));
end
%% ========== 簇特征计算 ==========
function cluster = computeClusterFeatures(model)
% 计算簇分布特征
cluster = struct();
CC = bwconncomp(model, 26);
cluster.numClusters = CC.NumObjects;
if CC.NumObjects == 0
    cluster.sizes = [];
    cluster.sizeDistribution = [];
    cluster.centroids = [];
    cluster.spatialDispersion = 0;
    return;
end
% 基本统计
cluster.sizes = cellfun(@numel, CC.PixelIdxList);
cluster.minSize = min(cluster.sizes);
cluster.maxSize = max(cluster.sizes);
cluster.meanSize = mean(cluster.sizes);
cluster.stdSize = std(cluster.sizes);
% 大小分布
[counts, edges] = histcounts(log10(cluster.sizes + 1), 20);
cluster.sizeDistribution = struct('counts', counts, 'edges', edges);
% 空间分布
nCentroids = min(100, CC.NumObjects);
cluster.centroids = zeros(nCentroids, 3);
for i = 1:nCentroids
    [x, y, z] = ind2sub(size(model), CC.PixelIdxList{i});
    cluster.centroids(i, :) = [mean(x), mean(y), mean(z)];
end
% 空间离散度
if size(cluster.centroids, 1) > 1
    distances = pdist(cluster.centroids);
    cluster.spatialDispersion = std(distances) / (mean(distances) + eps);
else
    cluster.spatialDispersion = 0;
end
% 簇形状多样性
cluster.shapeDiversity = computeShapeDiversity(CC, model);
end
function diversity = computeShapeDiversity(CC, model)
% 计算簇形状多样性
nAnalyze = min(20, CC.NumObjects);
if nAnalyze == 0
    diversity = 0;
    return;
end
aspectRatios = zeros(nAnalyze, 1);
sizes = cellfun(@numel, CC.PixelIdxList);
[~, sortIdx] = sort(sizes, 'descend');
for i = 1:nAnalyze
    idx = sortIdx(i);
    [x, y, z] = ind2sub(size(model), CC.PixelIdxList{idx});
    
    rangeX = max(x) - min(x) + 1;
    rangeY = max(y) - min(y) + 1;
    rangeZ = max(z) - min(z) + 1;
    
    maxDim = max([rangeX, rangeY, rangeZ]);
    minDim = min([rangeX, rangeY, rangeZ]);
    
    if minDim > 0
        aspectRatios(i) = maxDim / minDim;
    else
        aspectRatios(i) = 1;
    end
end
diversity = std(aspectRatios) / (mean(aspectRatios) + eps);
end
%% ========== 多尺度特征计算 ==========
function multiScale = computeMultiScaleFeatures(model)
% 计算多尺度特征
multiScale = struct();
scales = [1, 2, 4, 8, 16];
maxScale = min(scales(end), min(size(model))/4);
scales = scales(scales <= maxScale);
fprintf('  计算多尺度特征（%d个尺度）...\n', length(scales));
for s = 1:length(scales)
    scale = scales(s);
    
    % 下采样
    if scale > 1
        scaledModel = model(1:scale:end, 1:scale:end, 1:scale:end);
    else
        scaledModel = model;
    end
    
    % 计算该尺度的特征
    multiScale.scale(s).factor = scale;
    multiScale.scale(s).porosity = mean(scaledModel(:));
    
    % 两点相关
    maxDist = min([10, size(scaledModel)/2]);
    dists = 1:maxDist;
    multiScale.scale(s).twoPointCorr = computeTwoPointCorrelation(scaledModel, dists);
    
    % 簇统计
    CC = bwconncomp(scaledModel, 26);
    multiScale.scale(s).numClusters = CC.NumObjects;
    if CC.NumObjects > 0
        sizes = cellfun(@numel, CC.PixelIdxList);
        multiScale.scale(s).meanClusterSize = mean(sizes);
    else
        multiScale.scale(s).meanClusterSize = 0;
    end
end
% 计算尺度不变性
multiScale.scaleInvariance = computeScaleInvariance(multiScale);
end
function invariance = computeScaleInvariance(multiScale)
% 计算尺度不变性
nScales = length(multiScale.scale);
if nScales < 2
    invariance = 0;
    return;
end
% 提取各尺度的孔隙率
porosities = zeros(nScales, 1);
for s = 1:nScales
    porosities(s) = multiScale.scale(s).porosity;
end
% 计算变异系数
invariance = 1 - std(porosities) / (mean(porosities) + eps);
invariance = max(0, min(1, invariance));
end
%% ========== 匹配度评估 ==========
function matching = evaluateMatching(origSpatial, iterSpatial, ...
    origMorph, iterMorph, origCluster, iterCluster, ...
    origMultiScale, iterMultiScale)
% 评估各方面的匹配度
matching = struct();
% 空间特征匹配
matching.spatial = evaluateSpatialMatching(origSpatial, iterSpatial);
% 形态特征匹配
matching.morphology = evaluateMorphologyMatching(origMorph, iterMorph);
% 簇分布匹配
matching.cluster = evaluateClusterMatching(origCluster, iterCluster);
% 多尺度特征匹配
matching.multiScale = evaluateMultiScaleMatching(origMultiScale, iterMultiScale);
% 总体匹配度
matching.overall = computeOverallMatching(matching);
% 详细差异分析
matching.differences = computeDetailedDifferences(origSpatial, iterSpatial, ...
    origMorph, iterMorph, origCluster, iterCluster);
end
function spatial = evaluateSpatialMatching(orig, iter)
% 评估空间特征匹配度
spatial = struct();
% 两点相关函数
if ~isempty(orig.twoPointCorr) && ~isempty(iter.twoPointCorr)
    minSize = min(size(orig.twoPointCorr, 1), size(iter.twoPointCorr, 1));
    origTPC = orig.twoPointCorr(1:minSize, :);
    iterTPC = iter.twoPointCorr(1:minSize, :);
    
    % 计算相关系数
    spatial.tpcCorrelation = corr(origTPC(:), iterTPC(:));
    
    % 计算均方误差
    spatial.tpcRMSE = sqrt(mean((origTPC(:) - iterTPC(:)).^2));
    
    % 归一化匹配度
    spatial.tpcMatch = max(0, 1 - spatial.tpcRMSE);
else
    spatial.tpcCorrelation = 0;
    spatial.tpcRMSE = 1;
    spatial.tpcMatch = 0;
end
% 各向异性
spatial.anisotropyDiff = abs(orig.anisotropy - iter.anisotropy);
spatial.anisotropyMatch = max(0, 1 - spatial.anisotropyDiff);
% 孔隙率梯度
spatial.gradientDiff = abs(orig.porosityGradient - iter.porosityGradient);
spatial.gradientMatch = max(0, 1 - spatial.gradientDiff / (max(orig.porosityGradient, iter.porosityGradient) + eps));
% 连通性
spatial.connectivityMatch = 1 - abs(orig.connectivity.largestComponentRatio - ...
    iter.connectivity.largestComponentRatio);
% 迂曲度
spatial.tortuosityDiff = abs(orig.tortuosity - iter.tortuosity);
spatial.tortuosityMatch = max(0, 1 - spatial.tortuosityDiff / 2);
% 空间自相关
spatial.autocorrDiff = abs(orig.spatialAutocorrelation - iter.spatialAutocorrelation);
spatial.autocorrMatch = max(0, 1 - spatial.autocorrDiff);
% 综合空间匹配度
spatial.overall = mean([spatial.tpcMatch, spatial.anisotropyMatch, ...
    spatial.gradientMatch, spatial.connectivityMatch, ...
    spatial.tortuosityMatch, spatial.autocorrMatch]);
end
function morph = evaluateMorphologyMatching(orig, iter)
% 评估形态特征匹配度
morph = struct();
% 球形度
if ~isempty(orig.sphericity) && ~isempty(iter.sphericity)
    morph.sphericityDiff = abs(mean(orig.sphericity) - mean(iter.sphericity));
    morph.sphericityMatch = max(0, 1 - morph.sphericityDiff);
else
    morph.sphericityDiff = 0;
    morph.sphericityMatch = 1;
end
% 伸长率
if ~isempty(orig.elongation) && ~isempty(iter.elongation)
    morph.elongationDiff = abs(mean(orig.elongation) - mean(iter.elongation));
    morph.elongationMatch = max(0, 1 - morph.elongationDiff / 3);
else
    morph.elongationDiff = 0;
    morph.elongationMatch = 1;
end
% 紧凑度
if ~isempty(orig.compactness) && ~isempty(iter.compactness)
    morph.compactnessDiff = abs(mean(orig.compactness) - mean(iter.compactness));
    morph.compactnessMatch = max(0, 1 - morph.compactnessDiff);
else
    morph.compactnessDiff = 0;
    morph.compactnessMatch = 1;
end
% 网络密度
morph.networkDensityDiff = abs(orig.poreNetworkDensity - iter.poreNetworkDensity);
morph.networkDensityMatch = max(0, 1 - morph.networkDensityDiff / ...
    (max(orig.poreNetworkDensity, iter.poreNetworkDensity) + eps));
% 空隙率
morph.lacunarityDiff = abs(orig.lacunarity - iter.lacunarity);
morph.lacunarityMatch = max(0, 1 - morph.lacunarityDiff / ...
    (max(orig.lacunarity, iter.lacunarity) + eps));
% 分形维数
morph.fractalDiff = abs(orig.fractalDimension - iter.fractalDimension);
morph.fractalMatch = max(0, 1 - morph.fractalDiff);
% 综合形态匹配度
morph.overall = mean([morph.sphericityMatch, morph.elongationMatch, ...
    morph.compactnessMatch, morph.networkDensityMatch, ...
    morph.lacunarityMatch, morph.fractalMatch]);
end
function cluster = evaluateClusterMatching(orig, iter)
% 评估簇分布匹配度
cluster = struct();
% 簇数量
cluster.numDiff = abs(orig.numClusters - iter.numClusters);
cluster.numMatch = max(0, 1 - cluster.numDiff / (max(orig.numClusters, iter.numClusters) + 1));
% 簇大小统计
if ~isempty(orig.sizes) && ~isempty(iter.sizes)
    cluster.meanSizeDiff = abs(orig.meanSize - iter.meanSize);
    cluster.meanSizeMatch = max(0, 1 - cluster.meanSizeDiff / (max(orig.meanSize, iter.meanSize) + eps));
    
    cluster.stdSizeDiff = abs(orig.stdSize - iter.stdSize);
    cluster.stdSizeMatch = max(0, 1 - cluster.stdSizeDiff / (max(orig.stdSize, iter.stdSize) + eps));
    
    % 大小分布相似度（使用KL散度）
    cluster.sizeDistKL = computeKLDivergence(orig.sizeDistribution, iter.sizeDistribution);
    cluster.sizeDistMatch = exp(-cluster.sizeDistKL);
else
    cluster.meanSizeMatch = 0;
    cluster.stdSizeMatch = 0;
    cluster.sizeDistMatch = 0;
end
% 空间离散度
cluster.dispersionDiff = abs(orig.spatialDispersion - iter.spatialDispersion);
cluster.dispersionMatch = max(0, 1 - cluster.dispersionDiff);
% 形状多样性
cluster.diversityDiff = abs(orig.shapeDiversity - iter.shapeDiversity);
cluster.diversityMatch = max(0, 1 - cluster.diversityDiff);
% 综合簇匹配度
cluster.overall = mean([cluster.numMatch, cluster.meanSizeMatch, ...
    cluster.stdSizeMatch, cluster.sizeDistMatch, ...
    cluster.dispersionMatch, cluster.diversityMatch]);
end
function kl = computeKLDivergence(dist1, dist2)
% 计算KL散度
if isempty(dist1.counts) || isempty(dist2.counts)
    kl = 1;
    return;
end
% 归一化
p = dist1.counts / sum(dist1.counts);
q = dist2.counts / sum(dist2.counts);
% 确保相同长度
minLen = min(length(p), length(q));
p = p(1:minLen);
q = q(1:minLen);
% 避免零值
p = p + eps;
q = q + eps;
% 计算KL散度
kl = sum(p .* log(p ./ q));
end
function multiScale = evaluateMultiScaleMatching(orig, iter)
% 评估多尺度特征匹配度
multiScale = struct();
nScales = min(length(orig.scale), length(iter.scale));
if nScales == 0
    multiScale.overall = 0;
    return;
end
scaleMatches = zeros(nScales, 1);
for s = 1:nScales
    % 孔隙率匹配
    porDiff = abs(orig.scale(s).porosity - iter.scale(s).porosity);
    porMatch = max(0, 1 - porDiff);
    
    % 簇数量匹配
    clusterDiff = abs(orig.scale(s).numClusters - iter.scale(s).numClusters);
    clusterMatch = max(0, 1 - clusterDiff / (max(orig.scale(s).numClusters, iter.scale(s).numClusters) + 1));
    
    scaleMatches(s) = 0.5 * porMatch + 0.5 * clusterMatch;
end
multiScale.scaleMatches = scaleMatches;
% 尺度不变性匹配
multiScale.invarianceDiff = abs(orig.scaleInvariance - iter.scaleInvariance);
multiScale.invarianceMatch = max(0, 1 - multiScale.invarianceDiff);
% 综合多尺度匹配度
multiScale.overall = 0.7 * mean(scaleMatches) + 0.3 * multiScale.invarianceMatch;
end
function overall = computeOverallMatching(matching)
% 计算总体匹配度
overall = struct();
% 各部分权重
weights.spatial = 0.3;
weights.morphology = 0.3;
weights.cluster = 0.2;
weights.multiScale = 0.2;
% 加权平均
overall.score = weights.spatial * matching.spatial.overall + ...
    weights.morphology * matching.morphology.overall + ...
    weights.cluster * matching.cluster.overall + ...
    weights.multiScale * matching.multiScale.overall;
% 评级
if overall.score >= 0.9
    overall.grade = 'A - 优秀匹配';
elseif overall.score >= 0.8
    overall.grade = 'B - 良好匹配';
elseif overall.score >= 0.7
    overall.grade = 'C - 中等匹配';
elseif overall.score >= 0.6
    overall.grade = 'D - 较差匹配';
else
    overall.grade = 'F - 匹配失败';
end
overall.weights = weights;
end
function differences = computeDetailedDifferences(origSpatial, iterSpatial, ...
    origMorph, iterMorph, origCluster, iterCluster)
% 计算详细差异
differences = struct();
% 关键指标差异
differences.anisotropy = (iterSpatial.anisotropy - origSpatial.anisotropy) / (origSpatial.anisotropy + eps);
differences.tortuosity = (iterSpatial.tortuosity - origSpatial.tortuosity) / origSpatial.tortuosity;
differences.connectivity = iterSpatial.connectivity.largestComponentRatio - ...
    origSpatial.connectivity.largestComponentRatio;
if ~isempty(origMorph.sphericity) && ~isempty(iterMorph.sphericity)
    differences.sphericity = mean(iterMorph.sphericity) - mean(origMorph.sphericity);
    differences.elongation = mean(iterMorph.elongation) - mean(origMorph.elongation);
else
    differences.sphericity = 0;
    differences.elongation = 0;
end
differences.clusterNumber = (iterCluster.numClusters - origCluster.numClusters) / ...
    (origCluster.numClusters + 1);
differences.meanClusterSize = (iterCluster.meanSize - origCluster.meanSize) / ...
    (origCluster.meanSize + eps);
end
%% ========== 结果显示函数 ==========
function displayMatchingResults(matching)
% 显示匹配结果
fprintf('\n----- 匹配度评估结果 -----\n');
% 空间特征
fprintf('\n空间特征匹配度:\n');
fprintf('  两点相关函数: %.3f (相关系数: %.3f)\n', ...
    matching.spatial.tpcMatch, matching.spatial.tpcCorrelation);
fprintf('  各向异性: %.3f (差异: %.4f)\n', ...
    matching.spatial.anisotropyMatch, matching.spatial.anisotropyDiff);
fprintf('  连通性: %.3f\n', matching.spatial.connectivityMatch);
fprintf('  迂曲度: %.3f (差异: %.4f)\n', ...
    matching.spatial.tortuosityMatch, matching.spatial.tortuosityDiff);
fprintf('  综合: %.3f\n', matching.spatial.overall);
% 形态特征
fprintf('\n形态特征匹配度:\n');
fprintf('  球形度: %.3f (差异: %.4f)\n', ...
    matching.morphology.sphericityMatch, matching.morphology.sphericityDiff);
fprintf('  伸长率: %.3f (差异: %.4f)\n', ...
    matching.morphology.elongationMatch, matching.morphology.elongationDiff);
fprintf('  网络密度: %.3f\n', matching.morphology.networkDensityMatch);
fprintf('  分形维数: %.3f\n', matching.morphology.fractalMatch);
fprintf('  综合: %.3f\n', matching.morphology.overall);
% 簇分布
fprintf('\n簇分布匹配度:\n');
fprintf('  簇数量: %.3f (差异: %d)\n', ...
    matching.cluster.numMatch, matching.cluster.numDiff);
fprintf('  平均大小: %.3f\n', matching.cluster.meanSizeMatch);
fprintf('  大小分布: %.3f\n', matching.cluster.sizeDistMatch);
fprintf('  综合: %.3f\n', matching.cluster.overall);
% 多尺度
fprintf('\n多尺度特征匹配度:\n');
fprintf('  尺度不变性: %.3f\n', matching.multiScale.invarianceMatch);
fprintf('  综合: %.3f\n', matching.multiScale.overall);
% 总体评分
fprintf('\n===== 总体评估 =====\n');
fprintf('总体匹配度得分: %.3f\n', matching.overall.score);
fprintf('评级: %s\n', matching.overall.grade);
fprintf('\n各部分权重:\n');
fprintf('  空间特征: %.1f%%\n', matching.overall.weights.spatial * 100);
fprintf('  形态特征: %.1f%%\n', matching.overall.weights.morphology * 100);
fprintf('  簇分布: %.1f%%\n', matching.overall.weights.cluster * 100);
fprintf('  多尺度: %.1f%%\n', matching.overall.weights.multiScale * 100);
end
%% ========== 可视化报告生成 ==========
function generateComprehensiveReport(origModel, iterModel, ...
    origSpatial, iterSpatial, origMorph, iterMorph, ...
    origCluster, iterCluster, origMultiScale, iterMultiScale, ...
    matchingResults)
% 生成综合可视化报告
% 图1: 模型切片对比
figure('Name', '模型切片对比', 'Position', [100, 100, 1400, 900]);
plotModelComparison(origModel, iterModel);
% 图2: 空间特征对比
figure('Name', '空间特征对比', 'Position', [150, 150, 1400, 900]);
plotSpatialComparison(origSpatial, iterSpatial, matchingResults.spatial);
% 图3: 形态特征对比
figure('Name', '形态特征对比', 'Position', [200, 200, 1400, 900]);
plotMorphologyComparison(origMorph, iterMorph, matchingResults.morphology);
% 图4: 簇分布对比
figure('Name', '簇分布对比', 'Position', [250, 250, 1400, 900]);
plotClusterComparison(origCluster, iterCluster, matchingResults.cluster);
% 图5: 多尺度分析
figure('Name', '多尺度分析', 'Position', [300, 300, 1400, 900]);
plotMultiScaleAnalysis(origMultiScale, iterMultiScale, matchingResults.multiScale);
% 图6: 综合匹配度雷达图
figure('Name', '综合匹配度评估', 'Position', [350, 350, 1400, 900]);
plotOverallMatching(matchingResults);
end
function plotModelComparison(origModel, iterModel)
% 绘制模型切片对比
[nx, ny, nz] = size(origModel);
% XY切片
subplot(3,4,1);
imshow(origModel(:,:,round(nz/2)));
title('原始模型 - XY切面');
subplot(3,4,2);
imshow(iterModel(:,:,round(nz/2)));
title('迭代模型 - XY切面');
subplot(3,4,3);
diff = double(origModel(:,:,round(nz/2))) - double(iterModel(:,:,round(nz/2)));
imagesc(diff);
colormap(subplot(3,4,3), redBlueDivergingMap());
colorbar;
title('差异图 - XY切面');
% XZ切片
subplot(3,4,5);
imshow(squeeze(origModel(:,round(ny/2),:)));
title('原始模型 - XZ切面');
subplot(3,4,6);
imshow(squeeze(iterModel(:,round(ny/2),:)));
title('迭代模型 - XZ切面');
subplot(3,4,7);
diff = double(squeeze(origModel(:,round(ny/2),:))) - ...
    double(squeeze(iterModel(:,round(ny/2),:)));
imagesc(diff);
colormap(subplot(3,4,7), redBlueDivergingMap());
colorbar;
title('差异图 - XZ切面');
% YZ切片
subplot(3,4,9);
imshow(squeeze(origModel(round(nx/2),:,:)));
title('原始模型 - YZ切面');
subplot(3,4,10);
imshow(squeeze(iterModel(round(nx/2),:,:)));
title('迭代模型 - YZ切面');
subplot(3,4,11);
diff = double(squeeze(origModel(round(nx/2),:,:))) - ...
    double(squeeze(iterModel(round(nx/2),:,:)));
imagesc(diff);
colormap(subplot(3,4,11), redBlueDivergingMap());
colorbar;
title('差异图 - YZ切面');
% 3D投影
subplot(3,4,[4,8,12]);
projOrig = sum(origModel, 3);
projIter = sum(iterModel, 3);
imagesc([projOrig, zeros(nx, 5), projIter]);
colormap(subplot(3,4,[4,8,12]), 'hot');
colorbar;
title('3D投影对比 (左:原始, 右:迭代)');
axis equal tight;
sgtitle('模型切片对比分析');
end
function plotSpatialComparison(origSpatial, iterSpatial, spatialMatching)
% 绘制空间特征对比
% 两点相关函数
subplot(2,3,1);
if ~isempty(origSpatial.twoPointCorr) && ~isempty(iterSpatial.twoPointCorr)
    minLen = min(length(origSpatial.keyDistances), length(iterSpatial.keyDistances));
    plot(origSpatial.keyDistances(1:minLen), mean(origSpatial.twoPointCorr(1:minLen,:), 2), 'b-', 'LineWidth', 2);
    hold on;
    plot(iterSpatial.keyDistances(1:minLen), mean(iterSpatial.twoPointCorr(1:minLen,:), 2), 'r--', 'LineWidth', 2);
    xlabel('距离');
    ylabel('相关值');
    title(sprintf('两点相关函数 (匹配度: %.3f)', spatialMatching.tpcMatch));
    legend('原始', '迭代', 'Location', 'best');
    grid on;
end
% 径向分布函数
subplot(2,3,2);
if isfield(origSpatial, 'radialDistribution') && isfield(iterSpatial, 'radialDistribution')
    plot(origSpatial.radialDistribution.distances, origSpatial.radialDistribution.values, 'b-', 'LineWidth', 2);
    hold on;
    plot(iterSpatial.radialDistribution.distances, iterSpatial.radialDistribution.values, 'r--', 'LineWidth', 2);
    xlabel('距离');
    ylabel('RDF');
    title('径向分布函数');
    legend('原始', '迭代', 'Location', 'best');
    grid on;
end
% 各项指标对比
subplot(2,3,3);
metrics = {'各向异性', '梯度', '连通性', '迂曲度', '自相关'};
origValues = [origSpatial.anisotropy, origSpatial.porosityGradient, ...
    origSpatial.connectivity.largestComponentRatio, ...
    origSpatial.tortuosity, origSpatial.spatialAutocorrelation];
iterValues = [iterSpatial.anisotropy, iterSpatial.porosityGradient, ...
    iterSpatial.connectivity.largestComponentRatio, ...
    iterSpatial.tortuosity, iterSpatial.spatialAutocorrelation];
x = 1:length(metrics);
width = 0.35;
bar(x - width/2, origValues, width, 'FaceColor', 'b');
hold on;
bar(x + width/2, iterValues, width, 'FaceColor', 'r');
set(gca, 'XTick', x, 'XTickLabel', metrics);
ylabel('值');
title('空间特征指标对比');
legend('原始', '迭代', 'Location', 'best');
xtickangle(45);
% 匹配度条形图
subplot(2,3,4);
matchMetrics = {'两点相关', '各向异性', '梯度', '连通性', '迂曲度', '自相关'};
matchValues = [spatialMatching.tpcMatch, spatialMatching.anisotropyMatch, ...
    spatialMatching.gradientMatch, spatialMatching.connectivityMatch, ...
    spatialMatching.tortuosityMatch, spatialMatching.autocorrMatch];
bar(matchValues);
set(gca, 'XTickLabel', matchMetrics);
ylabel('匹配度');
title('空间特征匹配度');
ylim([0 1]);
xtickangle(45);
hold on;
plot([0 length(matchValues)+1], [0.8 0.8], 'g--', 'LineWidth', 1);
text(length(matchValues)/2, 0.82, '良好阈值', 'HorizontalAlignment', 'center');
% 差异热图
subplot(2,3,5);
differences = abs([origValues - iterValues] ./ (origValues + eps));
imagesc(differences');
colormap(subplot(2,3,5), 'hot');
colorbar;
set(gca, 'YTick', 1:length(metrics), 'YTickLabel', metrics);
title('相对差异热图');
% 综合评分
subplot(2,3,6);
pie([spatialMatching.overall, 1-spatialMatching.overall], ...
    {sprintf('匹配 %.1f%%', spatialMatching.overall*100), ...
     sprintf('差异 %.1f%%', (1-spatialMatching.overall)*100)});
title(sprintf('空间特征综合匹配度: %.3f', spatialMatching.overall));
sgtitle('空间特征对比分析');
end
function plotMorphologyComparison(origMorph, iterMorph, morphMatching)
% 绘制形态特征对比
% 球形度分布
subplot(2,3,1);
if ~isempty(origMorph.sphericity) && ~isempty(iterMorph.sphericity)
    histogram(origMorph.sphericity, 20, 'FaceColor', 'b', 'FaceAlpha', 0.5);
    hold on;
    histogram(iterMorph.sphericity, 20, 'FaceColor', 'r', 'FaceAlpha', 0.5);
    xlabel('球形度');
    ylabel('频数');
    title(sprintf('球形度分布 (匹配度: %.3f)', morphMatching.sphericityMatch));
    legend('原始', '迭代', 'Location', 'best');
end
% 伸长率分布
subplot(2,3,2);
if ~isempty(origMorph.elongation) && ~isempty(iterMorph.elongation)
    histogram(origMorph.elongation, 20, 'FaceColor', 'b', 'FaceAlpha', 0.5);
    hold on;
    histogram(iterMorph.elongation, 20, 'FaceColor', 'r', 'FaceAlpha', 0.5);
    xlabel('伸长率');
    ylabel('频数');
    title(sprintf('伸长率分布 (匹配度: %.3f)', morphMatching.elongationMatch));
    legend('原始', '迭代', 'Location', 'best');
end
% 形态指标箱线图
subplot(2,3,3);
if ~isempty(origMorph.sphericity) && ~isempty(iterMorph.sphericity)
    data1 = [origMorph.sphericity; origMorph.elongation; origMorph.compactness];
    data2 = [iterMorph.sphericity; iterMorph.elongation; iterMorph.compactness];
    
    positions = [1 2 4 5 7 8];
    boxplot([data1' data2'], positions, 'Colors', 'brbrbr');
    set(gca, 'XTick', [1.5 4.5 7.5], 'XTickLabel', {'球形度', '伸长率', '紧凑度'});
    ylabel('值');
    title('形态特征分布对比');
end
% 网络特征对比
subplot(2,3,4);
metrics = {'网络密度', '空隙率', '分形维数'};
origValues = [origMorph.poreNetworkDensity, origMorph.lacunarity, origMorph.fractalDimension];
iterValues = [iterMorph.poreNetworkDensity, iterMorph.lacunarity, iterMorph.fractalDimension];
x = 1:length(metrics);
width = 0.35;
bar(x - width/2, origValues, width, 'FaceColor', 'b');
hold on;
bar(x + width/2, iterValues, width, 'FaceColor', 'r');
set(gca, 'XTick', x, 'XTickLabel', metrics);
ylabel('值');
title('网络特征对比');
legend('原始', '迭代', 'Location', 'best');
% 匹配度雷达图
subplot(2,3,5);
categories = {'球形度', '伸长率', '紧凑度', '网络密度', '空隙率', '分形维数'};
matchValues = [morphMatching.sphericityMatch, morphMatching.elongationMatch, ...
    morphMatching.compactnessMatch, morphMatching.networkDensityMatch, ...
    morphMatching.lacunarityMatch, morphMatching.fractalMatch];
theta = linspace(0, 2*pi, length(categories)+1);
polarplot(theta, [matchValues matchValues(1)], 'r-', 'LineWidth', 2);
hold on;
polarplot(theta, ones(size(theta))*0.8, 'g--');
thetaticks(rad2deg(theta(1:end-1)));
thetaticklabels(categories);
title('形态特征匹配度雷达图');
% 综合评分
subplot(2,3,6);
barh(morphMatching.overall, 'FaceColor', 'g');
xlim([0 1]);
xlabel('匹配度');
title(sprintf('形态特征综合匹配度: %.3f', morphMatching.overall));
grid on;
sgtitle('形态特征对比分析');
end
function plotClusterComparison(origCluster, iterCluster, clusterMatching)
% 绘制簇分布对比
% 簇大小分布
subplot(2,3,1);
if ~isempty(origCluster.sizes) && ~isempty(iterCluster.sizes)
    histogram(log10(origCluster.sizes + 1), 20, 'FaceColor', 'b', 'FaceAlpha', 0.5);
    hold on;
    histogram(log10(iterCluster.sizes + 1), 20, 'FaceColor', 'r', 'FaceAlpha', 0.5);
    xlabel('log10(簇大小)');
    ylabel('频数');
    title(sprintf('簇大小分布 (匹配度: %.3f)', clusterMatching.sizeDistMatch));
    legend('原始', '迭代', 'Location', 'best');
end
% 簇统计对比
subplot(2,3,2);
metrics = {'簇数量', '平均大小', '标准差', '空间离散度'};
origValues = [origCluster.numClusters, origCluster.meanSize, ...
    origCluster.stdSize, origCluster.spatialDispersion];
iterValues = [iterCluster.numClusters, iterCluster.meanSize, ...
    iterCluster.stdSize, iterCluster.spatialDispersion];
% 归一化显示
origNorm = origValues ./ (origValues + iterValues + eps);
iterNorm = iterValues ./ (origValues + iterValues + eps);
x = 1:length(metrics);
width = 0.35;
bar(x - width/2, origNorm, width, 'FaceColor', 'b');
hold on;
bar(x + width/2, iterNorm, width, 'FaceColor', 'r');
set(gca, 'XTick', x, 'XTickLabel', metrics);
ylabel('归一化值');
title('簇特征对比（归一化）');
legend('原始', '迭代', 'Location', 'best');
xtickangle(45);
% 空间分布散点图
subplot(2,3,3);
if size(origCluster.centroids, 1) > 0 && size(iterCluster.centroids, 1) > 0
    scatter3(origCluster.centroids(:,1), origCluster.centroids(:,2), ...
        origCluster.centroids(:,3), 20, 'b', 'filled');
    hold on;
    scatter3(iterCluster.centroids(:,1), iterCluster.centroids(:,2), ...
        iterCluster.centroids(:,3), 20, 'r', 'filled');
    xlabel('X'); ylabel('Y'); zlabel('Z');
    title('簇质心空间分布');
    legend('原始', '迭代', 'Location', 'best');
    view(3);
    grid on;
end
% 匹配度条形图
subplot(2,3,4);
matchMetrics = {'数量', '平均大小', '标准差', '分布', '离散度', '多样性'};
matchValues = [clusterMatching.numMatch, clusterMatching.meanSizeMatch, ...
    clusterMatching.stdSizeMatch, clusterMatching.sizeDistMatch, ...
    clusterMatching.dispersionMatch, clusterMatching.diversityMatch];
bar(matchValues, 'FaceColor', [0.2 0.7 0.2]);
set(gca, 'XTickLabel', matchMetrics);
ylabel('匹配度');
title('簇特征匹配度');
ylim([0 1]);
xtickangle(45);
hold on;
plot([0 length(matchValues)+1], [0.8 0.8], 'r--', 'LineWidth', 1);
% 大小累积分布
subplot(2,3,5);
if ~isempty(origCluster.sizes) && ~isempty(iterCluster.sizes)
    [f1, x1] = ecdf(origCluster.sizes);
    [f2, x2] = ecdf(iterCluster.sizes);
    plot(x1, f1, 'b-', 'LineWidth', 2);
    hold on;
    plot(x2, f2, 'r--', 'LineWidth', 2);
    xlabel('簇大小');
    ylabel('累积概率');
    title('簇大小累积分布');
    legend('原始', '迭代', 'Location', 'best');
    grid on;
end
% 综合评分仪表盘
subplot(2,3,6);
theta = linspace(0, pi, 100);
r = ones(size(theta));
polarplot(theta, r, 'k-', 'LineWidth', 2);
hold on;
scoreAngle = (1 - clusterMatching.overall) * pi;
polarplot([scoreAngle scoreAngle], [0 1], 'r-', 'LineWidth', 3);
polarplot(scoreAngle, 1, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
thetaticks([0 45 90 135 180]);
thetaticklabels({'1.0', '0.75', '0.5', '0.25', '0.0'});
title(sprintf('簇分布综合匹配度: %.3f', clusterMatching.overall));
sgtitle('簇分布对比分析');
end
function plotMultiScaleAnalysis(origMultiScale, iterMultiScale, multiScaleMatching)
% 绘制多尺度分析
nScales = min(length(origMultiScale.scale), length(iterMultiScale.scale));
% 不同尺度的孔隙率
subplot(2,3,1);
scales = [origMultiScale.scale.factor];
origPor = [origMultiScale.scale.porosity];
iterPor = [iterMultiScale.scale.porosity];
plot(scales(1:nScales), origPor(1:nScales), 'b-o', 'LineWidth', 2);
hold on;
plot(scales(1:nScales), iterPor(1:nScales), 'r--s', 'LineWidth', 2);
xlabel('尺度');
ylabel('孔隙率');
title('多尺度孔隙率');
legend('原始', '迭代', 'Location', 'best');
grid on;
% 不同尺度的簇数量
subplot(2,3,2);
origClusters = [origMultiScale.scale.numClusters];
iterClusters = [iterMultiScale.scale.numClusters];
plot(scales(1:nScales), origClusters(1:nScales), 'b-o', 'LineWidth', 2);
hold on;
plot(scales(1:nScales), iterClusters(1:nScales), 'r--s', 'LineWidth', 2);
xlabel('尺度');
ylabel('簇数量');
title('多尺度簇数量');
legend('原始', '迭代', 'Location', 'best');
grid on;
% 尺度匹配度
subplot(2,3,3);
if ~isempty(multiScaleMatching.scaleMatches)
    bar(scales(1:nScales), multiScaleMatching.scaleMatches, 'FaceColor', [0.3 0.7 0.9]);
    xlabel('尺度');
    ylabel('匹配度');
    title('各尺度匹配度');
    ylim([0 1]);
    hold on;
    plot([min(scales) max(scales)], [0.8 0.8], 'g--', 'LineWidth', 1);
end
% 两点相关函数随尺度变化
subplot(2,3,4);
cmap = jet(nScales);
for s = 1:min(3, nScales)  % 只显示前3个尺度
    if ~isempty(origMultiScale.scale(s).twoPointCorr)
        tpc = mean(origMultiScale.scale(s).twoPointCorr, 2);
        plot(tpc, 'Color', cmap(s,:), 'LineWidth', 1.5);
        hold on;
    end
end
xlabel('距离');
ylabel('相关值');
title('原始模型多尺度两点相关');
legend(arrayfun(@(x) sprintf('尺度 %d', x), scales(1:min(3,nScales)), 'UniformOutput', false));
subplot(2,3,5);
for s = 1:min(3, nScales)  % 只显示前3个尺度
    if ~isempty(iterMultiScale.scale(s).twoPointCorr)
        tpc = mean(iterMultiScale.scale(s).twoPointCorr, 2);
        plot(tpc, 'Color', cmap(s,:), 'LineWidth', 1.5);
        hold on;
    end
end
xlabel('距离');
ylabel('相关值');
title('迭代模型多尺度两点相关');
legend(arrayfun(@(x) sprintf('尺度 %d', x), scales(1:min(3,nScales)), 'UniformOutput', false));
% 尺度不变性对比
subplot(2,3,6);
invariances = [origMultiScale.scaleInvariance, iterMultiScale.scaleInvariance];
bar(invariances, 'FaceColor', [0.5 0.5 0.8]);
set(gca, 'XTickLabel', {'原始', '迭代'});
ylabel('尺度不变性');
title(sprintf('尺度不变性 (匹配度: %.3f)', multiScaleMatching.invarianceMatch));
ylim([0 1]);
hold on;
plot([0 3], [mean(invariances) mean(invariances)], 'r--', 'LineWidth', 1);
text(1.5, mean(invariances)+0.05, sprintf('平均: %.3f', mean(invariances)), ...
    'HorizontalAlignment', 'center');
sgtitle('多尺度特征分析');
end
function plotOverallMatching(matchingResults)
% 绘制综合匹配度评估
% 主要类别匹配度
subplot(2,3,1);
categories = {'空间特征', '形态特征', '簇分布', '多尺度'};
values = [matchingResults.spatial.overall, ...
    matchingResults.morphology.overall, ...
    matchingResults.cluster.overall, ...
    matchingResults.multiScale.overall];
bar(values, 'FaceColor', [0.3 0.6 0.9]);
set(gca, 'XTickLabel', categories);
ylabel('匹配度');
title('主要特征类别匹配度');
ylim([0 1]);
hold on;
plot([0 5], [0.8 0.8], 'g--', 'LineWidth', 1);
plot([0 5], [0.6 0.6], 'y--', 'LineWidth', 1);
text(2.5, 0.82, '良好', 'HorizontalAlignment', 'center');
text(2.5, 0.62, '及格', 'HorizontalAlignment', 'center');
% 综合雷达图
subplot(2,3,2);
theta = linspace(0, 2*pi, length(categories)+1);
polarplot(theta, [values values(1)], 'b-', 'LineWidth', 2);
hold on;
polarplot(theta, ones(size(theta))*0.8, 'g--');
polarplot(theta, ones(size(theta))*0.6, 'y--');
thetaticks(rad2deg(theta(1:end-1)));
thetaticklabels(categories);
title('综合匹配度雷达图');
% 详细指标热图
subplot(2,3,[3,6]);
allMetrics = {
    '两点相关', matchingResults.spatial.tpcMatch;
    '各向异性', matchingResults.spatial.anisotropyMatch;
    '连通性', matchingResults.spatial.connectivityMatch;
    '迂曲度', matchingResults.spatial.tortuosityMatch;
    '球形度', matchingResults.morphology.sphericityMatch;
    '伸长率', matchingResults.morphology.elongationMatch;
    '网络密度', matchingResults.morphology.networkDensityMatch;
    '分形维数', matchingResults.morphology.fractalMatch;
    '簇数量', matchingResults.cluster.numMatch;
    '簇大小分布', matchingResults.cluster.sizeDistMatch;
    '尺度不变性', matchingResults.multiScale.invarianceMatch;
};
metricValues = cell2mat(allMetrics(:,2));
imagesc(metricValues');
colormap(jet);
colorbar;
set(gca, 'YTick', 1, 'YTickLabel', '匹配度');
set(gca, 'XTick', 1:length(allMetrics), 'XTickLabel', allMetrics(:,1));
xtickangle(45);
title('详细指标匹配度热图');
caxis([0 1]);
% 总体评分显示
subplot(2,3,4);
scoreText = sprintf('总体匹配度\n%.3f\n%s', ...
    matchingResults.overall.score, ...
    matchingResults.overall.grade);
text(0.5, 0.5, scoreText, 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', 'FontSize', 20, 'FontWeight', 'bold');
axis off;
% 权重分布饼图
subplot(2,3,5);
weights = struct2array(matchingResults.overall.weights);
pie(weights, categories);
title('评估权重分布');
sgtitle('综合匹配度评估报告');
end
%% ========== 结果保存函数 ==========
function saveResults(matchingResults, origStats, iterStats)
% 保存分析结果
% 创建结果结构
results = struct();
results.timestamp = datestr(now);
results.matching = matchingResults;
results.originalStats = origStats;
results.iteratedStats = iterStats;
% 保存到MAT文件
filename = sprintf('model_comparison_results_%s.mat', ...
    datestr(now, 'yyyymmdd_HHMMSS'));
save(filename, 'results');
fprintf('\n分析结果已保存至: %s\n', filename);
% 生成文本报告
reportName = sprintf('model_comparison_report_%s.txt', ...
    datestr(now, 'yyyymmdd_HHMMSS'));
fid = fopen(reportName, 'w');
fprintf(fid, '===== 模型对比分析报告 =====\n');
fprintf(fid, '生成时间: %s\n\n', results.timestamp);
fprintf(fid, '基本统计信息:\n');
fprintf(fid, '原始模型 - 孔隙率: %.4f, 簇数量: %d\n', ...
    origStats.porosity, origStats.numClusters);
fprintf(fid, '迭代模型 - 孔隙率: %.4f, 簇数量: %d\n\n', ...
    iterStats.porosity, iterStats.numClusters);
fprintf(fid, '匹配度评估结果:\n');
fprintf(fid, '空间特征匹配度: %.3f\n', matchingResults.spatial.overall);
fprintf(fid, '形态特征匹配度: %.3f\n', matchingResults.morphology.overall);
fprintf(fid, '簇分布匹配度: %.3f\n', matchingResults.cluster.overall);
fprintf(fid, '多尺度特征匹配度: %.3f\n', matchingResults.multiScale.overall);
fprintf(fid, '\n总体匹配度: %.3f\n', matchingResults.overall.score);
fprintf(fid, '评级: %s\n', matchingResults.overall.grade);
fclose(fid);
fprintf('文本报告已保存至: %s\n', reportName);
end
%% ========== 辅助函数 ==========
function cmap = redBlueDivergingMap(n)
% 生成类似RdBu的发散色图，避免依赖外部工具箱
if nargin < 1
    n = 256;
end

% 在[-1, 1]范围生成线性插值，创建红蓝两端、白色中间的渐变
t = linspace(-1, 1, n)';
% 红色通道在正半轴逐渐增强，蓝色通道在负半轴逐渐增强
r = 0.5 * (1 + t);
b = 0.5 * (1 - t);
% 绿色通道在中心区域保持较高值，营造白色过渡
g = 1 - abs(t);

cmap = [r, g, b];
cmap = max(0, min(1, cmap));
end

