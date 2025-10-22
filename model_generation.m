function N25_comprehensive_morphology_spatial_optimization
% N25_comprehensive_morphology_spatial_optimization - 综合形态和空间特征优化
% 结合了形态一致性和空间匹配度的双重优化
% 主要特点：
% 1. 双阶段优化：形态保持阶段和空间匹配阶段
% 2. 自适应权重调整
% 3. 多尺度特征匹配
% 4. 智能簇管理
% 5. 综合后处理
% 6. 支持用户选择原始模型大小或自定义大小
%% 0. 参数设置与初始化
% 原始模型尺寸（默认值）
originalDims = [150, 150, 40];
% 询问用户选择模型尺寸方式
fprintf('\n===== 模型尺寸设置 =====\n');
fprintf('1. 使用原始数据模型大小 (%d × %d × %d)\n', originalDims(1), originalDims(2), originalDims(3));
fprintf('2. 自定义模型大小\n');
sizeChoice = input('请选择 (1 或 2): ');
if sizeChoice == 2
    % 用户自定义尺寸
    fprintf('\n请输入自定义模型尺寸：\n');
    customX = input('  X 维度大小: ');
    customY = input('  Y 维度大小: ');
    customZ = input('  Z 维度大小: ');
    
    % 验证输入
    if customX <= 0 || customY <= 0 || customZ <= 0
        warning('无效的尺寸输入，将使用原始模型大小');
        dims = originalDims;
    else
        dims = [round(customX), round(customY), round(customZ)];
        fprintf('将使用自定义尺寸: %d × %d × %d\n', dims(1), dims(2), dims(3));
    end
else
    % 使用原始模型大小
    dims = originalDims;
    fprintf('将使用原始模型大小: %d × %d × %d\n', dims(1), dims(2), dims(3));
end
fileName = 'DATA1.raw'; % 输入文件
% 读取原始模型（使用原始尺寸）
rawModel = readRawModel(fileName, originalDims);
% 如果需要，调整原始模型大小
if ~isequal(dims, originalDims)
    fprintf('\n正在调整原始模型大小...\n');
    rawModel = resizeModel(rawModel, originalDims, dims);
end
% 设置孔隙阈值
pore_threshold = 120;
% 计算原始模型统计信息
raw_porosity = mean(rawModel(:) <= pore_threshold);
fprintf('原始灰度模型孔隙率：%.4f\n', raw_porosity);
rawBinaryModel = rawModel <= pore_threshold;
% 提取原始模型特征（只计算一次）
fprintf('正在提取原始模型的综合特征...\n');
tic;
originalFeatures = extractEfficientClusterFeatures(rawBinaryModel);
spatialFeatures = computeEnhancedSpatialFeatures(rawBinaryModel);
morphologyFeatures = computeDetailedMorphologyFeatures(rawBinaryModel);
multiScaleSpatialFeatures = computeMultiScaleSpatialFeatures(rawBinaryModel);
fprintf('特征提取完成，用时：%.2f秒\n', toc);
% 显示特征
displayClusterFeatures(originalFeatures);
displaySpatialFeatures(spatialFeatures);
displayMorphologyFeatures(morphologyFeatures);
displayMultiScaleSpatialFeatures(multiScaleSpatialFeatures);
%% 1. 二值化并保存原始模型
binVol = adaptiveBinarizeModel(rawModel, raw_porosity);
saveRawModel(binVol, 'originalDataModel.raw');
%% 2. 用户设定目标参数
target_porosity = input('请输入目标孔隙率（0-1之间）: ');
target_max = input('请输入目标最大孔隙簇: ');
target_min = input('请输入目标最小孔隙簇: ');
target_cluster_count = input('请输入目标孔隙簇数量: ');
if isempty(target_cluster_count) || ~isfinite(target_cluster_count)
    warning('未提供有效的孔隙簇数量，默认使用原始模型的簇数量。');
    target_cluster_count = originalFeatures.numClusters;
end
target_cluster_count = max(1, round(target_cluster_count));
% 询问是否保留小孔隙特征
fprintf('\n小孔隙保留选项：\n');
fprintf('1. 保留小孔隙（推荐，使结果更接近原始模型）\n');
fprintf('2. 去除小孔隙（获得更平滑的结果）\n');
preserve_small_pores = input('请选择 (1 或 2): ') == 1;
% 预计算优化参数
optParams = struct();
optParams.targetPorosity = target_porosity;
optParams.targetMax = target_max;
optParams.targetMin = target_min;
optParams.targetClusterCount = target_cluster_count;
optParams.originalFeatures = originalFeatures;
optParams.spatialFeatures = spatialFeatures;
optParams.morphologyFeatures = morphologyFeatures;
optParams.multiScaleSpatialFeatures = multiScaleSpatialFeatures;
optParams.modelSize = dims;
optParams.originalBinaryModel = binVol; % 添加原始二值模型供后处理使用
optParams.preserveSmallPores = preserve_small_pores; % 添加小孔隙保留标志
optParams.directionalPorosityProfile = computeDirectionalPorosityProfile(binVol);
optParams.referenceDensityMap = constructReferenceDensityMap(binVol, multiScaleSpatialFeatures);
optParams.clusterLibrary = sampleRepresentativeClusters(binVol, morphologyFeatures);
% 生成快速查找表
fprintf('正在生成快速查找表...\n');
lookupTables = generateComprehensiveLookupTables(binVol, originalFeatures, spatialFeatures, morphologyFeatures);
%% 3. 生成初始模型 - 综合版
fprintf('正在生成综合优化的初始模型...\n');
mcmcModel = generateComprehensiveInitialModel(dims, target_porosity, originalFeatures, ...
    spatialFeatures, morphologyFeatures, optParams);
%% 4. MCMC 迭代参数设置
maxIterations = 3000; % 增加迭代次数
batchSize = 12;
% 自适应能量函数权重
weights = struct();
weights.cluster = 6.0;
weights.porosity = 3.0;
weights.morphology = 8.0;
weights.spatial = 7.0;
weights.connectivity = 5.0;
weights.multiScale = 4.0;
weights.shapePreservation = 10.0;
weights.structureCoherence = 6.0; % 新增：结构一致性权重
% 初始化MCMC状态
mcmcState = initializeComprehensiveMCMCState(mcmcModel, optParams, weights);
% 自适应温度控制
T0 = 0.4;
cooling_rate = 0.9996;
morphology_annealing_rate = 0.9998;
spatial_annealing_rate = 0.9997;
% 性能监控
performanceMonitor = initializeComprehensivePerformanceMonitor(maxIterations);
%% 5. 综合MCMC主循环
fprintf('开始综合形态-空间特征MCMC优化...\n');
startTime = tic;
% 优化阶段控制
optimization_phase = 'morphology'; % 初始阶段
morphology_phase_end = round(maxIterations * 0.35);
spatial_phase_end = round(maxIterations * 0.65);
for iter = 1:maxIterations
    iterStart = tic;
    
    % 阶段控制和权重调整
    if iter <= morphology_phase_end
        optimization_phase = 'morphology';
        % 形态保持阶段：增强形态权重
        currentWeights = adjustWeightsForPhase(weights, 'morphology');
    elseif iter <= spatial_phase_end
        if strcmp(optimization_phase, 'morphology')
            fprintf('\n切换到空间优化阶段...\n');
        end
        optimization_phase = 'spatial';
        % 空间优化阶段：增强空间权重
        currentWeights = adjustWeightsForPhase(weights, 'spatial');
    else
        if ~strcmp(optimization_phase, 'balanced')
            fprintf('\n切换到平衡优化阶段...\n');
        end
        optimization_phase = 'balanced';
        % 平衡阶段：所有权重均衡
        currentWeights = weights;
    end
    
    % 更新当前权重
    mcmcState.weights = currentWeights;
    
    % 定期强制检查
    if mod(iter, 40) == 0
        mcmcState = enforceComprehensiveConstraints(mcmcState, optParams, optimization_phase);
        % 重新计算所有特征
        mcmcState = updateAllFeatures(mcmcState);
    end
    
    % 自适应选择移动策略
    moveStrategy = selectComprehensiveAdaptiveMoveStrategy(iter, maxIterations, ...
        mcmcState, optimization_phase);
    
    % 生成批量候选移动
    [moves, moveTypes] = generateComprehensiveBatchMoves(mcmcState.model, ...
        moveStrategy, batchSize, optParams, lookupTables);
    
    % 评估移动
    deltaEnergies = evaluateComprehensiveBatchMoves(mcmcState, moves, ...
        moveTypes, lookupTables, optimization_phase);
    
    % 应用最佳移动
    [mcmcState, accepted] = applyBestComprehensiveMoves(mcmcState, moves, ...
        deltaEnergies, mcmcState.temperature, moveTypes);
    
    % 更新温度
    if mod(iter, 100) == 0
        switch optimization_phase
            case 'morphology'
                mcmcState.temperature = mcmcState.temperature * morphology_annealing_rate;
            case 'spatial'
                mcmcState.temperature = mcmcState.temperature * spatial_annealing_rate;
            case 'balanced'
                mcmcState.temperature = mcmcState.temperature * cooling_rate;
        end
        mcmcState.temperature = max(mcmcState.temperature, 0.0001);
        
        % 记录性能指标
        idx = iter/100;
        if idx <= length(performanceMonitor.acceptanceRatio)
            performanceMonitor.acceptanceRatio(idx) = mcmcState.acceptanceRate;
            performanceMonitor.timePerIteration(idx) = toc(iterStart);
            performanceMonitor = updatePerformanceMetrics(performanceMonitor, mcmcState, idx);
        end
    end
    
    % 记录能量
    performanceMonitor.energyHistory(iter) = mcmcState.currentEnergy;
    
    % 定期输出进度
    if mod(iter, 500) == 0
        elapsedTime = toc(startTime);
        printComprehensiveProgress(iter, mcmcState, elapsedTime, optimization_phase);
        
        % 可视化
        if mod(iter, 1000) == 0
            visualizeComprehensiveModelSlices(mcmcState.model, iter, optimization_phase);
        end
    end
    
    % 自适应调整
    if mod(iter, 200) == 0
        mcmcState = comprehensiveAdaptiveAdjustment(mcmcState, performanceMonitor, ...
            iter, optimization_phase);
    end
end
totalTime = toc(startTime);
fprintf('MCMC优化完成，总用时：%.2f秒\n', totalTime);
fprintf('平均每次迭代用时：%.4f秒\n', totalTime/maxIterations);
%% 6. 综合后处理
fprintf('正在进行综合后处理...\n');
if optParams.preserveSmallPores
    fprintf('>>> 使用小孔隙保留模式 <<<\n');
    finalModel = comprehensivePostProcessWithSmallPores(mcmcState.bestModel, optParams);
else
    fprintf('>>> 使用平滑模式 <<<\n');
    finalModel = comprehensivePostProcess(mcmcState.bestModel, optParams);
end
% 检查模型质量
checkComprehensiveModelQuality(finalModel, binVol, optParams);
%% 7. 保存结果
outFileName = sprintf('newModel_comprehensive_optimized_%dx%dx%d.raw', dims(1), dims(2), dims(3));
saveRawModel(finalModel, outFileName);
fprintf('优化模型已保存至 %s\n', outFileName);
% 显示最终结果
displayComprehensiveFinalResults(binVol, finalModel, originalFeatures, ...
    spatialFeatures, morphologyFeatures, performanceMonitor, optParams);
end
%% ========== 新增的模型调整函数 ==========
function resizedModel = resizeModel(originalModel, originalDims, targetDims)
    % 调整模型大小到目标尺寸
    % 使用三维插值方法
    
    fprintf('  原始尺寸: %d × %d × %d\n', originalDims(1), originalDims(2), originalDims(3));
    fprintf('  目标尺寸: %d × %d × %d\n', targetDims(1), targetDims(2), targetDims(3));
    
    % 创建原始和目标网格
    [X_orig, Y_orig, Z_orig] = meshgrid(1:originalDims(2), 1:originalDims(1), 1:originalDims(3));
    [X_target, Y_target, Z_target] = meshgrid(...
        linspace(1, originalDims(2), targetDims(2)), ...
        linspace(1, originalDims(1), targetDims(1)), ...
        linspace(1, originalDims(3), targetDims(3)));
    
    % 使用三维插值
    resizedModel = interp3(X_orig, Y_orig, Z_orig, double(originalModel), ...
        X_target, Y_target, Z_target, 'linear');
    
    % 处理NaN值
    resizedModel(isnan(resizedModel)) = 0;
    
    % 转换回uint8
    resizedModel = uint8(resizedModel);
    
    fprintf('  模型大小调整完成\n');
end
%% ========== 文件I/O函数 ==========
function vol = readRawModel(fileName, dims)
    % 读取原始模型文件
    fid = fopen(fileName, 'rb');
    if fid == -1
        error('无法打开文件 %s', fileName);
    end
    vol = fread(fid, prod(dims), 'uint8=>uint8');
    fclose(fid);
    vol = reshape(vol, dims);
end
function saveRawModel(model, fileName)
    % 保存模型到原始文件
    model_uint8 = uint8(model);
    fid = fopen(fileName, 'wb');
    if fid == -1
        error('无法创建文件 %s', fileName);
    end
    fwrite(fid, model_uint8, 'uint8');
    fclose(fid);
    fprintf('模型已保存至: %s\n', fileName);
end
%% ========== 特征提取函数 ==========
function features = extractEfficientClusterFeatures(binaryModel)
    % 高效提取簇特征
    CC = bwconncomp(binaryModel, 26);
    features = struct();
    
    if CC.NumObjects == 0
        features = createEmptyFeatures();
        return;
    end
    
    % 基本统计
    features.numClusters = CC.NumObjects;
    features.sizes = cellfun(@numel, CC.PixelIdxList);
    features.sizeStats = [min(features.sizes), max(features.sizes), ...
        mean(features.sizes), std(features.sizes)];
    
    % 采样计算详细特征（限制计算量）
    nSample = min(100, CC.NumObjects);
    if nSample > 0
        sampleIdx = randperm(CC.NumObjects, nSample);
        features.centroids = zeros(nSample, 3);
        features.compactness = zeros(nSample, 1);
        
        for i = 1:nSample
            idx = sampleIdx(i);
            [x, y, z] = ind2sub(size(binaryModel), CC.PixelIdxList{idx});
            
            % 质心
            features.centroids(i, :) = [mean(x), mean(y), mean(z)];
            
            % 紧凑度
            rangeX = max(x) - min(x) + 1;
            rangeY = max(y) - min(y) + 1;
            rangeZ = max(z) - min(z) + 1;
            boundingBoxVolume = rangeX * rangeY * rangeZ;
            
            if boundingBoxVolume > 0
                features.compactness(i) = length(CC.PixelIdxList{idx}) / boundingBoxVolume;
            else
                features.compactness(i) = 1;
            end
        end
        
        % 空间均匀性
        if size(features.centroids, 1) > 1
            % 计算质心之间的距离变异系数
            distances = pdist(features.centroids);
            features.spatialUniformity = std(distances) / (mean(distances) + eps);
        else
            features.spatialUniformity = 0;
        end
    else
        features.centroids = [];
        features.compactness = [];
        features.spatialUniformity = 0;
    end
end
function features = createEmptyFeatures()
    % 创建空特征结构
    features = struct();
    features.numClusters = 0;
    features.sizes = [];
    features.sizeStats = [0, 0, 0, 0];
    features.centroids = [];
    features.compactness = [];
    features.spatialUniformity = 0;
end
function spatialFeatures = computeEnhancedSpatialFeatures(binaryModel)
    % 计算增强的空间特征
    spatialFeatures = struct();
    
    % 1. 两点相关函数
    keyDistances = [1, 3, 5, 10, 15, 20, 30, 40, 50, 70];
    spatialFeatures.twoPointCorr = computeFastTwoPointCorrelation(binaryModel, keyDistances);
    spatialFeatures.keyDistances = keyDistances;
    
    % 2. 各向异性
    spatialFeatures.anisotropy = computeQuickAnisotropy(binaryModel);
    
    % 3. 孔隙率梯度
    spatialFeatures.porosityGradient = computePorosityGradient(binaryModel);
    
    % 4. 连通性指标
    spatialFeatures.connectivity = computeConnectivityMetrics(binaryModel);
    
    % 5. 表面积体积比
    spatialFeatures.surfaceToVolumeRatio = computeSurfaceToVolumeRatio(binaryModel);
    
    % 6. 迂曲度估计
    spatialFeatures.tortuosityEstimate = estimateTortuosity(binaryModel);

    % 7. 空间自相关
    spatialFeatures.spatialAutocorrelation = computeSpatialAutocorrelation(binaryModel);

    % 8. 弦长分布（Chord Length Distribution）
    spatialFeatures.chordLengthDistribution = computeChordLengthDistribution(binaryModel);

    % 9. 孔隙大小分布（Maximum Ball Method）
    spatialFeatures.poreSizeDistribution = computePoreSizeDistribution(binaryModel);

    % 10. Minkowski 泛函（包含积分平均曲率）
    spatialFeatures.minkowskiFunctionals = computeMinkowskiFunctionals(binaryModel);

    % 11. 线性路径函数
    linealDistances = spatialFeatures.keyDistances;
    maxAxis = max(size(binaryModel));
    linealDistances = linealDistances(linealDistances <= maxAxis);
    if isempty(linealDistances)
        linealDistances = 1:min(10, maxAxis);
    end
    spatialFeatures.linealPathFunction = computeLinealPathFunction(binaryModel, linealDistances);
end
function morphologyFeatures = computeDetailedMorphologyFeatures(binaryModel)
    % 计算详细的形态学特征
    morphologyFeatures = struct();
    
    % 获取连通组分
    CC = bwconncomp(binaryModel, 26);
    if CC.NumObjects == 0
        morphologyFeatures = createEmptyMorphologyFeatures();
        return;
    end
    
    % 限制计算的簇数量
    nAnalyze = min(CC.NumObjects, 50);
    sizes = cellfun(@numel, CC.PixelIdxList);
    [~, sortIdx] = sort(sizes, 'descend');
    analyzeIdx = sortIdx(1:nAnalyze);
    
    % 初始化特征数组
    morphologyFeatures.elongation = zeros(nAnalyze, 1);
    morphologyFeatures.sphericity = zeros(nAnalyze, 1);
    morphologyFeatures.convexity = zeros(nAnalyze, 1);
    morphologyFeatures.solidity = zeros(nAnalyze, 1);
    
    % 计算每个簇的形态特征
    for i = 1:nAnalyze
        idx = analyzeIdx(i);
        [x, y, z] = ind2sub(size(binaryModel), CC.PixelIdxList{idx});
        
        if length(x) > 10 % 只对足够大的簇计算
            % 主成分分析
            coords = [x - mean(x), y - mean(y), z - mean(z)];
            try
                [V, D] = eig(cov(coords));
                eigenvalues = diag(D);
                eigenvalues = sort(eigenvalues, 'descend');
                
                % 伸长率
                if eigenvalues(3) > 0
                    morphologyFeatures.elongation(i) = sqrt(eigenvalues(1) / eigenvalues(3));
                else
                    morphologyFeatures.elongation(i) = 1;
                end
                
                % 球形度（简化计算）
                volume = length(x);
                % 使用椭球体近似
                a = sqrt(eigenvalues(1));
                b = sqrt(eigenvalues(2));
                c = sqrt(eigenvalues(3));
                if a > 0 && b > 0 && c > 0
                    ellipsoidVolume = (4/3) * pi * a * b * c;
                    morphologyFeatures.sphericity(i) = (volume / ellipsoidVolume)^(1/3);
                else
                    morphologyFeatures.sphericity(i) = 0.5;
                end
                
                % 凸性
                rangeX = max(x) - min(x) + 1;
                rangeY = max(y) - min(y) + 1;
                rangeZ = max(z) - min(z) + 1;
                boundingBoxVolume = rangeX * rangeY * rangeZ;
                morphologyFeatures.convexity(i) = volume / boundingBoxVolume;
                
                % 实心度
                morphologyFeatures.solidity(i) = morphologyFeatures.convexity(i);
                
            catch
                % 如果PCA失败，使用默认值
                morphologyFeatures.elongation(i) = 1;
                morphologyFeatures.sphericity(i) = 0.5;
                morphologyFeatures.convexity(i) = 0.5;
                morphologyFeatures.solidity(i) = 0.5;
            end
        else
            % 小簇使用默认值
            morphologyFeatures.elongation(i) = 1;
            morphologyFeatures.sphericity(i) = 0.8;
            morphologyFeatures.convexity(i) = 0.8;
            morphologyFeatures.solidity(i) = 0.8;
        end
    end
    
    % 计算网络特征
    morphologyFeatures.poreNetworkDensity = computePoreNetworkDensity(binaryModel);
    morphologyFeatures.coordinationNumber = computeAverageCoordinationNumber(binaryModel);
    
    % 计算其他形态特征
    morphologyFeatures.lacunarity = computeLacunarity(binaryModel);
    morphologyFeatures.textureFeatures = computeTextureFeatures(binaryModel);
    morphologyFeatures.skeletonFeatures = computeSkeletonFeatures(binaryModel);
end
function morphologyFeatures = createEmptyMorphologyFeatures()
    % 创建空的形态学特征结构
    morphologyFeatures = struct();
    morphologyFeatures.elongation = [];
    morphologyFeatures.sphericity = [];
    morphologyFeatures.convexity = [];
    morphologyFeatures.solidity = [];
    morphologyFeatures.poreNetworkDensity = 0;
    morphologyFeatures.coordinationNumber = 0;
    morphologyFeatures.lacunarity = 0;
    morphologyFeatures.textureFeatures = struct('entropy', 0, 'energy', 0, ...
        'contrast', 0, 'homogeneity', 0);
    morphologyFeatures.skeletonFeatures = struct('density', 0, 'numBranchPoints', 0, ...
        'numEndPoints', 0, 'avgBranchLength', 0);
end
function multiScaleFeatures = computeMultiScaleSpatialFeatures(binaryModel)
    % 计算多尺度空间特征
    multiScaleFeatures = struct();
    baseScales = [1, 2, 4, 8, 16, 32, 50];
    maxDim = max(size(binaryModel));
    scales = baseScales(baseScales <= maxDim);
    if isempty(scales)
        scales = 1;
    elseif scales(1) ~= 1
        scales = [1, scales];
    end
    baseDistances = [1, 3, 5, 10, 15, 20, 30, 40, 50, 70];
    for s = 1:length(scales)
        scale = scales(s);
        % 下采样
        if scale > 1
            scaledModel = binaryModel(1:scale:end, 1:scale:end, 1:scale:end);
        else
            scaledModel = binaryModel;
        end
        % 针对当前尺度筛选可用距离
        maxLag = (max(size(scaledModel)) - 1) * scale;
        validDistances = baseDistances(baseDistances <= maxLag);
        if isempty(validDistances)
            validDistances = baseDistances(1);
        end
        % 计算各尺度的特征
        multiScaleFeatures.scale(s).scaleFactor = scale;
        multiScaleFeatures.scale(s).twoPointCorr = computeFastTwoPointCorrelation(scaledModel, validDistances);
        multiScaleFeatures.scale(s).keyDistances = validDistances;
        multiScaleFeatures.scale(s).clusterDistribution = computeClusterDistribution(scaledModel);
        multiScaleFeatures.scale(s).spatialSpectrum = computeSpatialSpectrum(scaledModel);
    end
    % 计算尺度不变特征
    multiScaleFeatures.scaleInvariantFeatures = computeScaleInvariantFeatures(binaryModel);
end
%% ========== 特征计算辅助函数 ==========
function tpc = computeFastTwoPointCorrelation(binaryModel, keyDistances)
    % 快速计算两点相关函数，支持更大的尺度
    nDistances = length(keyDistances);
    tpc = zeros(nDistances, 3);
    for i = 1:nDistances
        lag = keyDistances(i);
        % 根据距离自适应采样步长
        sampleStep = max(1, round(lag / 10));
        sampledModel = binaryModel(1:sampleStep:end, 1:sampleStep:end, 1:sampleStep:end);
        if isempty(sampledModel)
            tpc(i, :) = mean(binaryModel(:))^2;
            continue;
        end
        for dir = 1:3
            shiftVec = [0, 0, 0];
            effectiveShift = max(1, round(lag / sampleStep));
            maxShift = size(sampledModel, dir) - 1;
            if maxShift <= 0
                tpc(i, dir) = mean(sampledModel(:))^2;
                continue;
            end
            effectiveShift = min(effectiveShift, maxShift);
            shiftVec(dir) = effectiveShift;
            shifted = circshift(sampledModel, shiftVec);
            tpc(i, dir) = mean(sampledModel(:) .* shifted(:));
        end
    end
end
function aniso = computeQuickAnisotropy(binaryModel)
    % 快速计算各向异性
    % 计算三个方向的投影面积
    projXY = sum(binaryModel, 3);
    projXZ = squeeze(sum(binaryModel, 2));
    projYZ = squeeze(sum(binaryModel, 1));
    
    areaXY = sum(projXY(:) > 0);
    areaXZ = sum(projXZ(:) > 0);
    areaYZ = sum(projYZ(:) > 0);
    
    areas = [areaXY, areaXZ, areaYZ];
    
    % 各向异性定义为面积的变异系数
    if mean(areas) > 0
        aniso = std(areas) / mean(areas);
    else
        aniso = 0;
    end
end
function grad = computePorosityGradient(binaryModel)
    % 计算孔隙率梯度
    windowSize = 10;
    stride = 5;
    
    [nx, ny, nz] = size(binaryModel);
    
    % 计算局部孔隙率
    nWindowsX = max(1, floor((nx - windowSize) / stride) + 1);
    nWindowsY = max(1, floor((ny - windowSize) / stride) + 1);
    nWindowsZ = max(1, floor((nz - windowSize) / stride) + 1);
    
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
                
                window = binaryModel(x1:x2, y1:y2, z1:z2);
                localPorosity(i,j,k) = mean(window(:));
            end
        end
    end
    
    % 计算梯度幅值
    [gx, gy, gz] = gradient(localPorosity);
    grad = mean(sqrt(gx(:).^2 + gy(:).^2 + gz(:).^2));
end
function connectivity = computeConnectivityMetrics(binaryModel)
    % 计算连通性指标
    connectivity = struct();
    
    % 3D欧拉数（使用切片平均近似）
    [nx, ny, nz] = size(binaryModel);
    eulerNumbers = zeros(3, 1);
    
    % X方向切片
    eulerSum = 0;
    nSlices = min(10, nx); % 限制计算量
    for i = round(linspace(1, nx, nSlices))
        eulerSum = eulerSum + bweuler(squeeze(binaryModel(i,:,:)), 8);
    end
    eulerNumbers(1) = eulerSum / nSlices;
    
    % Y方向切片
    eulerSum = 0;
    nSlices = min(10, ny);
    for i = round(linspace(1, ny, nSlices))
        eulerSum = eulerSum + bweuler(squeeze(binaryModel(:,i,:)), 8);
    end
    eulerNumbers(2) = eulerSum / nSlices;
    
    % Z方向切片
    eulerSum = 0;
    for i = 1:nz
        eulerSum = eulerSum + bweuler(squeeze(binaryModel(:,:,i)), 8);
    end
    eulerNumbers(3) = eulerSum / nz;
    
    connectivity.eulerNumber = mean(eulerNumbers);
    
    % 连通组分分析
    CC = bwconncomp(binaryModel, 26);
    connectivity.numComponents = CC.NumObjects;
    
    if CC.NumObjects > 0
        sizes = cellfun(@numel, CC.PixelIdxList);
        connectivity.largestComponentRatio = max(sizes) / sum(sizes);
        connectivity.connectivityDensity = CC.NumObjects / numel(binaryModel);
    else
        connectivity.largestComponentRatio = 0;
        connectivity.connectivityDensity = 0;
    end
end
function svRatio = computeSurfaceToVolumeRatio(binaryModel)
    % 计算表面积体积比
    % 使用边界体素近似表面积
    boundary = bwperim(binaryModel, 26);
    surfaceArea = sum(boundary(:));
    volume = sum(binaryModel(:));
    
    if volume > 0
        svRatio = surfaceArea / volume;
    else
        svRatio = 0;
    end
end
function tortuosity = estimateTortuosity(binaryModel)
    % 估计迂曲度（简化版）
    [nx, ny, nz] = size(binaryModel);
    
    % 在Z方向计算迂曲度
    tortuosity = 0;
    validPaths = 0;
    nSamples = min(10, nx*ny/100); % 限制采样数
    
    for s = 1:nSamples
        % 随机选择起点
        startX = randi(nx);
        startY = randi(ny);
        
        % 检查是否存在从顶部到底部的路径
        if binaryModel(startX, startY, 1) && binaryModel(startX, startY, nz)
            % 简化：使用直线距离比
            pathLength = nz;
            straightDistance = nz - 1;
            if straightDistance > 0
                tortuosity = tortuosity + pathLength / straightDistance;
                validPaths = validPaths + 1;
            end
        end
    end
    
    if validPaths > 0
        tortuosity = tortuosity / validPaths;
    else
        tortuosity = 1.5; % 默认值
    end
    
    % 确保迂曲度合理
    tortuosity = max(1, min(tortuosity, 3));
end
function autocorr = computeSpatialAutocorrelation(binaryModel)
    % 计算空间自相关（扩展至多尺度邻域）
    data = double(binaryModel);
    meanVal = mean(data(:));
    % 计算偏差
    deviation = data - meanVal;
    % 采样计算以加速
    totalElements = numel(data);
    if totalElements == 0
        autocorr = 0;
        return;
    end
    baseSamples = max(100, floor(totalElements / 10));
    nSamples = min([totalElements, 1000, baseSamples]);
    nSamples = max(1, nSamples);
    sampleIdx = randperm(totalElements, nSamples);
    sumNum = 0;
    sumDenom = sum(deviation(:).^2);
    nPairs = 0;
    distanceScales = [1, 5, 10, 20, 30, 40, 50];
    offsets = [1, 0, 0; -1, 0, 0; 0, 1, 0; 0, -1, 0; 0, 0, 1; 0, 0, -1];
    for i = 1:nSamples
        [x, y, z] = ind2sub(size(data), sampleIdx(i));
        for d = 1:length(distanceScales)
            scale = distanceScales(d);
            for n = 1:size(offsets, 1)
                nx = x + offsets(n,1) * scale;
                ny = y + offsets(n,2) * scale;
                nz = z + offsets(n,3) * scale;
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
function cld = computeChordLengthDistribution(binaryModel)
    % 计算弦长分布（Chord Length Distribution, CLD）
    poreSpace = logical(binaryModel);
    dims = size(poreSpace);
    dirNames = {'x', 'y', 'z'};
    dirLengths = cell(3, 1);
    directionalStats = struct();

    for axis = 1:3
        permOrder = [axis, setdiff(1:ndims(poreSpace), axis)];
        permuted = permute(poreSpace, permOrder);
        lines = reshape(permuted, dims(axis), []);
        dirLengths{axis} = extractChordLengthsFromLines(lines);

        if isempty(dirLengths{axis})
            directionalStats.(dirNames{axis}) = struct( ...
                'mean', 0, 'std', 0, 'sampleCount', 0, 'histogram', []);
        else
            directionalStats.(dirNames{axis}) = struct( ...
                'mean', mean(dirLengths{axis}), ...
                'std', std(dirLengths{axis}), ...
                'sampleCount', numel(dirLengths{axis}), ...
                'histogram', []);
        end
    end

    allLengths = vertcat(dirLengths{:});

    if isempty(allLengths)
        cld = struct('binCenters', [], 'probability', [], 'meanLength', 0, ...
            'stdLength', 0, 'sampleCount', 0, 'directionalStats', directionalStats);
        return;
    end

    maxLen = max(allLengths);
    binEdges = 0.5:(maxLen + 0.5);
    probability = histcounts(allLengths, binEdges, 'Normalization', 'probability');
    binCenters = 1:maxLen;

    % 计算方向直方图
    for axis = 1:3
        lengths = dirLengths{axis};
        if isempty(lengths)
            histVals = zeros(size(probability));
        else
            histVals = histcounts(lengths, binEdges, 'Normalization', 'probability');
        end
        directionalStats.(dirNames{axis}).histogram = histVals;
    end

    cld = struct();
    cld.binCenters = binCenters;
    cld.probability = probability;
    cld.meanLength = mean(allLengths);
    cld.stdLength = std(allLengths);
    cld.sampleCount = numel(allLengths);
    cld.directionalStats = directionalStats;
end
function lengths = extractChordLengthsFromLines(lines)
    % 从多条线段中提取连续孔隙段的长度
    nLines = size(lines, 2);
    lengthCells = cell(nLines, 1);
    totalCount = 0;

    for idx = 1:nLines
        line = lines(:, idx);
        if any(line)
            d = diff([0; double(line(:)); 0]);
            starts = find(d == 1);
            ends = find(d == -1) - 1;
            segLengths = ends - starts + 1;
            lengthCells{idx} = segLengths;
            totalCount = totalCount + numel(segLengths);
        end
    end

    if totalCount == 0
        lengths = [];
        return;
    end

    lengths = zeros(totalCount, 1);
    pos = 1;
    for idx = 1:nLines
        segLengths = lengthCells{idx};
        if ~isempty(segLengths)
            n = numel(segLengths);
            lengths(pos:pos+n-1) = segLengths;
            pos = pos + n;
        end
    end
end
function psd = computePoreSizeDistribution(binaryModel)
    % 计算孔隙大小分布（Maximum Ball Method）
    poreSpace = logical(binaryModel);
    if ~any(poreSpace(:))
        psd = struct('binCenters', [], 'probability', [], 'meanRadius', 0, ...
            'stdRadius', 0, 'maxRadius', 0, 'sampleCount', 0);
        return;
    end

    distanceMap = bwdist(~poreSpace);
    try
        maxima = imregionalmax(distanceMap);
    catch
        % 如果缺少该函数，退化为直接使用距离场
        maxima = false(size(distanceMap));
    end

    radii = distanceMap(maxima & poreSpace);
    if isempty(radii)
        radii = distanceMap(poreSpace);
    end

    maxRadius = max(radii);
    if maxRadius == 0
        psd = struct('binCenters', 0, 'probability', 1, 'meanRadius', 0, ...
            'stdRadius', 0, 'maxRadius', 0, 'sampleCount', numel(radii));
        return;
    end

    nBins = min(30, max(10, round(maxRadius * 2)));
    binEdges = linspace(0, maxRadius, nBins + 1);
    probability = histcounts(radii, binEdges, 'Normalization', 'probability');
    binCenters = (binEdges(1:end-1) + binEdges(2:end)) / 2;

    psd = struct();
    psd.binCenters = binCenters;
    psd.probability = probability;
    psd.meanRadius = mean(radii);
    psd.stdRadius = std(radii);
    psd.maxRadius = maxRadius;
    psd.sampleCount = numel(radii);
end
function minkowski = computeMinkowskiFunctionals(binaryModel)
    % 计算 Minkowski 泛函（含积分平均曲率）
    poreSpace = logical(binaryModel);
    n3 = sum(poreSpace(:));

    % 计算共享面、边、顶点数量
    faceOffsets = [1, 0, 0; 0, 1, 0; 0, 0, 1];
    edgeOffsets = [1, 1, 0; 1, -1, 0; 1, 0, 1; 1, 0, -1; 0, 1, 1; 0, 1, -1];
    vertexOffsets = [1, 1, 1; 1, 1, -1; 1, -1, 1; 1, -1, -1];

    n2 = 0;
    for i = 1:size(faceOffsets, 1)
        n2 = n2 + countOffsetPairs(poreSpace, faceOffsets(i, :));
    end

    n1 = 0;
    for i = 1:size(edgeOffsets, 1)
        n1 = n1 + countOffsetPairs(poreSpace, edgeOffsets(i, :));
    end

    n0 = 0;
    for i = 1:size(vertexOffsets, 1)
        n0 = n0 + countOffsetPairs(poreSpace, vertexOffsets(i, :));
    end

    surfaceArea = 6 * n3 - 2 * n2;
    meanBreadth = 3 * n3 - 2 * n2 + n1;
    integralMeanCurvature = 2 * pi * meanBreadth;
    eulerCharacteristic = n3 - n2 + n1 - n0;

    minkowski = struct();
    minkowski.volume = n3;
    minkowski.surfaceArea = surfaceArea;
    minkowski.meanBreadth = meanBreadth;
    minkowski.integralMeanCurvature = integralMeanCurvature;
    minkowski.eulerCharacteristic = eulerCharacteristic;
end
function count = countOffsetPairs(volume, offset)
    % 统计在给定偏移下的重叠体素对数量
    sx = offset(1);
    sy = offset(2);
    sz = offset(3);
    [nx, ny, nz] = size(volume);

    xIdx = max(1, 1 - sx):min(nx, nx - sx);
    yIdx = max(1, 1 - sy):min(ny, ny - sy);
    zIdx = max(1, 1 - sz):min(nz, nz - sz);

    if isempty(xIdx) || isempty(yIdx) || isempty(zIdx)
        count = 0;
        return;
    end

    sub1 = volume(xIdx, yIdx, zIdx);
    sub2 = volume(xIdx + sx, yIdx + sy, zIdx + sz);
    count = sum(sub1(:) & sub2(:));
end
function lineal = computeLinealPathFunction(binaryModel, distances)
    % 计算线性路径函数 L(r)
    if nargin < 2 || isempty(distances)
        maxDim = max(size(binaryModel));
        distances = 1:min(50, maxDim);
    end

    poreSpace = logical(binaryModel);
    dirValues = zeros(length(distances), 3);

    for axis = 1:3
        dirValues(:, axis) = computeLinealPathAlongAxis(poreSpace, axis, distances);
    end

    lineal = struct();
    lineal.distances = distances(:);
    lineal.values = dirValues;
    lineal.mean = mean(dirValues, 2, 'omitnan');
    lineal.mean(isnan(lineal.mean)) = 0;
    lineal.byDirection = struct('x', dirValues(:,1), 'y', dirValues(:,2), 'z', dirValues(:,3));
end
function values = computeLinealPathAlongAxis(volume, axis, distances)
    % 计算指定方向上的线性路径函数
    dims = size(volume);
    len = dims(axis);
    permOrder = [axis, setdiff(1:ndims(volume), axis)];
    permuted = permute(volume, permOrder);
    lines = reshape(permuted, len, []);
    data = double(lines);
    nLines = size(lines, 2);

    values = nan(length(distances), 1);
    for idx = 1:length(distances)
        r = distances(idx);
        if r <= 0 || r > len
            values(idx) = NaN;
            continue;
        end

        windowSum = convn(data, ones(r, 1), 'valid');
        totalSegments = (len - r + 1) * nLines;
        if totalSegments <= 0
            values(idx) = NaN;
        else
            values(idx) = sum(windowSum(:) == r) / totalSegments;
        end
    end
end
function diffVal = computeDistributionDifference(currentBins, currentProb, targetBins, targetProb)
    % 计算两个分布之间的差异（直方图插值比较）
    if isempty(currentBins) || isempty(targetBins) || isempty(currentProb) || isempty(targetProb)
        diffVal = 0;
        return;
    end

    minVal = max(min(currentBins), min(targetBins));
    maxVal = min(max(currentBins), max(targetBins));

    if maxVal <= minVal
        diffVal = 0;
        return;
    end

    nSamples = min(100, max(20, round((maxVal - minVal) * 5)));
    samplePoints = linspace(minVal, maxVal, nSamples);
    currentInterp = interp1(currentBins, currentProb, samplePoints, 'linear', 0);
    targetInterp = interp1(targetBins, targetProb, samplePoints, 'linear', 0);

    currentInterp = max(currentInterp, 0);
    targetInterp = max(targetInterp, 0);

    diffVal = mean(abs(currentInterp - targetInterp));
end
function diffVal = normalizeDifference(currentValue, targetValue)
    % 归一化的绝对差异
    if nargin < 2
        diffVal = 0;
        return;
    end

    if isempty(currentValue) || ~isfinite(currentValue)
        currentValue = 0;
    end
    if isempty(targetValue) || ~isfinite(targetValue)
        targetValue = 0;
    end

    denom = abs(targetValue) + 1e-6;
    diffVal = abs(currentValue - targetValue) / denom;
end
function diffVal = computeLinealPathDifference(currentLineal, targetLineal)
    % 计算线性路径函数差异
    if isempty(currentLineal) || isempty(targetLineal)
        diffVal = 0;
        return;
    end

    if ~isfield(currentLineal, 'distances') || ~isfield(targetLineal, 'distances') || ...
            ~isfield(currentLineal, 'mean') || ~isfield(targetLineal, 'mean')
        diffVal = 0;
        return;
    end

    minDist = max(min(currentLineal.distances), min(targetLineal.distances));
    maxDist = min(max(currentLineal.distances), max(targetLineal.distances));

    if maxDist <= minDist
        diffVal = 0;
        return;
    end

    nSamples = min(50, max(10, numel(unique([currentLineal.distances(:); targetLineal.distances(:)]))));
    samplePoints = linspace(minDist, maxDist, nSamples);
    currentInterp = interp1(currentLineal.distances, currentLineal.mean, samplePoints, 'linear', 'extrap');
    targetInterp = interp1(targetLineal.distances, targetLineal.mean, samplePoints, 'linear', 'extrap');

    diffVal = mean(abs(currentInterp - targetInterp), 'omitnan');
    if isnan(diffVal)
        diffVal = 0;
    end
end
function density = computePoreNetworkDensity(model)
    % 计算孔隙网络密度
    % 使用骨架化来近似网络密度
    try
        % 简化的骨架提取
        skeleton = bwmorph(model, 'skel', 5);
        density = sum(skeleton(:)) / (sum(model(:)) + eps);
    catch
        density = 0.1; % 默认值
    end
end
function avgCoordination = computeAverageCoordinationNumber(model)
    % 计算平均配位数
    CC = bwconncomp(model, 26);
    if CC.NumObjects <= 1
        avgCoordination = 0;
        return;
    end
    
    % 简化计算：只考虑前20个最大的簇
    nAnalyze = min(20, CC.NumObjects);
    sizes = cellfun(@numel, CC.PixelIdxList);
    [~, sortIdx] = sort(sizes, 'descend');
    
    coordinationNumbers = zeros(nAnalyze, 1);
    
    for i = 1:nAnalyze
        idx = sortIdx(i);
        clusterMask = false(size(model));
        clusterMask(CC.PixelIdxList{idx}) = true;
        
        % 膨胀以找到相邻簇
        dilated = imdilate(clusterMask, ones(3,3,3));
        
        % 计算接触的簇数
        contactingClusters = 0;
        for j = 1:CC.NumObjects
            if j ~= idx
                if any(dilated(CC.PixelIdxList{j}))
                    contactingClusters = contactingClusters + 1;
                end
            end
        end
        
        coordinationNumbers(i) = contactingClusters;
    end
    
    avgCoordination = mean(coordinationNumbers);
end
function lacunarity = computeLacunarity(model)
    % 计算空隙率（简化版）
    boxSizes = [5, 10, 15];
    lacunarityValues = zeros(length(boxSizes), 1);
    
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
            lacunarityValues(b) = var(masses) / mean(masses)^2;
        else
            lacunarityValues(b) = 0;
        end
    end
    
    lacunarity = mean(lacunarityValues);
end
function textureFeatures = computeTextureFeatures(model)
    % 计算纹理特征（简化版）
    textureFeatures = struct();
    
    % 使用中间切片
    midSlice = model(:, :, round(size(model, 3)/2));
    midSlice = double(midSlice);
    
    % 熵
    p = midSlice(:) / sum(midSlice(:) + eps);
    p(p == 0) = [];
    textureFeatures.entropy = -sum(p .* log2(p + eps));
    
    % 能量
    textureFeatures.energy = sum(midSlice(:).^2) / numel(midSlice);
    
    % 对比度
    [gx, gy] = gradient(midSlice);
    textureFeatures.contrast = mean(sqrt(gx(:).^2 + gy(:).^2));
    
    % 同质性
    localStd = stdfilt(midSlice, ones(5));
    textureFeatures.homogeneity = 1 / (mean(localStd(:)) + 1);
end
function skeletonFeatures = computeSkeletonFeatures(model)
    % 计算骨架特征（简化版）
    skeletonFeatures = struct();
    
    try
        % 简化的骨架提取
        skeleton = bwmorph(model, 'skel', 5);
        
        % 骨架密度
        skeletonFeatures.density = sum(skeleton(:)) / (sum(model(:)) + eps);
        
        % 分支点（简化计算）
        branchPoints = bwmorph(skeleton, 'branchpoints');
        skeletonFeatures.numBranchPoints = sum(branchPoints(:));
        
        % 端点
        endPoints = bwmorph(skeleton, 'endpoints');
        skeletonFeatures.numEndPoints = sum(endPoints(:));
        
        % 平均分支长度
        if skeletonFeatures.numBranchPoints > 0
            skeletonFeatures.avgBranchLength = sum(skeleton(:)) / ...
                (skeletonFeatures.numBranchPoints + 1);
        else
            skeletonFeatures.avgBranchLength = 0;
        end
        
    catch
        % 如果失败，返回默认值
        skeletonFeatures.density = 0.1;
        skeletonFeatures.numBranchPoints = 10;
        skeletonFeatures.numEndPoints = 20;
        skeletonFeatures.avgBranchLength = 5;
    end
end
function distribution = computeClusterDistribution(model)
    % 计算簇分布
    CC = bwconncomp(model, 26);
    distribution = struct();
    
    if CC.NumObjects == 0
        distribution.sizes = [];
        distribution.positions = [];
        distribution.density = 0;
        return;
    end
    
    distribution.sizes = cellfun(@numel, CC.PixelIdxList);
    
    % 只计算前50个簇的位置
    nCalc = min(50, CC.NumObjects);
    distribution.positions = zeros(nCalc, 3);
    
    for i = 1:nCalc
        [x, y, z] = ind2sub(size(model), CC.PixelIdxList{i});
        distribution.positions(i, :) = [mean(x), mean(y), mean(z)];
    end
    
    distribution.density = CC.NumObjects / numel(model);
end
function spectrum = computeSpatialSpectrum(model)
    % 计算空间频谱（简化版）
    % 使用2D FFT在中间切片
    midSlice = model(:, :, round(size(model, 3)/2));
    fftSlice = fft2(double(midSlice));
    powerSpectrum = abs(fftSlice).^2;
    
    % 径向平均
    [nx, ny] = size(midSlice);
    cx = floor(nx/2) + 1;
    cy = floor(ny/2) + 1;
    maxRadius = min(cx, cy) - 1;
    nBins = min(20, maxRadius);
    
    spectrum = zeros(nBins, 1);
    
    for r = 1:nBins
        radius = r * maxRadius / nBins;
        innerRadius = (r-1) * maxRadius / nBins;
        
        % 创建环形掩码
        [X, Y] = meshgrid(1:ny, 1:nx);
        dist = sqrt((X-cx).^2 + (Y-cy).^2);
        mask = (dist >= innerRadius) & (dist < radius);
        
        spectrum(r) = mean(powerSpectrum(mask));
    end
end
function invariantFeatures = computeScaleInvariantFeatures(model)
    % 计算尺度不变特征
    invariantFeatures = struct();
    % 分形维数（盒计数法简化版）
    boxSizes = [1, 2, 4, 8, 16, 32, 64];
    maxBox = min(size(model));
    boxSizes = boxSizes(boxSizes <= maxBox);
    if isempty(boxSizes)
        boxSizes = 1;
    end
    counts = zeros(length(boxSizes), 1);
    for i = 1:length(boxSizes)
        boxSize = boxSizes(i);
        % 计算覆盖模型所需的盒子数
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
    % 线性拟合计算分形维数
    if sum(counts > 0) > 1
        validIdx = counts > 0;
        p = polyfit(log(1./boxSizes(validIdx)), log(counts(validIdx)), 1);
        invariantFeatures.fractalDimension = p(1);
    else
        invariantFeatures.fractalDimension = 2.5; % 默认值
    end
    % 确保分形维数在合理范围
    invariantFeatures.fractalDimension = max(2, min(3, invariantFeatures.fractalDimension));
    % 其他尺度不变特征（简化）
    invariantFeatures.voidScalingExponent = 1.5; % 默认值
    invariantFeatures.connectivityScaling = 0.8; % 默认值
end
%% ========== 显示函数 ==========
function displayClusterFeatures(features)
    % 显示簇特征
    fprintf('\n孔隙簇特征分析：\n');
    fprintf('  簇数量: %d\n', features.numClusters);
    if ~isempty(features.sizes)
        fprintf('  簇大小统计：\n');
        fprintf('    最小: %d\n', features.sizeStats(1));
        fprintf('    最大: %d\n', features.sizeStats(2));
        fprintf('    平均: %.1f\n', features.sizeStats(3));
        fprintf('    标准差: %.1f\n', features.sizeStats(4));
    end
    if ~isempty(features.compactness)
        fprintf('  平均紧凑度: %.3f\n', mean(features.compactness));
    end
    fprintf('  空间均匀性: %.3f\n', features.spatialUniformity);
end
function displaySpatialFeatures(spatialFeatures)
    % 显示空间特征
    fprintf('\n空间特征分析：\n');
    fprintf('  各向异性: %.4f\n', spatialFeatures.anisotropy);
    fprintf('  孔隙率梯度: %.4f\n', spatialFeatures.porosityGradient);
    if isfield(spatialFeatures, 'connectivity')
        fprintf('  连通性指标：\n');
        fprintf('    欧拉数: %.2f\n', spatialFeatures.connectivity.eulerNumber);
        fprintf('    最大连通组分占比: %.4f\n', spatialFeatures.connectivity.largestComponentRatio);
        fprintf('    连通组分数量: %d\n', spatialFeatures.connectivity.numComponents);
    end
    if isfield(spatialFeatures, 'surfaceToVolumeRatio')
        fprintf('  表面积体积比: %.4f\n', spatialFeatures.surfaceToVolumeRatio);
    end
    if isfield(spatialFeatures, 'tortuosityEstimate')
        fprintf('  迂曲度估计: %.4f\n', spatialFeatures.tortuosityEstimate);
    end
    if isfield(spatialFeatures, 'spatialAutocorrelation')
        fprintf('  空间自相关: %.4f\n', spatialFeatures.spatialAutocorrelation);
    end
    if isfield(spatialFeatures, 'chordLengthDistribution') && ...
            ~isempty(spatialFeatures.chordLengthDistribution)
        cld = spatialFeatures.chordLengthDistribution;
        fprintf('  弦长分布: 均值=%.3f, 标准差=%.3f, 样本=%d\n', ...
            cld.meanLength, cld.stdLength, cld.sampleCount);
    end
    if isfield(spatialFeatures, 'poreSizeDistribution') && ...
            ~isempty(spatialFeatures.poreSizeDistribution)
        psd = spatialFeatures.poreSizeDistribution;
        fprintf('  孔隙大小分布: 均值=%.3f, 最大=%.3f\n', ...
            psd.meanRadius, psd.maxRadius);
    end
    if isfield(spatialFeatures, 'minkowskiFunctionals') && ...
            ~isempty(spatialFeatures.minkowskiFunctionals)
        minkowski = spatialFeatures.minkowskiFunctionals;
        fprintf('  积分平均曲率: %.3f\n', minkowski.integralMeanCurvature);
    end
    if isfield(spatialFeatures, 'linealPathFunction') && ...
            ~isempty(spatialFeatures.linealPathFunction)
        lpf = spatialFeatures.linealPathFunction;
        sampleIdx = min(3, length(lpf.distances));
        if sampleIdx > 0
            fprintf('  线性路径函数: ');
            for i = 1:sampleIdx
                fprintf('L(%d)=%.3f ', lpf.distances(i), lpf.mean(i));
            end
            fprintf('\n');
        end
    end
end
function displayMorphologyFeatures(morphologyFeatures)
    % 显示形态学特征
    fprintf('\n形态学特征分析：\n');
    if ~isempty(morphologyFeatures.elongation)
        fprintf('  平均伸长率: %.3f (±%.3f)\n', ...
            mean(morphologyFeatures.elongation), std(morphologyFeatures.elongation));
        fprintf('  平均球形度: %.3f (±%.3f)\n', ...
            mean(morphologyFeatures.sphericity), std(morphologyFeatures.sphericity));
        fprintf('  平均凸性: %.3f (±%.3f)\n', ...
            mean(morphologyFeatures.convexity), std(morphologyFeatures.convexity));
    end
    fprintf('  孔隙网络密度: %.4f\n', morphologyFeatures.poreNetworkDensity);
    fprintf('  平均配位数: %.2f\n', morphologyFeatures.coordinationNumber);
    fprintf('  空隙率(Lacunarity): %.4f\n', morphologyFeatures.lacunarity);
    if isfield(morphologyFeatures, 'textureFeatures')
        fprintf('  纹理特征：\n');
        fprintf('    熵: %.4f\n', morphologyFeatures.textureFeatures.entropy);
        fprintf('    对比度: %.4f\n', morphologyFeatures.textureFeatures.contrast);
    end
    if isfield(morphologyFeatures, 'skeletonFeatures')
        fprintf('  骨架特征：\n');
        fprintf('    骨架密度: %.4f\n', morphologyFeatures.skeletonFeatures.density);
        fprintf('    分支点数: %d\n', morphologyFeatures.skeletonFeatures.numBranchPoints);
    end
end
function displayMultiScaleSpatialFeatures(multiScaleFeatures)
    % 显示多尺度空间特征
    fprintf('\n多尺度空间特征分析：\n');
    nScales = length(multiScaleFeatures.scale);
    for s = 1:nScales
        scaleLabel = s;
        if isfield(multiScaleFeatures.scale(s), 'scaleFactor')
            scaleLabel = multiScaleFeatures.scale(s).scaleFactor;
        end
        fprintf('  尺度 %d (采样步长: %.0f)：\n', s, scaleLabel);
        if isfield(multiScaleFeatures.scale(s), 'twoPointCorr') && ...
            ~isempty(multiScaleFeatures.scale(s).twoPointCorr)
            fprintf('    两点相关函数均值: %.4f\n', ...
                mean(multiScaleFeatures.scale(s).twoPointCorr(:)));
            if isfield(multiScaleFeatures.scale(s), 'keyDistances')
                fprintf('    覆盖距离: %s\n', mat2str(multiScaleFeatures.scale(s).keyDistances));
            end
        end
        if isfield(multiScaleFeatures.scale(s), 'clusterDistribution')
            fprintf('    簇密度: %.6f\n', ...
                multiScaleFeatures.scale(s).clusterDistribution.density);
        end
    end
    if isfield(multiScaleFeatures, 'scaleInvariantFeatures')
        fprintf('\n尺度不变特征：\n');
        fprintf('  分形维数: %.3f\n', ...
            multiScaleFeatures.scaleInvariantFeatures.fractalDimension);
    end
end
%% ========== 模型初始化函数 ==========
function binVol = adaptiveBinarizeModel(vol, raw_porosity)
    % 自适应二值化
    vol_double = double(vol);
    
    % 使用二分搜索找到合适的阈值
    low = min(vol_double(:));
    high = max(vol_double(:));
    tol = 0.001;
    maxIter = 50;
    
    for iter = 1:maxIter
        threshold = (low + high) / 2;
        binVol = vol_double <= threshold;
        curr_por = mean(binVol(:));
        
        if abs(curr_por - raw_porosity) <= tol
            break;
        end
        
        if curr_por > raw_porosity
            high = threshold;
        else
            low = threshold;
        end
    end
    
    % 快速形态学清理
    se = strel('cube', 2);
    binVol = imopen(binVol, se);
    
    fprintf('二值化完成，阈值: %.2f，孔隙率: %.4f\n', threshold, mean(binVol(:)));
end
function lookupTables = generateComprehensiveLookupTables(binaryModel, ...
    originalFeatures, spatialFeatures, morphologyFeatures)
    % 生成综合的快速查找表
    lookupTables = struct();
    
    % 基础查找表
    lookupTables.distanceTransform = bwdist(~binaryModel);
    lookupTables.localPorosityMap = computeLocalPorosityMap(binaryModel, 10);
    
    % 空间特征查找表
    lookupTables.spatialGradient = computeSpatialGradientField(binaryModel);
    lookupTables.localAnisotropy = computeLocalAnisotropyMap(binaryModel);
    lookupTables.targetSpatialSignature = spatialFeatures;
    
    % 形态特征查找表
    lookupTables.localCurvature = computeLocalCurvatureMap(binaryModel);
    lookupTables.localThickness = computeLocalThicknessMap(binaryModel);
    lookupTables.targetMorphologySignature = morphologyFeatures;
    
    % 结构连贯性查找表
    lookupTables.structureCoherenceMap = computeStructureCoherenceMap(binaryModel);
    
    % 目标特征统计
    if ~isempty(originalFeatures.sizes)
        lookupTables.targetSizeHist = histcounts(log10(originalFeatures.sizes), 20);
        lookupTables.targetSizeRange = [min(originalFeatures.sizes), max(originalFeatures.sizes)];
    else
        lookupTables.targetSizeHist = [];
        lookupTables.targetSizeRange = [1, 1000];
    end
end
function localPorosityMap = computeLocalPorosityMap(binaryModel, windowSize)
    % 计算局部孔隙率图
    [nx, ny, nz] = size(binaryModel);
    localPorosityMap = zeros(nx, ny, nz);
    halfSize = floor(windowSize/2);
    
    % 使用积分图像加速
    integralImage = cumsum(cumsum(cumsum(double(binaryModel), 1), 2), 3);
    
    for x = 1:nx
        for y = 1:ny
            for z = 1:nz
                % 定义窗口边界
                x1 = max(1, x - halfSize);
                x2 = min(nx, x + halfSize);
                y1 = max(1, y - halfSize);
                y2 = min(ny, y + halfSize);
                z1 = max(1, z - halfSize);
                z2 = min(nz, z + halfSize);
                
                % 使用积分图像快速计算
                windowSum = getBoxSum(integralImage, x1, y1, z1, x2, y2, z2);
                windowSize = (x2-x1+1) * (y2-y1+1) * (z2-z1+1);
                localPorosityMap(x, y, z) = windowSum / windowSize;
            end
        end
    end
end
function boxSum = getBoxSum(integralImage, x1, y1, z1, x2, y2, z2)
    % 使用积分图像计算盒子和
    boxSum = integralImage(x2, y2, z2);
    
    if x1 > 1
        boxSum = boxSum - integralImage(x1-1, y2, z2);
    end
    if y1 > 1
        boxSum = boxSum - integralImage(x2, y1-1, z2);
    end
    if z1 > 1
        boxSum = boxSum - integralImage(x2, y2, z1-1);
    end
    if x1 > 1 && y1 > 1
        boxSum = boxSum + integralImage(x1-1, y1-1, z2);
    end
    if x1 > 1 && z1 > 1
        boxSum = boxSum + integralImage(x1-1, y2, z1-1);
    end
    if y1 > 1 && z1 > 1
        boxSum = boxSum + integralImage(x2, y1-1, z1-1);
    end
    if x1 > 1 && y1 > 1 && z1 > 1
        boxSum = boxSum - integralImage(x1-1, y1-1, z1-1);
    end
end
function gradientField = computeSpatialGradientField(binaryModel)
    % 计算空间梯度场
    % 高斯平滑
    smoothed = imgaussfilt3(double(binaryModel), 2);
    
    % 计算梯度
    [gx, gy, gz] = gradient(smoothed);
    
    % 梯度幅值
    gradientField = sqrt(gx.^2 + gy.^2 + gz.^2);
end
function localAnisotropyMap = computeLocalAnisotropyMap(binaryModel)
    % 计算局部各向异性图
    windowSize = 10;
    stride = 5;
    [nx, ny, nz] = size(binaryModel);
    
    % 初始化
    localAnisotropyMap = zeros(nx, ny, nz);
    
    % 滑动窗口计算
    for x = 1:stride:nx-windowSize+1
        for y = 1:stride:ny-windowSize+1
            for z = 1:stride:nz-windowSize+1
                % 提取窗口
                x2 = min(x+windowSize-1, nx);
                y2 = min(y+windowSize-1, ny);
                z2 = min(z+windowSize-1, nz);
                window = binaryModel(x:x2, y:y2, z:z2);
                
                % 计算局部各向异性
                localAniso = computeQuickAnisotropy(window);
                
                % 填充区域
                localAnisotropyMap(x:x2, y:y2, z:z2) = localAniso;
            end
        end
    end
end
function curvatureMap = computeLocalCurvatureMap(binaryModel)
    % 计算局部曲率图
    % 使用高斯平滑和二阶导数估计曲率
    smoothed = imgaussfilt3(double(binaryModel), 2);
    
    [gx, gy, gz] = gradient(smoothed);
    [gxx, gxy, gxz] = gradient(gx);
    [~, gyy, gyz] = gradient(gy);
    [~, ~, gzz] = gradient(gz);
    
    % 平均曲率的简化计算
    curvatureMap = abs(gxx + gyy + gzz) ./ (sqrt(gx.^2 + gy.^2 + gz.^2) + eps);
end
function thicknessMap = computeLocalThicknessMap(binaryModel)
    % 计算局部厚度图
    % 使用距离变换的简化方法
    distMap = bwdist(~binaryModel);
    thicknessMap = distMap;
    
    % 平滑处理
    thicknessMap = imgaussfilt3(thicknessMap, 1);
end
function coherenceMap = computeStructureCoherenceMap(binaryModel)
    % 计算结构连贯性图
    % 基于局部结构张量
    [gx, gy, gz] = gradient(double(binaryModel));
    
    % 结构张量元素
    Jxx = imgaussfilt3(gx.^2, 2);
    Jyy = imgaussfilt3(gy.^2, 2);
    Jzz = imgaussfilt3(gz.^2, 2);
    Jxy = imgaussfilt3(gx.*gy, 2);
    Jxz = imgaussfilt3(gx.*gz, 2);
    Jyz = imgaussfilt3(gy.*gz, 2);
    
    % 连贯性度量（简化版）
    trace = Jxx + Jyy + Jzz;
    coherenceMap = trace ./ (max(trace(:)) + eps);
end
function mcmcModel = generateComprehensiveInitialModel(dims, targetPorosity, ...
    originalFeatures, spatialFeatures, morphologyFeatures, optParams)
    % 生成综合优化的初始模型
    fprintf('  生成综合优化的初始模型...');
    % 1. 基于空间相关性与各向异性生成基础场
    targetTPC = spatialFeatures.twoPointCorr;
    targetAnisotropy = spatialFeatures.anisotropy;
    correlationLength = estimateCorrelationLength(targetTPC);
    baseField = synthesizeDirectionalGaussianField(dims, correlationLength, ...
        targetAnisotropy, spatialFeatures, optParams);
    % 2. 构造多尺度密度引导场，并与基础场融合
    densityMap = constructMultiScaleDensityMap(optParams, dims);
    baseField = blendReferenceDirectionalPatterns(baseField, densityMap);
    % 3. 自适应阈值化以获得目标孔隙率
    threshold = computeAdaptiveThreshold(baseField, targetPorosity, optParams);
    model = baseField > threshold;
    model = reinforceDensityTargets(model, densityMap, targetPorosity);
    % 4. 形态与代表性簇嵌入
    model = applyTargetMorphology(model, morphologyFeatures);
    model = adjustClusterSizeDistribution(model, optParams);
    model = embedRepresentativeClusters(model, densityMap, optParams, morphologyFeatures);
    % 5. 匹配孔隙尺寸分布
    model = matchPoreSizeDistribution(model, originalFeatures, optParams);
    % 6. 多指标确定性优化
    nOptSteps = 40;
    for step = 1:nOptSteps
        currentSpatial = computeEnhancedSpatialFeatures(model);
        currentMorph = computeDetailedMorphologyFeatures(model);
        currentMultiScale = computeMultiScaleSpatialFeatures(model);
        spatialMatch = calculateSpatialMatch(currentSpatial, spatialFeatures);
        morphMatch = calculateMorphologyMatch(currentMorph, morphologyFeatures);
        multiScaleMatch = calculateMultiScaleMatch(currentMultiScale, optParams.multiScaleSpatialFeatures);
        if min([spatialMatch, morphMatch, multiScaleMatch]) > 0.92
            break;
        end
        [~, primaryTarget] = min([spatialMatch, morphMatch, multiScaleMatch]);
        switch primaryTarget
            case 1
                model = applySpatialCorrection(model, currentSpatial, spatialFeatures);
            case 2
                model = applyMorphologyCorrection(model, currentMorph, morphologyFeatures);
            case 3
                model = applyMultiScaleCorrection(model, currentMultiScale, optParams.multiScaleSpatialFeatures);
        end
        model = adjustPorosity(model, targetPorosity);
        model = reinforceDensityTargets(model, densityMap, targetPorosity);
    end
    % 7. 引入空间特征预优化MCMC
    fprintf('  执行空间特征预优化MCMC...');
    model = refineInitialModelWithSpatialMCMC(model, spatialFeatures, morphologyFeatures, ...
        targetPorosity, optParams);
    model = enforceTargetClusterCount(model, optParams);
    model = matchPoreSizeDistribution(model, originalFeatures, optParams);
    model = adjustPorosity(model, targetPorosity);
    fprintf('  初始模型生成完成');
    mcmcModel = model;
end
function field = synthesizeDirectionalGaussianField(dims, correlationLength, anisotropy, ...
    spatialFeatures, optParams)
    % 生成包含方向性与梯度的相关高斯场
    whiteNoise = randn(dims);
    [X, Y, Z] = meshgrid(-dims(1)/2:dims(1)/2-1, ...
        -dims(2)/2:dims(2)/2-1, ...
        -dims(3)/2:dims(3)/2-1);
    anisotropyFactors = [1 + anisotropy * 0.3, 1 + anisotropy * 0.3, 1 + anisotropy];
    sigma = max(1, correlationLength);
    kernel = exp(-(X.^2/(2*(sigma*anisotropyFactors(1))^2) + ...
        Y.^2/(2*(sigma*anisotropyFactors(2))^2) + ...
        Z.^2/(2*(sigma*anisotropyFactors(3))^2)));
    kernel = kernel / sum(kernel(:));
    fftNoise = fftn(whiteNoise);
    fftKernel = fftn(ifftshift(kernel));
    field = real(ifftn(fftNoise .* sqrt(abs(fftKernel))));
    field = (field - mean(field(:))) / (std(field(:)) + eps);
    if isfield(spatialFeatures, 'porosityGradient') && ~isempty(spatialFeatures.porosityGradient)
        gradientBias = spatialFeatures.porosityGradient;
        gradientField = (Z / max(1, max(abs(Z(:))))) * gradientBias;
        field = field + gradientField;
    end
    if nargin >= 5 && isfield(optParams, 'directionalPorosityProfile') && ...
            ~isempty(optParams.directionalPorosityProfile)
        profile = optParams.directionalPorosityProfile;
        if isfield(profile, 'z') && ~isempty(profile.z)
            zProfile = profile.z(:)';
            zProfile = zProfile / (max(zProfile) + eps);
            zWeights = interp1(linspace(-0.5, 0.5, numel(zProfile)), zProfile, ...
                linspace(-0.5, 0.5, dims(3)), 'linear', 'extrap');
            zWeights = reshape(zWeights, 1, 1, []);
            field = field + zWeights .* std(field(:)) * 0.3;
        end
    end
end
function densityMap = constructMultiScaleDensityMap(optParams, targetSize)
    % 构造多尺度密度引导图
    densityMap = [];
    if nargin < 1 || isempty(optParams)
        return;
    end
    reference = [];
    if isfield(optParams, 'referenceDensityMap') && ~isempty(optParams.referenceDensityMap)
        reference = optParams.referenceDensityMap;
    elseif isfield(optParams, 'originalBinaryModel') && ~isempty(optParams.originalBinaryModel)
        reference = double(optParams.originalBinaryModel);
    end
    if isempty(reference)
        densityMap = zeros(targetSize);
        return;
    end
    reference = double(reference);
    reference = reference / max(1e-6, max(reference(:)));
    reference = imgaussfilt3(reference, 1.2);
    densityMap = resizeVolume(reference, targetSize);
    if isfield(optParams, 'multiScaleSpatialFeatures') && ...
            isfield(optParams.multiScaleSpatialFeatures, 'scale')
        scales = optParams.multiScaleSpatialFeatures.scale;
        aggregated = zeros(targetSize);
        weightSum = 0;
        for s = 1:length(scales)
            scaleFactor = scales(s).scaleFactor;
            weight = 1 / scaleFactor;
            scaled = imgaussfilt3(densityMap, max(0.5, scaleFactor / 2));
            aggregated = aggregated + weight * scaled;
            weightSum = weightSum + weight;
        end
        if weightSum > 0
            densityMap = 0.6 * densityMap + 0.4 * (aggregated / weightSum);
        end
    end
    if isfield(optParams, 'directionalPorosityProfile')
        densityMap = applyDirectionalProfile(densityMap, optParams.directionalPorosityProfile);
    end
    densityMap = densityMap / max(densityMap(:) + eps);
end
function blendedField = blendReferenceDirectionalPatterns(baseField, densityMap)
    % 将参考密度模式与基础场融合
    if isempty(densityMap)
        blendedField = baseField;
        return;
    end
    normalizedMap = densityMap - mean(densityMap(:));
    blendedField = baseField + 0.5 * normalizedMap;
    lowFreq = imgaussfilt3(normalizedMap, 3);
    blendedField = blendedField + 0.3 * (lowFreq - mean(lowFreq(:)));
    blendedField = (blendedField - mean(blendedField(:))) / (std(blendedField(:)) + eps);
end
function threshold = computeAdaptiveThreshold(field, targetPorosity, optParams)
    % 根据方向性和目标孔隙率自适应确定阈值
    baseThreshold = quantile(field(:), 1 - targetPorosity);
    adjustment = 0;
    if nargin >= 3 && isfield(optParams, 'directionalPorosityProfile') && ...
            ~isempty(optParams.directionalPorosityProfile)
        profile = optParams.directionalPorosityProfile;
        variations = [];
        gradients = [];
        if isfield(profile, 'x') && ~isempty(profile.x)
            variations(end+1) = std(profile.x); %#ok<AGROW>
            gradients(end+1) = max(abs(diff(profile.x))); %#ok<AGROW>
        end
        if isfield(profile, 'y') && ~isempty(profile.y)
            variations(end+1) = std(profile.y); %#ok<AGROW>
            gradients(end+1) = max(abs(diff(profile.y))); %#ok<AGROW>
        end
        if isfield(profile, 'z') && ~isempty(profile.z)
            variations(end+1) = std(profile.z); %#ok<AGROW>
            gradients(end+1) = max(abs(diff(profile.z))); %#ok<AGROW>
        end
        if ~isempty(variations)
            adjustment = adjustment + mean(variations) * 0.05;
        end
        if ~isempty(gradients)
            adjustment = adjustment + mean(gradients) * 0.02;
        end
    end
    threshold = baseThreshold - adjustment;
    threshold = min(max(threshold, min(field(:))), max(field(:)));
end
function model = reinforceDensityTargets(model, densityMap, targetPorosity)
    % 根据密度引导图强化局部孔隙分布
    if isempty(densityMap)
        return;
    end
    normalized = densityMap / max(densityMap(:) + eps);
    totalVoxels = numel(model);
    desiredCount = round(totalVoxels * targetPorosity);
    [~, order] = sort(normalized(:), 'descend');
    desiredMask = false(size(model));
    desiredMask(order(1:min(desiredCount, numel(order)))) = true;
    removalMask = (~desiredMask) & model;
    removalChance = 0.45 - 0.4 * normalized;
    removalChance = max(0.05, min(0.9, removalChance));
    randField = rand(size(model));
    removalSelection = false(size(model));
    removalSelection(removalMask) = randField(removalMask) < removalChance(removalMask);
    model(removalSelection) = false;
    additionMask = desiredMask & ~model;
    additionChance = 0.35 + 0.5 * normalized;
    additionChance = max(0.05, min(0.95, additionChance));
    randField = rand(size(model));
    additionSelection = false(size(model));
    additionSelection(additionMask) = randField(additionMask) < additionChance(additionMask);
    model(additionSelection) = true;
end
function match = calculateMultiScaleMatch(currentMultiScale, targetMultiScale)
    % 计算多尺度特征匹配度
    if nargin < 2 || isempty(currentMultiScale) || isempty(targetMultiScale)
        match = 0.5;
        return;
    end
    match = 0;
    terms = 0;
    if isfield(currentMultiScale, 'scale') && isfield(targetMultiScale, 'scale')
        nScales = min(length(currentMultiScale.scale), length(targetMultiScale.scale));
        for s = 1:nScales
            cScale = currentMultiScale.scale(s);
            tScale = targetMultiScale.scale(s);
            if isfield(cScale, 'twoPointCorr') && isfield(tScale, 'twoPointCorr')
                minRows = min(size(cScale.twoPointCorr, 1), size(tScale.twoPointCorr, 1));
                if minRows > 0
                    diffVal = mean(abs(cScale.twoPointCorr(1:minRows, :) - ...
                        tScale.twoPointCorr(1:minRows, :)), 'all');
                    match = match + (1 - min(diffVal, 1));
                    terms = terms + 1;
                end
            end
            if isfield(cScale, 'clusterDistribution') && ...
                    isfield(cScale.clusterDistribution, 'density') && ...
                    isfield(tScale, 'clusterDistribution') && ...
                    isfield(tScale.clusterDistribution, 'density')
                diffVal = abs(cScale.clusterDistribution.density - ...
                    tScale.clusterDistribution.density);
                match = match + (1 - min(diffVal * 50, 1));
                terms = terms + 1;
            end
            if isfield(cScale, 'spatialSpectrum') && isfield(tScale, 'spatialSpectrum')
                minLen = min(length(cScale.spatialSpectrum), length(tScale.spatialSpectrum));
                if minLen > 0
                    diffVal = mean(abs(cScale.spatialSpectrum(1:minLen) - ...
                        tScale.spatialSpectrum(1:minLen)));
                    match = match + (1 - min(diffVal / (max(tScale.spatialSpectrum(:)) + eps), 1));
                    terms = terms + 1;
                end
            end
        end
    end
    if isfield(currentMultiScale, 'scaleInvariantFeatures') && ...
            isfield(targetMultiScale, 'scaleInvariantFeatures')
        cInv = currentMultiScale.scaleInvariantFeatures;
        tInv = targetMultiScale.scaleInvariantFeatures;
        if isfield(cInv, 'fractalDimension') && isfield(tInv, 'fractalDimension')
            diffVal = abs(cInv.fractalDimension - tInv.fractalDimension);
            match = match + (1 - min(diffVal, 1));
            terms = terms + 1;
        end
    end
    if terms == 0
        match = 0.5;
    else
        match = max(0, min(1, match / terms));
    end
end
function model = applyMultiScaleCorrection(model, currentMultiScale, targetMultiScale)
    % 针对多尺度差异进行纠正
    if nargin < 3 || isempty(currentMultiScale) || isempty(targetMultiScale)
        return;
    end
    corrected = model;
    nScales = min(length(currentMultiScale.scale), length(targetMultiScale.scale));
    for s = 1:nScales
        cScale = currentMultiScale.scale(s);
        tScale = targetMultiScale.scale(s);
        scaleFactor = max(1, round(cScale.scaleFactor));
        desiredDensity = 0;
        currentDensity = 0;
        if isfield(tScale, 'clusterDistribution') && isfield(tScale.clusterDistribution, 'density')
            desiredDensity = tScale.clusterDistribution.density;
        end
        if isfield(cScale, 'clusterDistribution') && isfield(cScale.clusterDistribution, 'density')
            currentDensity = cScale.clusterDistribution.density;
        end
        diffVal = desiredDensity - currentDensity;
        if abs(diffVal) < 1e-4
            continue;
        end
        radius = max(1, round(scaleFactor / 2));
        se = strel('sphere', radius);
        if diffVal > 0
            candidate = imdilate(corrected, se);
            addMask = candidate & ~corrected;
            prob = min(0.6, diffVal * 8);
            if prob <= 0
                continue;
            end
            addIdx = find(addMask);
            if isempty(addIdx)
                continue;
            end
            if prob >= 1
                selectedIdx = addIdx;
            else
                selection = rand(numel(addIdx), 1) < prob;
                selectedIdx = addIdx(selection);
            end
            if ~isempty(selectedIdx)
                corrected(selectedIdx) = true;
            end
        else
            candidate = imerode(corrected, se);
            removeMask = corrected & ~candidate;
            prob = min(0.6, -diffVal * 8);
            if prob <= 0
                continue;
            end
            removeIdx = find(removeMask);
            if isempty(removeIdx)
                continue;
            end
            if prob >= 1
                selectedIdx = removeIdx;
            else
                selection = rand(numel(removeIdx), 1) < prob;
                selectedIdx = removeIdx(selection);
            end
            if ~isempty(selectedIdx)
                corrected(selectedIdx) = false;
            end
        end
    end
    model = corrected;
end
function model = embedRepresentativeClusters(model, densityMap, optParams, morphologyFeatures)
    % 将代表性簇嵌入到初始模型中
    if nargin < 4
        morphologyFeatures = struct();
    end
    library = {};
    if isfield(optParams, 'clusterLibrary') && ~isempty(optParams.clusterLibrary)
        library = optParams.clusterLibrary;
    elseif isfield(optParams, 'originalBinaryModel') && ~isempty(optParams.originalBinaryModel)
        library = sampleRepresentativeClusters(optParams.originalBinaryModel, morphologyFeatures);
    end
    if isempty(library)
        return;
    end
    targetCount = [];
    if isfield(optParams, 'targetClusterCount') && ~isempty(optParams.targetClusterCount)
        targetCount = max(0, round(optParams.targetClusterCount));
    end
    CC = bwconncomp(model, 26);
    currentCount = CC.NumObjects;
    if ~isempty(targetCount)
        missing = max(0, targetCount - currentCount);
    else
        missing = max(1, round(length(library) * 0.2));
    end
    if missing <= 0
        return;
    end
    nEmbed = min(length(library), max(1, missing));
    for i = 1:nEmbed
        idx = 1 + mod(i-1, length(library));
        model = placeClusterSample(model, library{idx}, densityMap);
    end
end
function model = placeClusterSample(model, clusterSample, densityMap)
    % 将单个代表簇放置到模型中
    if ~isstruct(clusterSample) || ~isfield(clusterSample, 'patch') || isempty(clusterSample.patch)
        return;
    end
    patch = clusterSample.patch;
    [px, py, pz] = size(patch);
    [nx, ny, nz] = size(model);
    if px > nx || py > ny || pz > nz
        return;
    end
    maxAttempts = 30;
    attempt = 0;
    while attempt < maxAttempts
        attempt = attempt + 1;
        if ~isempty(densityMap)
            weights = densityMap(:) + eps;
            weights = weights / sum(weights);
            r = rand();
            cdf = cumsum(weights);
            idx = find(cdf >= r, 1, 'first');
            [cx, cy, cz] = ind2sub(size(densityMap), idx);
            x = max(1, min(nx - px + 1, cx - round(px/2)));
            y = max(1, min(ny - py + 1, cy - round(py/2)));
            z = max(1, min(nz - pz + 1, cz - round(pz/2)));
        else
            x = randi([1, nx - px + 1]);
            y = randi([1, ny - py + 1]);
            z = randi([1, nz - pz + 1]);
        end
        region = model(x:x+px-1, y:y+py-1, z:z+pz-1);
        overlapRatio = sum(region(:) & patch(:)) / max(1, clusterSample.size);
        if overlapRatio < 0.6
            combined = region | patch;
            model(x:x+px-1, y:y+py-1, z:z+pz-1) = combined;
            break;
        end
    end
end
function model = matchPoreSizeDistribution(model, originalFeatures, optParams)
    % 使模型的孔隙尺寸分布更接近原始数据
    if nargin < 2 || isempty(originalFeatures) || ~isfield(originalFeatures, 'sizes')
        return;
    end
    targetHist = computeTargetPoreHistogram(originalFeatures.sizes);
    CC = bwconncomp(model, 26);
    if CC.NumObjects == 0
        return;
    end
    currentSizes = cellfun(@numel, CC.PixelIdxList);
    currentHist = computeTargetPoreHistogram(currentSizes);
    diffHist = targetHist - currentHist;
    if all(abs(diffHist) < 1e-3)
        return;
    end
    edges = linspace(0, 1, numel(targetHist));
    for i = 1:length(diffHist)
        diffVal = diffHist(i);
        if abs(diffVal) < 1e-3
            continue;
        end
        radius = max(1, round(1 + edges(i) * 5));
        se = strel('sphere', radius);
        if diffVal > 0
            expanded = imdilate(model, se);
            addMask = expanded & ~model;
            prob = min(0.5, diffVal * 5);
            if prob > 0 && any(addMask(:))
                addIdx = find(addMask);
                randVals = rand(numel(addIdx), 1);
                chosen = addIdx(randVals < prob);
                if ~isempty(chosen)
                    selectionMask = false(size(model));
                    selectionMask(chosen) = true;
                    model(selectionMask) = true;
                end
            end
        else
            eroded = imerode(model, se);
            removeMask = model & ~eroded;
            prob = min(0.5, -diffVal * 5);
            if prob > 0 && any(removeMask(:))
                removeIdx = find(removeMask);
                randVals = rand(numel(removeIdx), 1);
                chosen = removeIdx(randVals < prob);
                if ~isempty(chosen)
                    selectionMask = false(size(model));
                    selectionMask(chosen) = true;
                    model(selectionMask) = false;
                end
            end
        end
    end
    if nargin >= 3 && isfield(optParams, 'targetPorosity')
        model = adjustPorosity(model, optParams.targetPorosity);
    end
end
function histVals = computeTargetPoreHistogram(clusterSizes)
    % 计算归一化的孔隙尺寸直方图
    if isempty(clusterSizes)
        histVals = zeros(6, 1);
        return;
    end
    normalized = clusterSizes / max(clusterSizes);
    edges = linspace(0, 1, 7);
    histVals = histcounts(normalized, edges, 'Normalization', 'probability');
    histVals = histVals(:);
    if all(histVals == 0)
        histVals = ones(size(histVals)) / numel(histVals);
    end
end
function profile = computeDirectionalPorosityProfile(model)
    % 计算模型在三个方向上的孔隙率分布及梯度
    profile = struct('x', [], 'y', [], 'z', []);
    if nargin == 0 || isempty(model)
        return;
    end
    profile.x = squeeze(mean(mean(model, 2), 3));
    profile.y = squeeze(mean(mean(model, 1), 3));
    profile.z = squeeze(mean(mean(model, 1), 2));
    profile.x = profile.x(:);
    profile.y = profile.y(:);
    profile.z = profile.z(:);
    profile.meanPorosity = mean(model(:));
    profile.variance = [std(profile.x), std(profile.y), std(profile.z)];
    profile.gradientStrength = [max(abs(diff(profile.x))), ...
        max(abs(diff(profile.y))), max(abs(diff(profile.z)))];
end
function densityMap = constructReferenceDensityMap(referenceModel, multiScaleSpatialFeatures)
    % 基于参考模型生成多尺度密度图
    densityMap = [];
    if nargin < 1 || isempty(referenceModel)
        return;
    end
    densityMap = double(referenceModel);
    densityMap = densityMap / max(densityMap(:) + eps);
    densityMap = imgaussfilt3(densityMap, 1.0);
    if nargin > 1 && ~isempty(multiScaleSpatialFeatures) && ...
            isfield(multiScaleSpatialFeatures, 'scale')
        aggregated = zeros(size(densityMap));
        weightSum = 0;
        for s = 1:length(multiScaleSpatialFeatures.scale)
            scaleFactor = multiScaleSpatialFeatures.scale(s).scaleFactor;
            weight = 1 / max(1, scaleFactor);
            aggregated = aggregated + weight * imgaussfilt3(densityMap, max(0.5, scaleFactor/2));
            weightSum = weightSum + weight;
        end
        if weightSum > 0
            densityMap = 0.5 * densityMap + 0.5 * (aggregated / weightSum);
        end
    end
    densityMap = densityMap / max(densityMap(:) + eps);
end
function resized = resizeVolume(volume, targetSize)
    % 使用线性插值调整体数据大小
    if nargin < 2 || isempty(targetSize)
        resized = volume;
        return;
    end
    if all(size(volume) == targetSize)
        resized = volume;
        return;
    end
    [nx, ny, nz] = size(volume);
    tx = targetSize(1); ty = targetSize(2); tz = targetSize(3);
    [Xq, Yq, Zq] = ndgrid(linspace(1, nx, tx), linspace(1, ny, ty), linspace(1, nz, tz));
    F = griddedInterpolant({1:nx, 1:ny, 1:nz}, double(volume), 'linear', 'nearest');
    resized = F(Xq, Yq, Zq);
end
function densityMap = applyDirectionalProfile(densityMap, profile)
    % 将方向性孔隙率分布施加到密度图上
    if isempty(densityMap) || nargin < 2 || isempty(profile)
        return;
    end
    [nx, ny, nz] = size(densityMap);
    if isfield(profile, 'x') && ~isempty(profile.x)
        xProfile = interp1(linspace(0, 1, numel(profile.x)), profile.x, ...
            linspace(0, 1, nx), 'linear', 'extrap');
        xWeights = reshape(xProfile / (max(xProfile) + eps), [nx, 1, 1]);
        densityMap = densityMap .* (0.5 + 0.5 * xWeights);
    end
    if isfield(profile, 'y') && ~isempty(profile.y)
        yProfile = interp1(linspace(0, 1, numel(profile.y)), profile.y, ...
            linspace(0, 1, ny), 'linear', 'extrap');
        yWeights = reshape(yProfile / (max(yProfile) + eps), [1, ny, 1]);
        densityMap = densityMap .* (0.5 + 0.5 * yWeights);
    end
    if isfield(profile, 'z') && ~isempty(profile.z)
        zProfile = interp1(linspace(0, 1, numel(profile.z)), profile.z, ...
            linspace(0, 1, nz), 'linear', 'extrap');
        zWeights = reshape(zProfile / (max(zProfile) + eps), [1, 1, nz]);
        densityMap = densityMap .* (0.5 + 0.5 * zWeights);
    end
    densityMap = densityMap / max(densityMap(:) + eps);
end
function clusters = sampleRepresentativeClusters(referenceModel, morphologyFeatures)
    % 从参考模型中提取代表性簇模板
    clusters = {};
    if nargin < 1 || isempty(referenceModel)
        return;
    end
    CC = bwconncomp(referenceModel, 26);
    if CC.NumObjects == 0
        return;
    end
    sizes = cellfun(@numel, CC.PixelIdxList);
    [~, order] = sort(sizes, 'descend');
    nSelect = min(max(5, round(CC.NumObjects * 0.15)), min(25, CC.NumObjects));
    clusters = cell(nSelect, 1);
    for i = 1:nSelect
        idx = order(i);
        voxels = CC.PixelIdxList{idx};
        [x, y, z] = ind2sub(size(referenceModel), voxels);
        xmin = min(x); xmax = max(x);
        ymin = min(y); ymax = max(y);
        zmin = min(z); zmax = max(z);
        patchSize = [xmax - xmin + 1, ymax - ymin + 1, zmax - zmin + 1];
        patch = false(patchSize);
        localIdx = sub2ind(patchSize, x - xmin + 1, y - ymin + 1, z - zmin + 1);
        patch(localIdx) = true;
        clusters{i} = struct('patch', patch, 'size', sizes(idx), ...
            'bbox', [xmin, xmax, ymin, ymax, zmin, zmax]);
    end
    % 追加一些较小的簇以丰富多样性
    if CC.NumObjects > nSelect
        remaining = setdiff(1:CC.NumObjects, order(1:nSelect));
        extra = remaining(randperm(length(remaining), min(5, numel(remaining))));
        for i = 1:length(extra)
            idx = extra(i);
            voxels = CC.PixelIdxList{idx};
            [x, y, z] = ind2sub(size(referenceModel), voxels);
            xmin = min(x); xmax = max(x);
            ymin = min(y); ymax = max(y);
            zmin = min(z); zmax = max(z);
            patchSize = [xmax - xmin + 1, ymax - ymin + 1, zmax - zmin + 1];
            patch = false(patchSize);
            localIdx = sub2ind(patchSize, x - xmin + 1, y - ymin + 1, z - zmin + 1);
            patch(localIdx) = true;
            clusters{end+1,1} = struct('patch', patch, 'size', numel(voxels), ...
                'bbox', [xmin, xmax, ymin, ymax, zmin, zmax]);
        end
    end
    if nargin > 1 && ~isempty(morphologyFeatures) && isfield(morphologyFeatures, 'sphericity')
        % 根据参考球形度对模板进行排序，优先使用接近平均值的簇
        targetSphericity = mean(morphologyFeatures.sphericity);
        scores = zeros(length(clusters), 1);
        for i = 1:length(clusters)
            patch = clusters{i}.patch;
            if any(patch(:))
                try
                    props = regionprops3(patch, 'PrincipalAxisLength');
                    if ~isempty(props)
                        axes = sort(props.PrincipalAxisLength, 'descend');
                        sph = axes(3) / (axes(1) + eps);
                        scores(i) = abs(sph - targetSphericity);
                    else
                        scores(i) = inf;
                    end
                catch
                    localPorosity = mean(patch(:));
                    scores(i) = abs(localPorosity - targetSphericity);
                end
            else
                scores(i) = inf;
            end
        end
        [~, order] = sort(scores);
        clusters = clusters(order);
    end
end
function corrLength = estimateCorrelationLength(twoPointCorr)
    % 估计相关长度
    % 找到相关函数下降到1/e的距离
    if isempty(twoPointCorr)
        corrLength = 5; % 默认值
        return;
    end
    
    % 使用平均相关函数
    meanCorr = mean(twoPointCorr, 2);
    if isempty(meanCorr) || meanCorr(1) == 0
        corrLength = 5;
        return;
    end
    
    threshold = meanCorr(1) / exp(1);
    
    % 找到第一个低于阈值的点
    idx = find(meanCorr < threshold, 1);
    if isempty(idx)
        corrLength = size(twoPointCorr, 1) * 2;
    else
        % 插值获得更准确的相关长度
        if idx > 1
            % 线性插值
            y1 = meanCorr(idx-1);
            y2 = meanCorr(idx);
            x1 = idx - 1;
            x2 = idx;
            corrLength = x1 + (threshold - y1) * (x2 - x1) / (y2 - y1);
        else
            corrLength = idx;
        end
    end
    
    % 确保合理范围
    corrLength = max(1, min(corrLength, 20));
end
function filteredField = anisotropicFilter3D(field, anisotropy)
    % 3D各向异性滤波
    % 根据各向异性程度调整滤波器
    if anisotropy > 0.5
        % 高各向异性：Z方向较弱的滤波
        hx = gausswin(5, 2.5);
        hy = gausswin(5, 2.5);
        hz = gausswin(3, 1.5);
    else
        % 低各向异性：各向同性滤波
        hx = gausswin(5, 2.5);
        hy = gausswin(5, 2.5);
        hz = gausswin(5, 2.5);
    end
    
    % 分离滤波
    filteredField = field;
    for i = 1:size(field, 3)
        filteredField(:,:,i) = conv2(filteredField(:,:,i), hx*hy', 'same');
    end
    
    for i = 1:size(field, 1)
        for j = 1:size(field, 2)
            filteredField(i,j,:) = conv(squeeze(filteredField(i,j,:)), hz, 'same');
        end
    end
end
function model = applyTargetMorphology(model, morphologyFeatures)
    % 应用目标形态特征
    if ~isempty(morphologyFeatures.sphericity)
        targetSphericity = mean(morphologyFeatures.sphericity);
        
        if targetSphericity > 0.7
            % 高球形度：使用球形结构元素
            se = strel('sphere', 2);
            model = imopen(model, se);
            model = imclose(model, se);
        elseif targetSphericity < 0.4
            % 低球形度：使用方向性结构元素
            for angle = 0:60:120
                se = strel('line', 5, angle);
                model = imopen(model, se);
            end
        else
            % 中等球形度：轻微形态学操作
            se = strel('cube', 2);
            model = imopen(model, se);
        end
    end
end
function model = adjustClusterSizeDistribution(model, optParams)
    % 调整簇大小分布
    CC = bwconncomp(model, 26);
    if CC.NumObjects == 0
        return;
    end
    sizes = cellfun(@numel, CC.PixelIdxList);
    % 处理极端大小的簇
    maxAllowed = optParams.targetMax * 3;
    minAllowed = max(5, optParams.targetMin / 10);
    % 缩减过大的簇
    largeClusters = find(sizes > maxAllowed);
    for i = 1:length(largeClusters)
        idx = largeClusters(i);
        fprintf('    调整大簇尺寸 (大小=%d)...\n', sizes(idx));
        model = reduceOversizedCluster(model, CC.PixelIdxList{idx}, optParams);
    end
    % 处理过小的簇
    tinyClusters = find(sizes < minAllowed);
    if ~isempty(tinyClusters)
        fprintf('    移除或扩展%d个过小的簇...\n', length(tinyClusters));
        for i = 1:length(tinyClusters)
            clusterIdx = tinyClusters(i);
            if isfield(optParams, 'preserveSmallPores') && optParams.preserveSmallPores
                model = growSmallCluster(model, CC.PixelIdxList{clusterIdx});
            else
                model(CC.PixelIdxList{clusterIdx}) = false;
            end
        end
    end
end
function model = reduceOversizedCluster(model, clusterIdx, optParams)
    % 通过平滑收缩的方式减少大簇体量
    clusterMask = false(size(model));
    clusterMask(clusterIdx) = true;
    currentSize = numel(clusterIdx);
    targetSize = max(1, optParams.targetMax);
    if currentSize <= targetSize
        return;
    end
    distMap = bwdist(~clusterMask);
    reductionRatio = min(0.45, max(0, (currentSize - targetSize) / currentSize));
    distances = distMap(clusterMask);
    cutoff = quantile(distances, reductionRatio);
    removalMask = clusterMask & (distMap <= cutoff);
    refined = clusterMask;
    refined(removalMask) = false;
    % 轻度平滑以保持连通性
    se = strel('sphere', 1);
    refined = imclose(refined, se);
    % 防止完全消失
    if ~any(refined(:))
        refined = clusterMask;
    end
    model(clusterIdx) = false;
    model = model | refined;
end
function model = growSmallCluster(model, clusterIdx)
    % 通过局部膨胀扩展小簇，避免直接删除
    clusterMask = false(size(model));
    clusterMask(clusterIdx) = true;
    se = strel('sphere', 1);
    grown = imdilate(clusterMask, se);
    model(clusterIdx) = false;
    model = model | grown;
end
function model = applySpatialCorrection(model, currentSpatial, targetSpatial)
    % 应用空间修正
    % 比较两点相关函数
    if isfield(currentSpatial, 'twoPointCorr') && isfield(targetSpatial, 'twoPointCorr')
        currentTPC = mean(currentSpatial.twoPointCorr, 2);
        targetTPC = mean(targetSpatial.twoPointCorr, 2);
        
        % 确保尺寸匹配
        minLen = min(length(currentTPC), length(targetTPC));
        if minLen > 0
            tpcDiff = currentTPC(1:minLen) - targetTPC(1:minLen);
            
            if mean(tpcDiff) > 0
                % 当前相关性太高，增加随机性
                nFlip = round(0.001 * numel(model));
                flipIdx = randperm(numel(model), nFlip);
                model(flipIdx) = ~model(flipIdx);
            else
                % 当前相关性太低，增加聚集
                boundary = bwperim(model, 26);
                growthCandidates = find(boundary);
                
                if ~isempty(growthCandidates)
                    nGrow = min(round(0.001 * numel(model)), length(growthCandidates));
                    growIdx = growthCandidates(randperm(length(growthCandidates), nGrow));
                    
                    for i = 1:length(growIdx)
                        [x, y, z] = ind2sub(size(model), growIdx(i));
                        
                        % 在邻域生长
                        for dx = -1:1
                            for dy = -1:1
                                for dz = -1:1
                                    nx = x + dx;
                                    ny = y + dy;
                                    nz = z + dz;
                                    if nx >= 1 && nx <= size(model, 1) && ...
                                        ny >= 1 && ny <= size(model, 2) && ...
                                        nz >= 1 && nz <= size(model, 3)
                                        if ~model(nx, ny, nz) && rand() < 0.3
                                            model(nx, ny, nz) = true;
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
function model = applyMorphologyCorrection(model, currentMorph, targetMorph)
    % 应用形态学修正
    % 根据当前和目标形态特征的差异进行调整
    
    % 球形度修正
    if ~isempty(currentMorph.sphericity) && ~isempty(targetMorph.sphericity)
        currentSpher = mean(currentMorph.sphericity);
        targetSpher = mean(targetMorph.sphericity);
        
        if abs(currentSpher - targetSpher) > 0.1
            if targetSpher > currentSpher
                % 需要增加球形度 - 使用形态学闭操作
                se = strel('sphere', 1);
                model = imclose(model, se);
            else
                % 需要减少球形度 - 使用定向腐蚀
                se = strel('line', 3, 0);
                model = imerode(model, se);
                se = strel('line', 3, 90);
                model = imdilate(model, se);
            end
        end
    end
    
    % 伸长率修正
    if ~isempty(currentMorph.elongation) && ~isempty(targetMorph.elongation)
        currentElong = mean(currentMorph.elongation);
        targetElong = mean(targetMorph.elongation);
        
        if abs(currentElong - targetElong) > 0.5
            if targetElong > currentElong
                % 需要增加伸长率 - 使用方向性操作
                % 沿主轴方向膨胀
                for angle = 0:45:90
                    se = strel('line', 2, angle);
                    model = imdilate(model, se);
                end
                
                % 横向腐蚀
                se = strel('disk', 1);
                for z = 1:size(model, 3)
                    model(:,:,z) = imerode(model(:,:,z), se);
                end
            else
                % 需要减少伸长率 - 各向同性操作
                se = strel('cube', 2);
                model = imclose(model, se);
            end
        end
    end
end
function model = adjustPorosity(model, targetPorosity)
    % 调整孔隙率到目标值
    currentPorosity = mean(model(:));
    diff = targetPorosity - currentPorosity;
    if abs(diff) < 0.001
        return;
    end
    if diff > 0
        % 需要增加孔隙
        boundary = bwperim(model, 26);
        candidates = find(boundary & ~model);
        if ~isempty(candidates)
            nAdd = round(abs(diff) * numel(model));
            nAdd = min(nAdd, length(candidates));
            addIdx = candidates(randperm(length(candidates), nAdd));
            model(addIdx) = true;
        end
    else
        % 需要减少孔隙
        boundary = bwperim(model, 26);
        candidates = find(boundary & model);
        if ~isempty(candidates)
            nRemove = round(abs(diff) * numel(model));
            nRemove = min(nRemove, length(candidates));
            removeIdx = candidates(randperm(length(candidates), nRemove));
            model(removeIdx) = false;
        end
    end
end
function model = refineInitialModelWithSpatialMCMC(model, targetSpatial, targetMorph, targetPorosity, optParams)
    % 使用MCMC在初始化阶段细化模型，使空间特征与原始模型更匹配
    maxIterations = 200;
    temperature = 0.5;
    coolingRate = 0.97;
    bestModel = model;
    currentEnergy = computeInitialMCMCEnergy(model, targetSpatial, targetMorph, targetPorosity);
    bestEnergy = currentEnergy;
    for iter = 1:maxIterations
        move = proposeClusterNeighborhoodMove(model, targetPorosity, optParams);
        if isempty(move.linearIdx)
            continue;
        end
        candidateModel = applyClusterNeighborhoodMove(model, move);
        candidateEnergy = computeInitialMCMCEnergy(candidateModel, targetSpatial, targetMorph, targetPorosity);
        deltaE = candidateEnergy - currentEnergy;
        if deltaE < 0 || rand() < exp(-deltaE / max(temperature, eps))
            model = candidateModel;
            currentEnergy = candidateEnergy;
            if currentEnergy < bestEnergy
                bestEnergy = currentEnergy;
                bestModel = model;
            end
        end
        temperature = max(temperature * coolingRate, 0.05);
    end
    model = bestModel;
    % 记录预优化后的匹配度，便于跟踪初始化质量
    refinedSpatial = computeEnhancedSpatialFeatures(model);
    refinedMorph = computeDetailedMorphologyFeatures(model);
    fprintf('    预优化匹配度 -> 空间: %.3f, 形态: %.3f', ...
        calculateSpatialMatch(refinedSpatial, targetSpatial), ...
        calculateMorphologyMatch(refinedMorph, targetMorph));
end
function energy = computeInitialMCMCEnergy(model, targetSpatial, targetMorph, targetPorosity)
    % 计算初始化阶段MCMC的能量函数，强调空间特征匹配
    currentSpatial = computeEnhancedSpatialFeatures(model);
    currentMorph = computeDetailedMorphologyFeatures(model);
    spatialMismatch = 1 - calculateSpatialMatch(currentSpatial, targetSpatial);
    morphMismatch = 1 - calculateMorphologyMatch(currentMorph, targetMorph);
    porosityMismatch = abs(mean(model(:)) - targetPorosity);
    energy = 0.65 * spatialMismatch + 0.25 * morphMismatch + 0.1 * porosityMismatch;
end
function move = proposeClusterNeighborhoodMove(model, targetPorosity, optParams)
    % 生成围绕孔隙簇的候选移动，使MCMC集中在真实结构附近
    if nargin < 3
        optParams = struct();
    end
    move = struct();
    move.linearIdx = [];
    move.oldValues = [];
    move.newValues = [];
    boundary = bwperim(model, 26);
    if ~any(boundary(:))
        return;
    end
    % 根据参数选择邻域半径
    radius = 2;
    if isfield(optParams, 'clusterPreMcmcRadius')
        radius = max(1, round(optParams.clusterPreMcmcRadius));
    end
    neighborhoodMask = imdilate(boundary, strel('sphere', radius));
    candidateIdx = find(neighborhoodMask);
    if isempty(candidateIdx)
        return;
    end
    centerIdx = candidateIdx(randi(length(candidateIdx)));
    [cx, cy, cz] = ind2sub(size(model), centerIdx);
    localIdx = [];
    for dx = -radius:radius
        for dy = -radius:radius
            for dz = -radius:radius
                x = cx + dx;
                y = cy + dy;
                z = cz + dz;
                if x >= 1 && x <= size(model, 1) && ...
                        y >= 1 && y <= size(model, 2) && ...
                        z >= 1 && z <= size(model, 3)
                    if sqrt(double(dx)^2 + double(dy)^2 + double(dz)^2) <= radius + 0.01
                        localIdx(end+1, 1) = sub2ind(size(model), x, y, z); %#ok<AGROW>
                    end
                end
            end
        end
    end
    localIdx = unique(localIdx);
    if isempty(localIdx)
        return;
    end
    localValues = model(localIdx);
    poreIdx = localIdx(localValues);
    matrixIdx = localIdx(~localValues);
    if isempty(poreIdx) && isempty(matrixIdx)
        return;
    end
    currentPorosity = mean(model(:));
    porosityDiff = targetPorosity - currentPorosity;
    nPairs = max(1, round(0.15 * min(numel(poreIdx), numel(matrixIdx))));
    nAdd = nPairs;
    nRemove = nPairs;
    if porosityDiff > 0 && ~isempty(matrixIdx)
        extra = min(numel(matrixIdx) - nPairs, max(0, round(porosityDiff * numel(localIdx))));
        nAdd = nAdd + extra;
    elseif porosityDiff < 0 && ~isempty(poreIdx)
        extra = min(numel(poreIdx) - nPairs, max(0, round(-porosityDiff * numel(localIdx))));
        nRemove = nRemove + extra;
    end
    if ~isempty(matrixIdx)
        nAdd = min(nAdd, numel(matrixIdx));
        addIdx = matrixIdx(randperm(numel(matrixIdx), nAdd));
        move.linearIdx = [move.linearIdx; addIdx(:)]; %#ok<AGROW>
        move.oldValues = [move.oldValues; false(numel(addIdx), 1)]; %#ok<AGROW>
        move.newValues = [move.newValues; true(numel(addIdx), 1)]; %#ok<AGROW>
    end
    if ~isempty(poreIdx)
        nRemove = min(nRemove, numel(poreIdx));
        removeIdx = poreIdx(randperm(numel(poreIdx), nRemove));
        move.linearIdx = [move.linearIdx; removeIdx(:)]; %#ok<AGROW>
        move.oldValues = [move.oldValues; true(numel(removeIdx), 1)]; %#ok<AGROW>
        move.newValues = [move.newValues; false(numel(removeIdx), 1)]; %#ok<AGROW>
    end
    if ~isempty(move.linearIdx)
        [move.linearIdx, uniqueIdx] = unique(move.linearIdx, 'stable');
        move.oldValues = move.oldValues(uniqueIdx);
        move.newValues = move.newValues(uniqueIdx);
    end
end
function newModel = applyClusterNeighborhoodMove(model, move)
    % 将簇邻域移动应用到模型
    newModel = model;
    if ~isfield(move, 'linearIdx') || isempty(move.linearIdx)
        return;
    end
    valid = move.linearIdx >= 1 & move.linearIdx <= numel(model);
    if ~all(valid)
        move.linearIdx = move.linearIdx(valid);
        move.newValues = move.newValues(valid);
    end
    if isempty(move.linearIdx)
        return;
    end
    newModel(move.linearIdx) = move.newValues;
end
%% ========== MCMC状态和控制函数 ==========
function mcmcState = initializeComprehensiveMCMCState(model, optParams, weights)
    % 初始化综合MCMC状态
    mcmcState = struct();
    mcmcState.model = model;
    mcmcState.bestModel = model;
    mcmcState.temperature = 0.4;
    
    % 存储优化参数和权重
    mcmcState.optParams = optParams;
    mcmcState.weights = weights;
    
    % 计算初始特征
    mcmcState.features = extractEfficientClusterFeatures(model);
    mcmcState.spatialFeatures = computeEnhancedSpatialFeatures(model);
    mcmcState.morphologyFeatures = computeDetailedMorphologyFeatures(model);
    mcmcState.multiScaleSpatialFeatures = computeMultiScaleSpatialFeatures(model);
    
    % 计算初始能量
    mcmcState.currentEnergy = calculateComprehensiveEnergy(mcmcState);
    mcmcState.bestEnergy = mcmcState.currentEnergy;
    
    % 性能跟踪
    mcmcState.acceptanceRate = 0.3;
    mcmcState.moveHistory = zeros(1, 10);
    mcmcState.energyTrend = zeros(1, 10);
    
    % 优化历史
    mcmcState.optimizationHistory = struct();
    mcmcState.optimizationHistory.morphologyMatches = [];
    mcmcState.optimizationHistory.spatialMatches = [];
    mcmcState.optimizationHistory.phaseTransitions = [];
end
function performanceMonitor = initializeComprehensivePerformanceMonitor(maxIterations)
    % 初始化综合性能监控器
    performanceMonitor = struct();
    performanceMonitor.energyHistory = zeros(1, maxIterations);
    performanceMonitor.acceptanceRatio = zeros(1, floor(maxIterations/100));
    performanceMonitor.timePerIteration = zeros(1, floor(maxIterations/100));
    performanceMonitor.spatialMatchHistory = zeros(1, floor(maxIterations/100));
    performanceMonitor.morphologyMatchHistory = zeros(1, floor(maxIterations/100));
    performanceMonitor.multiScaleMatchHistory = zeros(1, floor(maxIterations/100));
    performanceMonitor.anisotropyHistory = zeros(1, floor(maxIterations/100));
    performanceMonitor.connectivityHistory = zeros(1, floor(maxIterations/100));
    performanceMonitor.clusterSizeHistory = zeros(3, floor(maxIterations/100)); % min, max, mean
    performanceMonitor.phaseHistory = cell(1, floor(maxIterations/100));
end
function currentWeights = adjustWeightsForPhase(baseWeights, phase)
    % 根据优化阶段调整权重
    currentWeights = baseWeights;
    
    switch phase
        case 'morphology'
            % 形态优化阶段
            currentWeights.morphology = baseWeights.morphology * 1.5;
            currentWeights.shapePreservation = baseWeights.shapePreservation * 1.5;
            currentWeights.spatial = baseWeights.spatial * 0.7;
            currentWeights.multiScale = baseWeights.multiScale * 0.7;
            
        case 'spatial'
            % 空间优化阶段
            currentWeights.spatial = baseWeights.spatial * 1.5;
            currentWeights.multiScale = baseWeights.multiScale * 1.5;
            currentWeights.morphology = baseWeights.morphology * 0.7;
            currentWeights.shapePreservation = baseWeights.shapePreservation * 0.7;
            
        case 'balanced'
            % 平衡阶段，使用原始权重
            % 可以稍微增强连通性和结构一致性
            currentWeights.connectivity = baseWeights.connectivity * 1.2;
            currentWeights.structureCoherence = baseWeights.structureCoherence * 1.2;
    end
end
%% ========== 能量计算函数 ==========
function energy = calculateComprehensiveEnergy(mcmcState)
    % 计算综合能量函数
    features = mcmcState.features;
    spatialFeatures = mcmcState.spatialFeatures;
    morphologyFeatures = mcmcState.morphologyFeatures;
    multiScaleSpatialFeatures = mcmcState.multiScaleSpatialFeatures;
    optParams = mcmcState.optParams;
    weights = mcmcState.weights;
    
    % 基础能量组件
    E_porosity = calculatePorosityEnergy(features, optParams);
    E_cluster = calculateClusterEnergy(features, optParams);
    E_morphology = calculateMorphologyEnergy(morphologyFeatures, optParams);
    E_spatial = calculateSpatialEnergy(spatialFeatures, optParams);
    E_connectivity = calculateConnectivityEnergy(spatialFeatures, optParams);
    E_multiScale = calculateMultiScaleEnergy(multiScaleSpatialFeatures, optParams);
    E_shapePreservation = calculateShapePreservationEnergy(features, morphologyFeatures, optParams);
    E_structureCoherence = calculateStructureCoherenceEnergy(mcmcState);
    
    % 综合能量
    energy = weights.porosity * E_porosity + ...
        weights.cluster * E_cluster + ...
        weights.morphology * E_morphology + ...
        weights.spatial * E_spatial + ...
        weights.connectivity * E_connectivity + ...
        weights.multiScale * E_multiScale + ...
        weights.shapePreservation * E_shapePreservation + ...
        weights.structureCoherence * E_structureCoherence;
end
function E_porosity = calculatePorosityEnergy(features, optParams)
    % 计算孔隙率能量
    if features.numClusters > 0
        totalVolume = sum(features.sizes);
        modelVolume = prod(optParams.modelSize);
        currentPorosity = totalVolume / modelVolume;
    else
        currentPorosity = 0;
    end
    
    E_porosity = abs(currentPorosity - optParams.targetPorosity) / optParams.targetPorosity;
end
function E_cluster = calculateClusterEnergy(features, optParams)
    % 计算簇能量（改进版）
    if features.numClusters == 0
        E_cluster = 10.0;
        return;
    end
    
    % 最大簇惩罚（使用指数惩罚）
    maxSizeRatio = features.sizeStats(2) / optParams.targetMax;
    if maxSizeRatio > 1
        E_cluster_max = exp(2 * (maxSizeRatio - 1)) - 1;
    else
        E_cluster_max = 0;
    end
    
    % 最小簇惩罚
    minSizeRatio = features.sizeStats(1) / optParams.targetMin;
    if minSizeRatio < 1
        E_cluster_min = (1 - minSizeRatio)^2;
    else
        E_cluster_min = 0;
    end
    
    % 平均簇大小惩罚
    targetMeanSize = (optParams.targetMax + optParams.targetMin) / 2;
    E_cluster_mean = abs(features.sizeStats(3) - targetMeanSize) / targetMeanSize;
    
    % 簇数量惩罚
    expectedNumClusters = sum(features.sizes) / targetMeanSize;
    E_cluster_num = abs(features.numClusters - expectedNumClusters) / (expectedNumClusters + 1);
    if isfield(optParams, 'targetClusterCount') && optParams.targetClusterCount > 0
        targetCount = max(1, round(optParams.targetClusterCount));
        E_cluster_target = abs(features.numClusters - targetCount) / targetCount;
    else
        E_cluster_target = 0;
    end
    % 综合簇能量
    E_cluster = 3.0 * E_cluster_max + E_cluster_min + 0.5 * E_cluster_mean + ...
        0.15 * E_cluster_num + 0.65 * E_cluster_target;
end
function E_morphology = calculateMorphologyEnergy(morphologyFeatures, optParams)
    % 计算形态学能量
    E_morphology = 0;
    nTerms = 0;
    
    targetMorph = optParams.morphologyFeatures;
    
    % 形状描述符匹配
    if ~isempty(morphologyFeatures.elongation) && ~isempty(targetMorph.elongation)
        E_morphology = E_morphology + abs(mean(morphologyFeatures.elongation) - mean(targetMorph.elongation));
        nTerms = nTerms + 1;
    end
    
    if ~isempty(morphologyFeatures.sphericity) && ~isempty(targetMorph.sphericity)
        E_morphology = E_morphology + abs(mean(morphologyFeatures.sphericity) - mean(targetMorph.sphericity));
        nTerms = nTerms + 1;
    end
    
    % 网络特征匹配
    if isfield(morphologyFeatures, 'poreNetworkDensity') && isfield(targetMorph, 'poreNetworkDensity')
        E_morphology = E_morphology + abs(morphologyFeatures.poreNetworkDensity - targetMorph.poreNetworkDensity) / ...
            (targetMorph.poreNetworkDensity + 0.01);
        nTerms = nTerms + 1;
    end
    
    if nTerms > 0
        E_morphology = E_morphology / nTerms;
    end
end
function E_spatial = calculateSpatialEnergy(spatialFeatures, optParams)
    % 计算空间能量
    E_spatial = 0;
    nTerms = 0;
    
    targetSpatial = optParams.spatialFeatures;
    
    % 各向异性
    if isfield(spatialFeatures, 'anisotropy') && isfield(targetSpatial, 'anisotropy')
        E_spatial = E_spatial + abs(spatialFeatures.anisotropy - targetSpatial.anisotropy);
        nTerms = nTerms + 1;
    end
    
    % 两点相关函数
    if isfield(spatialFeatures, 'twoPointCorr') && isfield(targetSpatial, 'twoPointCorr')
        tpcDiff = mean(abs(spatialFeatures.twoPointCorr(:) - targetSpatial.twoPointCorr(:)));
        E_spatial = E_spatial + tpcDiff;
        nTerms = nTerms + 1;
    end
    
    % 空间自相关
    if isfield(spatialFeatures, 'spatialAutocorrelation') && isfield(targetSpatial, 'spatialAutocorrelation')
        E_spatial = E_spatial + abs(spatialFeatures.spatialAutocorrelation - targetSpatial.spatialAutocorrelation);
        nTerms = nTerms + 1;
    end

    % 弦长分布
    if isfield(spatialFeatures, 'chordLengthDistribution') && ...
            isfield(targetSpatial, 'chordLengthDistribution')
        currentCLD = spatialFeatures.chordLengthDistribution;
        targetCLD = targetSpatial.chordLengthDistribution;
        histDiff = computeDistributionDifference(currentCLD.binCenters, currentCLD.probability, ...
            targetCLD.binCenters, targetCLD.probability);
        meanDiff = normalizeDifference(currentCLD.meanLength, targetCLD.meanLength);
        E_spatial = E_spatial + 0.5 * histDiff + 0.5 * meanDiff;
        nTerms = nTerms + 1;
    end

    % 孔隙大小分布
    if isfield(spatialFeatures, 'poreSizeDistribution') && ...
            isfield(targetSpatial, 'poreSizeDistribution')
        currentPSD = spatialFeatures.poreSizeDistribution;
        targetPSD = targetSpatial.poreSizeDistribution;
        histDiff = computeDistributionDifference(currentPSD.binCenters, currentPSD.probability, ...
            targetPSD.binCenters, targetPSD.probability);
        meanDiff = normalizeDifference(currentPSD.meanRadius, targetPSD.meanRadius);
        E_spatial = E_spatial + 0.5 * histDiff + 0.5 * meanDiff;
        nTerms = nTerms + 1;
    end

    % Minkowski 积分平均曲率
    if isfield(spatialFeatures, 'minkowskiFunctionals') && ...
            isfield(targetSpatial, 'minkowskiFunctionals')
        currentMinkowski = spatialFeatures.minkowskiFunctionals;
        targetMinkowski = targetSpatial.minkowskiFunctionals;
        if isfield(currentMinkowski, 'integralMeanCurvature') && ...
                isfield(targetMinkowski, 'integralMeanCurvature')
            imcDiff = normalizeDifference(currentMinkowski.integralMeanCurvature, ...
                targetMinkowski.integralMeanCurvature);
            E_spatial = E_spatial + imcDiff;
            nTerms = nTerms + 1;
        end
    end

    % 线性路径函数
    if isfield(spatialFeatures, 'linealPathFunction') && ...
            isfield(targetSpatial, 'linealPathFunction')
        lpfDiff = computeLinealPathDifference(spatialFeatures.linealPathFunction, ...
            targetSpatial.linealPathFunction);
        E_spatial = E_spatial + lpfDiff;
        nTerms = nTerms + 1;
    end

    if nTerms > 0
        E_spatial = E_spatial / nTerms;
    end
end
function E_connectivity = calculateConnectivityEnergy(spatialFeatures, optParams)
    % 计算连通性能量
    E_connectivity = 0;
    nTerms = 0;
    
    targetConn = optParams.spatialFeatures.connectivity;
    
    if isfield(spatialFeatures, 'connectivity') && isfield(targetConn, 'eulerNumber')
        % 欧拉数差异
        eulerDiff = abs(spatialFeatures.connectivity.eulerNumber - targetConn.eulerNumber) / ...
            (abs(targetConn.eulerNumber) + 1);
        E_connectivity = E_connectivity + eulerDiff;
        nTerms = nTerms + 1;
        
        % 最大组分比例
        if isfield(spatialFeatures.connectivity, 'largestComponentRatio')
            ratioDiff = abs(spatialFeatures.connectivity.largestComponentRatio - ...
                targetConn.largestComponentRatio);
            E_connectivity = E_connectivity + ratioDiff;
            nTerms = nTerms + 1;
        end
    end
    
    if nTerms > 0
        E_connectivity = E_connectivity / nTerms;
    end
end
function E_multiScale = calculateMultiScaleEnergy(multiScaleFeatures, optParams)
    % 计算多尺度能量
    E_multiScale = 0;
    targetMultiScale = optParams.multiScaleSpatialFeatures;
    
    nScales = min(length(multiScaleFeatures.scale), length(targetMultiScale.scale));
    
    for s = 1:nScales
        if isfield(multiScaleFeatures.scale(s), 'twoPointCorr') && ...
            isfield(targetMultiScale.scale(s), 'twoPointCorr')
            tpcDiff = mean(abs(multiScaleFeatures.scale(s).twoPointCorr(:) - ...
                targetMultiScale.scale(s).twoPointCorr(:)));
            E_multiScale = E_multiScale + tpcDiff;
        end
    end
    
    if nScales > 0
        E_multiScale = E_multiScale / nScales;
    end
end
function E_shape = calculateShapePreservationEnergy(features, morphologyFeatures, optParams)
    % 计算形状保持能量
    E_shape = 0;
    nTerms = 0;
    
    % 簇大小分布的形状保持
    if ~isempty(features.sizes) && ~isempty(optParams.originalFeatures.sizes)
        % 计算大小分布的差异
        currentHist = histcounts(log10(features.sizes + 1), 20);
        targetHist = histcounts(log10(optParams.originalFeatures.sizes + 1), 20);
        
        % 归一化
        currentHist = currentHist / sum(currentHist);
        targetHist = targetHist / sum(targetHist);
        
        % KL散度
        kl_div = sum(targetHist .* log((targetHist + eps) ./ (currentHist + eps)));
        E_shape = E_shape + kl_div;
        nTerms = nTerms + 1;
    end
    
    % 形态特征的保持
    if ~isempty(morphologyFeatures.sphericity) && ...
        isfield(optParams, 'morphologyFeatures') && ...
        ~isempty(optParams.morphologyFeatures.sphericity)
        
        % 球形度分布差异
        spherDiff = abs(std(morphologyFeatures.sphericity) - ...
            std(optParams.morphologyFeatures.sphericity));
        E_shape = E_shape + spherDiff;
        nTerms = nTerms + 1;
        
        % 伸长率分布差异
        if ~isempty(morphologyFeatures.elongation) && ...
            ~isempty(optParams.morphologyFeatures.elongation)
            elongDiff = abs(std(morphologyFeatures.elongation) - ...
                std(optParams.morphologyFeatures.elongation));
            E_shape = E_shape + elongDiff;
            nTerms = nTerms + 1;
        end
    end
    
    % 空间分布的形状保持
    if ~isempty(features.compactness) && ...
        isfield(optParams.originalFeatures, 'compactness') && ...
        ~isempty(optParams.originalFeatures.compactness)
        compactDiff = abs(mean(features.compactness) - ...
            mean(optParams.originalFeatures.compactness));
        E_shape = E_shape + compactDiff;
        nTerms = nTerms + 1;
    end
    
    if nTerms > 0
        E_shape = E_shape / nTerms;
    end
end
function E_structure = calculateStructureCoherenceEnergy(mcmcState)
    % 计算结构一致性能量
    % 评估局部结构的连贯性
    model = mcmcState.model;
    [nx, ny, nz] = size(model);
    
    % 采样评估结构连贯性
    nSamples = 20;
    coherenceScores = zeros(nSamples, 1);
    
    for i = 1:nSamples
        % 随机选择一个局部区域
        x = randi([10, nx-10]);
        y = randi([10, ny-10]);
        z = randi([5, nz-5]);
        
        % 提取局部窗口
        window = model(x-9:x+9, y-9:y+9, z-4:z+4);
        
        % 计算局部连贯性
        coherenceScores(i) = computeLocalCoherence(window);
    end
    
    E_structure = 1 - mean(coherenceScores);
end
function coherence = computeLocalCoherence(window)
    % 计算局部连贯性
    % 基于梯度一致性
    [gx, gy, gz] = gradient(double(window));
    gradMag = sqrt(gx.^2 + gy.^2 + gz.^2);
    
    % 计算梯度方向的一致性
    validIdx = gradMag > 0.1;
    if sum(validIdx(:)) < 10
        coherence = 0.5;
        return;
    end
    
    % 归一化梯度向量
    gx_norm = gx(validIdx) ./ gradMag(validIdx);
    gy_norm = gy(validIdx) ./ gradMag(validIdx);
    gz_norm = gz(validIdx) ./ gradMag(validIdx);
    
    % 计算平均方向
    mean_dir = [mean(gx_norm), mean(gy_norm), mean(gz_norm)];
    mean_dir = mean_dir / (norm(mean_dir) + eps);
    
    % 计算与平均方向的一致性
    coherence = mean(abs(gx_norm * mean_dir(1) + gy_norm * mean_dir(2) + gz_norm * mean_dir(3)));
end
%% ========== 特征更新和约束函数 ==========
function mcmcState = updateAllFeatures(mcmcState)
    % 更新所有特征
    mcmcState.features = extractEfficientClusterFeatures(mcmcState.model);
    mcmcState.spatialFeatures = computeEnhancedSpatialFeatures(mcmcState.model);
    mcmcState.morphologyFeatures = computeDetailedMorphologyFeatures(mcmcState.model);
    mcmcState.multiScaleSpatialFeatures = computeMultiScaleSpatialFeatures(mcmcState.model);
    mcmcState.currentEnergy = calculateComprehensiveEnergy(mcmcState);
end
function mcmcState = enforceComprehensiveConstraints(mcmcState, optParams, phase)
    % 强制执行综合约束
    switch phase
        case 'morphology'
            % 形态阶段：重点强制形态约束
            mcmcState = enforceMorphologyConstraints(mcmcState, optParams);
        case 'spatial'
            % 空间阶段：重点强制空间约束
            mcmcState = enforceSpatialConstraints(mcmcState, optParams);
        case 'balanced'
            % 平衡阶段：综合约束
            mcmcState = enforceBalancedConstraints(mcmcState, optParams);
    end
    
    % 始终强制簇大小约束
    mcmcState.model = enforceClusterSizeConstraints(mcmcState.model, optParams);
    mcmcState.model = enforceTargetClusterCount(mcmcState.model, optParams);
end
function mcmcState = enforceMorphologyConstraints(mcmcState, optParams)
    % 强制执行形态学约束
    targetMorph = optParams.morphologyFeatures;
    currentMorph = mcmcState.morphologyFeatures;
    
    % 球形度约束
    if ~isempty(targetMorph.sphericity) && ~isempty(currentMorph.sphericity)
        targetSpher = mean(targetMorph.sphericity);
        currentSpher = mean(currentMorph.sphericity);
        
        if abs(currentSpher - targetSpher) > 0.2
            % 选择一些簇进行形态调整
            CC = bwconncomp(mcmcState.model, 26);
            if CC.NumObjects > 0
                nAdjust = min(5, CC.NumObjects);
                adjustIdx = randperm(CC.NumObjects, nAdjust);
                
                for i = 1:nAdjust
                    clusterMask = false(size(mcmcState.model));
                    clusterMask(CC.PixelIdxList{adjustIdx(i)}) = true;
                    
                    if targetSpher > currentSpher
                        % 增加球形度
                        se = strel('sphere', 1);
                        adjusted = imclose(clusterMask, se);
                    else
                        % 减少球形度
                        se = strel('line', 3, randi([0, 180]));
                        adjusted = imdilate(clusterMask, se);
                    end
                    
                    % 更新模型
                    mcmcState.model(CC.PixelIdxList{adjustIdx(i)}) = false;
                    mcmcState.model(adjusted) = true;
                end
            end
        end
    end
    
    % 更新特征
    mcmcState.morphologyFeatures = computeDetailedMorphologyFeatures(mcmcState.model);
end
function mcmcState = enforceSpatialConstraints(mcmcState, optParams)
    % 强制执行空间约束
    targetSpatial = optParams.spatialFeatures;
    currentSpatial = mcmcState.spatialFeatures;
    
    % 各向异性约束
    if abs(currentSpatial.anisotropy - targetSpatial.anisotropy) > 0.1
        mcmcState.model = adjustModelAnisotropy(mcmcState.model, ...
            currentSpatial.anisotropy, targetSpatial.anisotropy);
    end
    
    % 连通性约束
    if isfield(currentSpatial, 'connectivity') && isfield(targetSpatial, 'connectivity')
        if currentSpatial.connectivity.largestComponentRatio < ...
            targetSpatial.connectivity.largestComponentRatio - 0.1
            % 增强连通性
            mcmcState.model = enhanceConnectivity(mcmcState.model);
        end
    end
    
    % 更新特征
    mcmcState.spatialFeatures = computeEnhancedSpatialFeatures(mcmcState.model);
end
function mcmcState = enforceBalancedConstraints(mcmcState, optParams)
    % 强制执行平衡约束
    % 检查各项指标并进行针对性调整
    morphMatch = calculateMorphologyMatch(mcmcState.morphologyFeatures, optParams.morphologyFeatures);
    spatialMatch = calculateSpatialMatch(mcmcState.spatialFeatures, optParams.spatialFeatures);
    
    % 根据匹配度决定调整策略
    if morphMatch < 0.5 && spatialMatch < 0.5
        % 两者都较差，进行综合调整
        mcmcState.model = comprehensiveModelAdjustment(mcmcState.model, optParams);
    elseif morphMatch < spatialMatch - 0.2
        % 形态匹配较差
        mcmcState = enforceMorphologyConstraints(mcmcState, optParams);
    elseif spatialMatch < morphMatch - 0.2
        % 空间匹配较差
        mcmcState = enforceSpatialConstraints(mcmcState, optParams);
    end
end
function model = enforceClusterSizeConstraints(model, optParams)
    % 强制执行簇大小约束
    CC = bwconncomp(model, 26);
    if CC.NumObjects == 0
        return;
    end
    
    sizes = cellfun(@numel, CC.PixelIdxList);
    modified = false;
    
    % 处理过大的簇
    largeClusters = find(sizes > optParams.targetMax * 2);
    for i = 1:length(largeClusters)
        idx = largeClusters(i);
        if sizes(idx) > optParams.targetMax * 2
            model = reduceOversizedCluster(model, CC.PixelIdxList{idx}, optParams);
            modified = true;
        end
    end
    % 处理过小的簇
    tinyClusters = find(sizes < optParams.targetMin / 2);
    if ~isempty(tinyClusters)
        for i = 1:length(tinyClusters)
            if isfield(optParams, 'preserveSmallPores') && optParams.preserveSmallPores
                model = growSmallCluster(model, CC.PixelIdxList{tinyClusters(i)});
            else
                model(CC.PixelIdxList{tinyClusters(i)}) = false;
            end
        end
        modified = true;
    end
    
    % 如果有修改，更新连通组分
    if modified
        CC = bwconncomp(model, 26);
        sizes = cellfun(@numel, CC.PixelIdxList);
    end
    
    % 尝试合并接近的小簇
    smallClusters = find(sizes >= optParams.targetMin/2 & sizes < optParams.targetMin);
    if length(smallClusters) >= 2
        % 找到接近的小簇对
        for i = 1:length(smallClusters)-1
            for j = i+1:length(smallClusters)
                idx1 = smallClusters(i);
                idx2 = smallClusters(j);
                
                % 检查是否接近
                [dist, ~, ~] = findClosestPoints(model, CC.PixelIdxList{idx1}, CC.PixelIdxList{idx2});
                if dist < 5
                    % 连接两个簇
                    model = connectClusters(model, CC.PixelIdxList{idx1}, CC.PixelIdxList{idx2});
                    break;
                end
            end
        end
    end
end
function model = enforceTargetClusterCount(model, optParams)
    % 强制模型的簇数量与目标值一致
    if ~isfield(optParams, 'targetClusterCount') || isempty(optParams.targetClusterCount)
        return;
    end
    targetCount = max(1, round(optParams.targetClusterCount));
    CC = bwconncomp(model, 26);
    % 如果当前没有簇，尝试播种新的孔隙簇
    if CC.NumObjects == 0
        model = seedRandomCluster(model, optParams);
        CC = bwconncomp(model, 26);
    end
    maxIterations = 50;
    iter = 0;
    while CC.NumObjects > targetCount && iter < maxIterations
        model = mergeClosestClusters(model, CC);
        CC = bwconncomp(model, 26);
        iter = iter + 1;
    end
    iter = 0;
    while CC.NumObjects < targetCount && iter < maxIterations
        [model, added] = addRepresentativeCluster(model, CC, optParams);
        CC = bwconncomp(model, 26);
        iter = iter + 1;
        if ~added
            break;
        end
    end
    % 如果迭代后仍未达到目标，执行微调
    if CC.NumObjects ~= targetCount
        tolerance = abs(CC.NumObjects - targetCount);
        if CC.NumObjects > targetCount && tolerance > 0
            % 移除最小的簇以精确匹配
            sizes = cellfun(@numel, CC.PixelIdxList);
            [~, order] = sort(sizes, 'ascend');
            removeCount = min(tolerance, length(order));
            for i = 1:removeCount
                model(CC.PixelIdxList{order(i)}) = false;
            end
        elseif CC.NumObjects < targetCount
            % 通过播种新的小簇补足
            for i = 1:(targetCount - CC.NumObjects)
                [model, added] = addRepresentativeCluster(model, CC, optParams);
                if ~added
                    model = seedRandomCluster(model, optParams);
                end
                CC = bwconncomp(model, 26);
            end
        end
    end
    % 调整孔隙率以补偿操作的影响
    model = adjustPorosity(model, optParams.targetPorosity);
end
function [model, added] = addRepresentativeCluster(model, CC, optParams)
    % 添加新的代表性簇以增加簇数量
    added = false;
    library = {};
    if isfield(optParams, 'clusterLibrary') && ~isempty(optParams.clusterLibrary)
        library = optParams.clusterLibrary;
    elseif isfield(optParams, 'originalBinaryModel') && ~isempty(optParams.originalBinaryModel)
        library = sampleRepresentativeClusters(optParams.originalBinaryModel, optParams.morphologyFeatures);
    end
    if isempty(library)
        model = seedRandomCluster(model, optParams);
        added = true;
        return;
    end
    densityMap = constructMultiScaleDensityMap(optParams, size(model));
    beforeCount = CC.NumObjects;
    order = randperm(length(library));
    for i = 1:length(order)
        candidate = placeClusterSample(model, library{order(i)}, densityMap);
        CC_new = bwconncomp(candidate, 26);
        if CC_new.NumObjects > beforeCount
            model = candidate;
            added = true;
            return;
        end
    end
    % 如果所有代表簇都未能增加数量，则退化为随机播种
    model = seedRandomCluster(model, optParams);
    added = true;
end
function model = mergeClosestClusters(model, CC)
    % 合并最近的两个簇以减少簇数量
    if CC.NumObjects < 2
        return;
    end
    sizes = cellfun(@numel, CC.PixelIdxList);
    [~, order] = sort(sizes, 'ascend');
    merged = false;
    for idx = 1:length(order)
        currentIdx = order(idx);
        bestPartner = [];
        bestDist = inf;
        for j = 1:CC.NumObjects
            if j == currentIdx
                continue;
            end
            [dist, ~, ~] = findClosestPoints(model, CC.PixelIdxList{currentIdx}, CC.PixelIdxList{j});
            if dist < bestDist
                bestDist = dist;
                bestPartner = j;
            end
        end
        if ~isempty(bestPartner)
            model = connectClusters(model, CC.PixelIdxList{currentIdx}, CC.PixelIdxList{bestPartner});
            merged = true;
            break;
        end
    end
    if ~merged
        % 如果无法连接，则移除最小簇
        smallest = order(1);
        model(CC.PixelIdxList{smallest}) = false;
    end
end
function model = seedRandomCluster(model, optParams)
    % 随机播种一个新的孔隙簇
    dims = size(model);
    radius = max(2, round((optParams.targetMin)^(1/3)));
    center = [randi(dims(1)), randi(dims(2)), randi(dims(3))];
    for dx = -radius:radius
        for dy = -radius:radius
            for dz = -radius:radius
                pos = center + [dx, dy, dz];
                if all(pos >= 1) && pos(1) <= dims(1) && pos(2) <= dims(2) && pos(3) <= dims(3)
                    if sqrt(double(dx)^2 + double(dy)^2 + double(dz)^2) <= radius
                        model(pos(1), pos(2), pos(3)) = true;
                    end
                end
            end
        end
    end
end
function model = enhanceConnectivity(model)
    % 增强模型连通性
    CC = bwconncomp(model, 26);
    if CC.NumObjects <= 1
        return;
    end
    
    sizes = cellfun(@numel, CC.PixelIdxList);
    [~, sortIdx] = sort(sizes, 'descend');
    
    % 尝试连接前几个最大的组分
    nConnect = min(3, CC.NumObjects);
    for i = 1:nConnect-1
        for j = i+1:nConnect
            [dist, p1, p2] = findClosestPoints(model, ...
                CC.PixelIdxList{sortIdx(i)}, CC.PixelIdxList{sortIdx(j)});
            
            if dist < 8 && dist > 1
                % 创建连接
                model = createConnectionPath(model, p1, p2);
            end
        end
    end
end
function model = createConnectionPath(model, p1, p2)
    % 创建两点之间的连接路径
    nSteps = ceil(norm(p1 - p2) * 1.5);
    
    for t = 0:1/nSteps:1
        pos = round(p1 * (1-t) + p2 * t);
        
        % 在路径上添加体素
        for dx = -1:1
            for dy = -1:1
                for dz = 0:0
                    x = pos(1) + dx;
                    y = pos(2) + dy;
                    z = pos(3) + dz;
                    if x >= 1 && x <= size(model,1) && ...
                        y >= 1 && y <= size(model,2) && ...
                        z >= 1 && z <= size(model,3)
                        model(x, y, z) = true;
                    end
                end
            end
        end
    end
end
function model = adjustModelAnisotropy(model, currentAniso, targetAniso)
    % 调整模型各向异性
    diff = targetAniso - currentAniso;
    
    if abs(diff) < 0.05
        return; % 已经足够接近
    end
    
    [nx, ny, nz] = size(model);
    
    if diff > 0
        % 需要增加各向异性（增强Z方向的差异）
        for z = 2:nz-1
            if rand() < 0.1 % 10%的层进行调整
                slice = model(:, :, z);
                prevSlice = model(:, :, z-1);
                nextSlice = model(:, :, z+1);
                
                % 减少与相邻层的相似性
                similarity = (slice == prevSlice) | (slice == nextSlice);
                changeIdx = find(similarity);
                
                if ~isempty(changeIdx)
                    nChange = round(length(changeIdx) * 0.1);
                    selectedIdx = changeIdx(randperm(length(changeIdx), nChange));
                    
                    for i = 1:length(selectedIdx)
                        [x, y] = ind2sub([nx, ny], selectedIdx(i));
                        model(x, y, z) = ~model(x, y, z);
                    end
                end
            end
        end
    else
        % 需要减少各向异性（增强各向同性）
        % 使用3D平滑
        smoothed = imgaussfilt3(double(model), 1);
        threshold = 0.5;
        
        % 混合原始和平滑结果
        alpha = min(0.3, abs(diff));
        model = (1-alpha) * double(model) + alpha * smoothed;
        model = model > threshold;
    end
end
function model = comprehensiveModelAdjustment(model, optParams)
    % 综合模型调整
    % 在模型中创建具有目标特征的示例区域
    [nx, ny, nz] = size(model);
    nRegions = 2;
    regionSize = 15;
    
    for r = 1:nRegions
        % 随机选择区域
        cx = randi([regionSize, nx-regionSize]);
        cy = randi([regionSize, ny-regionSize]);
        cz = randi([regionSize/2, nz-regionSize/2]);
        
        % 创建具有目标特征的局部结构
        localStructure = createTargetStructure(regionSize, optParams);
        
        % 混合到模型中
        model = blendStructureIntoModel(model, localStructure, cx, cy, cz);
    end
end
function localStructure = createTargetStructure(size, optParams)
    % 创建具有目标特征的局部结构
    targetSphericity = mean(optParams.morphologyFeatures.sphericity);
    targetAnisotropy = optParams.spatialFeatures.anisotropy;
    
    % 根据目标特征生成结构
    if targetSphericity > 0.7
        % 球形结构
        localStructure = createSphericalStructure(size);
    elseif targetSphericity < 0.4
        % 伸长结构
        localStructure = createElongatedStructure(size, targetAnisotropy);
    else
        % 中等结构
        localStructure = createModerateStructure(size);
    end
end
function structure = createSphericalStructure(size)
    % 创建球形结构
    [x, y, z] = meshgrid(1:size, 1:size, 1:size/2);
    center = [size/2, size/2, size/4];
    radius = size/3;
    structure = sqrt((x-center(1)).^2 + (y-center(2)).^2 + (z-center(3)).^2) <= radius;
end
function structure = createElongatedStructure(size, anisotropy)
    % 创建伸长结构
    structure = false(size, size, size/2);
    
    % 创建主轴
    for i = 1:size
        for j = 1:size
            for k = 1:size/2
                % 椭球体方程
                if ((i-size/2)^2/(size/2)^2 + (j-size/2)^2/(size/3)^2 + ...
                    (k-size/4)^2/((size/4)*(1+anisotropy))^2) <= 1
                    structure(i,j,k) = true;
                end
            end
        end
    end
end
function structure = createModerateStructure(size)
    % 创建中等结构
    % 使用随机游走创建不规则形状
    structure = false(size, size, size/2);
    nWalkers = 5;
    nSteps = size^2;
    
    for w = 1:nWalkers
        % 起始位置
        pos = [size/2, size/2, size/4] + randn(1,3)*size/10;
        
        for step = 1:nSteps
            % 随机游走
            pos = pos + randn(1,3);
            
            % 边界检查
            pos = max(1, min(pos, [size, size, size/2]));
            
            % 在路径上创建球形区域
            x = round(pos(1));
            y = round(pos(2));
            z = round(pos(3));
            
            for dx = -2:2
                for dy = -2:2
                    for dz = -1:1
                        nx = x + dx;
                        ny = y + dy;
                        nz = z + dz;
                        if nx >= 1 && nx <= size && ny >= 1 && ny <= size && ...
                            nz >= 1 && nz <= size/2
                            if sqrt(dx^2 + dy^2 + dz^2) <= 2
                                structure(nx, ny, nz) = true;
                            end
                        end
                    end
                end
            end
        end
    end
end
function model = blendStructureIntoModel(model, structure, cx, cy, cz)
    % 将局部结构混合到模型中
    [sx, sy, sz] = size(structure);
    blendFactor = 0.7;
    
    for i = 1:sx
        for j = 1:sy
            for k = 1:sz
                x = cx - sx/2 + i;
                y = cy - sy/2 + j;
                z = cz - sz/2 + k;
                
                if x >= 1 && x <= size(model,1) && ...
                    y >= 1 && y <= size(model,2) && ...
                    z >= 1 && z <= size(model,3)
                    if rand() < blendFactor
                        model(x, y, z) = structure(i, j, k);
                    end
                end
            end
        end
    end
end
function model = connectClusters(model, cluster1, cluster2)
    % 连接两个簇
    [x1, y1, z1] = ind2sub(size(model), cluster1);
    [x2, y2, z2] = ind2sub(size(model), cluster2);
    
    % 找到最近的两点
    minDist = inf;
    p1 = []; p2 = [];
    
    for i = 1:min(50, length(x1))
        for j = 1:min(50, length(x2))
            dist = sqrt((x1(i)-x2(j))^2 + (y1(i)-y2(j))^2 + (z1(i)-z2(j))^2);
            if dist < minDist
                minDist = dist;
                p1 = [x1(i), y1(i), z1(i)];
                p2 = [x2(j), y2(j), z2(j)];
            end
        end
    end
    
    % 创建连接路径
    if ~isempty(p1) && ~isempty(p2)
        nSteps = ceil(minDist);
        for t = 0:1/nSteps:1
            pos = round(p1 * (1-t) + p2 * t);
            if all(pos >= 1) && pos(1) <= size(model,1) && ...
                pos(2) <= size(model,2) && pos(3) <= size(model,3)
                model(pos(1), pos(2), pos(3)) = true;
            end
        end
    end
end
function [dist, point1, point2] = findClosestPoints(model, cluster1Idx, cluster2Idx)
    % 找到两个簇之间的最近点
    % 初始化输出参数
    dist = inf;
    point1 = [];
    point2 = [];
    
    % 检查输入是否为空
    if isempty(cluster1Idx) || isempty(cluster2Idx)
        return;
    end
    
    [x1, y1, z1] = ind2sub(size(model), cluster1Idx);
    [x2, y2, z2] = ind2sub(size(model), cluster2Idx);
    
    % 确保坐标是列向量
    x1 = x1(:); y1 = y1(:); z1 = z1(:);
    x2 = x2(:); y2 = y2(:); z2 = z2(:);
    
    % 再次检查坐标是否为空
    if isempty(x1) || isempty(x2)
        return;
    end
    
    % 限制采样数量以加速
    n1 = length(x1);
    n2 = length(x2);
    maxSample = 200;
    
    if n1 > maxSample
        sample1 = randperm(n1, maxSample);
        x1 = x1(sample1);
        y1 = y1(sample1);
        z1 = z1(sample1);
        n1 = maxSample;
    end
    
    if n2 > maxSample
        sample2 = randperm(n2, maxSample);
        x2 = x2(sample2);
        y2 = y2(sample2);
        z2 = z2(sample2);
        n2 = maxSample;
    end
    
    % 计算距离矩阵（分批处理）
    minDist = inf;
    point1 = [];
    point2 = [];
    
    % 分批计算以避免内存问题
    batchSize = 50;
    for i = 1:batchSize:n1
        endIdx = min(i+batchSize-1, n1);
        batchSize1 = endIdx - i + 1;
        
        % 创建批次矩阵
        x1_batch = repmat(x1(i:endIdx), 1, n2);
        y1_batch = repmat(y1(i:endIdx), 1, n2);
        z1_batch = repmat(z1(i:endIdx), 1, n2);
        
        x2_batch = repmat(x2', batchSize1, 1);
        y2_batch = repmat(y2', batchSize1, 1);
        z2_batch = repmat(z2', batchSize1, 1);
        
        % 计算批次距离
        distances = sqrt((x1_batch - x2_batch).^2 + ...
            (y1_batch - y2_batch).^2 + ...
            (z1_batch - z2_batch).^2);
        
        % 找到最小距离
        [minBatchDist, minIdx] = min(distances(:));
        if minBatchDist < minDist
            minDist = minBatchDist;
            [row, col] = ind2sub(size(distances), minIdx);
            point1 = [x1(i+row-1), y1(i+row-1), z1(i+row-1)];
            point2 = [x2(col), y2(col), z2(col)];
        end
    end
    
    % 最终赋值
    dist = minDist;
    
    % 如果没有找到有效点，使用第一个点
    if isempty(point1) && n1 > 0
        point1 = [x1(1), y1(1), z1(1)];
    end
    if isempty(point2) && n2 > 0
        point2 = [x2(1), y2(1), z2(1)];
    end
end
%% ========== 移动策略和生成函数 ==========
function strategy = selectComprehensiveAdaptiveMoveStrategy(iter, maxIter, mcmcState, phase)
    % 选择综合自适应移动策略
    progress = iter / maxIter;
    
    switch phase
        case 'morphology'
            % 形态优化阶段策略
            if progress < 0.1
                strategies = {'morphology_preserving', 'local_shape', 'boundary_smooth'};
                weights = [0.5, 0.3, 0.2];
            else
                strategies = {'morphology_preserving', 'cluster_shape', 'local_shape', 'fine_morphology'};
                weights = [0.4, 0.3, 0.2, 0.1];
            end
            
        case 'spatial'
            % 空间优化阶段策略
            strategies = {'spatial_aware', 'spatial_correlation', 'anisotropy_adjust', 'connectivity_enhance'};
            weights = [0.4, 0.3, 0.2, 0.1];
            
        case 'balanced'
            % 平衡阶段策略
            if progress < 0.8
                strategies = {'local', 'boundary', 'cluster', 'spatial_aware', 'morphology_preserving'};
                weights = [0.2, 0.2, 0.2, 0.2, 0.2];
            else
                strategies = {'fine_tune', 'local', 'structure_coherence'};
                weights = [0.4, 0.4, 0.2];
            end
            
        otherwise
            strategies = {'local'};
            weights = [1.0];
    end
    
    % 根据接受率调整
    if mcmcState.acceptanceRate < 0.1
        % 接受率太低，使用更保守的策略
        strategies = [strategies, {'fine_tune'}];
        weights = [weights * 0.7, 0.3];
    elseif mcmcState.acceptanceRate > 0.9
        % 接受率太高，使用更激进的策略
        strategies = [strategies, {'large_scale'}];
        weights = [weights * 0.8, 0.2];
    end
    
    % 归一化权重
    weights = weights / sum(weights);
    
    % 选择策略
    cumWeights = cumsum(weights);
    r = rand();
    idx = find(cumWeights >= r, 1);
    strategy = strategies{idx};
end
%% ========== 移动策略和生成函数（续） ==========
function [moves, moveTypes] = generateComprehensiveBatchMoves(model, strategy, ...
    batchSize, optParams, lookupTables)
    % 生成综合批量移动
    moves = cell(batchSize, 1);
    moveTypes = cell(batchSize, 1);
    
    for i = 1:batchSize
        switch strategy
            % 形态相关移动
            case 'morphology_preserving'
                moves{i} = generateMorphologyPreservingMove(model, optParams);
                moveTypes{i} = 'morphology_preserving';
                
            case 'local_shape'
                moves{i} = generateLocalShapeMove(model, lookupTables);
                moveTypes{i} = 'local_shape';
                
            case 'boundary_smooth'
                moves{i} = generateBoundarySmoothingMove(model);
                moveTypes{i} = 'boundary_smooth';
                
            case 'cluster_shape'
                moves{i} = generateClusterShapeMove(model, optParams);
                moveTypes{i} = 'cluster_shape';
                
            case 'fine_morphology'
                moves{i} = generateFineMorphologyMove(model, optParams);
                moveTypes{i} = 'fine_morphology';
                
            % 空间相关移动
            case 'spatial_aware'
                moves{i} = generateSpatialAwareMove(model, lookupTables);
                moveTypes{i} = 'spatial_aware';
                
            case 'spatial_correlation'
                moves{i} = generateSpatialCorrelationMove(model, optParams);
                moveTypes{i} = 'spatial_correlation';
                
            case 'anisotropy_adjust'
                moves{i} = generateAnisotropyAdjustMove(model, lookupTables);
                moveTypes{i} = 'anisotropy_adjust';
                
            case 'connectivity_enhance'
                moves{i} = generateConnectivityEnhanceMove(model, optParams);
                moveTypes{i} = 'connectivity_enhance';
                
            % 通用移动
            case 'structure_coherence'
                moves{i} = generateStructureCoherenceMove(model, lookupTables);
                moveTypes{i} = 'structure_coherence';
                
            case 'large_scale'
                moves{i} = generateLargeScaleMove(model, optParams);
                moveTypes{i} = 'large_scale';
                
            case 'fine_tune'
                moves{i} = generateFineTuneMove(model);
                moveTypes{i} = 'fine_tune';
                
            case 'cluster'
                moves{i} = generateClusterMove(model, optParams);
                moveTypes{i} = 'cluster';
                
            case 'boundary'
                moves{i} = generateBoundaryMove(model);
                moveTypes{i} = 'boundary';
                
            otherwise
                % 默认移动类型
                moves{i} = generateLocalMove(model, 5);
                moveTypes{i} = 'local';
        end
    end
end
%% ========== 具体移动生成函数 ==========
function move = generateLocalMove(model, radius)
    % 生成围绕孔隙簇的局部移动，优先在簇边界附近调整
    if nargin < 2
        radius = 3;
    end
    move = struct();
    move.linearIdx = [];
    move.oldValues = [];
    move.newValues = [];
    [nx, ny, nz] = size(model);
    boundary = bwperim(model, 26);
    if any(boundary(:))
        neighborhood = imdilate(boundary, strel('sphere', max(1, radius)));
        candidateIdx = find(neighborhood);
    else
        candidateIdx = [];
    end
    if isempty(candidateIdx)
        cx = randi(nx);
        cy = randi(ny);
        cz = randi(nz);
    else
        centerLinear = candidateIdx(randi(length(candidateIdx)));
        [cx, cy, cz] = ind2sub(size(model), centerLinear);
    end
    localIdx = [];
    for dx = -radius:radius
        for dy = -radius:radius
            for dz = -radius:radius
                x = cx + dx;
                y = cy + dy;
                z = cz + dz;
                if x >= 1 && x <= nx && y >= 1 && y <= ny && z >= 1 && z <= nz
                    if sqrt(double(dx)^2 + double(dy)^2 + double(dz)^2) <= radius + 0.01
                        localIdx(end+1, 1) = sub2ind(size(model), x, y, z); %#ok<AGROW>
                    end
                end
            end
        end
    end
    localIdx = unique(localIdx);
    if isempty(localIdx)
        return;
    end
    localValues = model(localIdx);
    poreIdx = localIdx(localValues);
    matrixIdx = localIdx(~localValues);
    if isempty(poreIdx) && isempty(matrixIdx)
        return;
    end
    localPorosity = mean(localValues);
    globalPorosity = mean(model(:));
    targetLocalPorosity = 0.5 * localPorosity + 0.5 * globalPorosity;
    nPairs = max(1, round(0.12 * min(numel(poreIdx), numel(matrixIdx))));
    addCount = nPairs;
    removeCount = nPairs;
    if targetLocalPorosity > localPorosity && ~isempty(matrixIdx)
        addCount = min(numel(matrixIdx), addCount + 1);
    elseif targetLocalPorosity < localPorosity && ~isempty(poreIdx)
        removeCount = min(numel(poreIdx), removeCount + 1);
    end
    if ~isempty(matrixIdx)
        addIdx = matrixIdx(randperm(numel(matrixIdx), min(addCount, numel(matrixIdx))));
        move.linearIdx = [move.linearIdx; addIdx(:)]; %#ok<AGROW>
        move.oldValues = [move.oldValues; false(numel(addIdx), 1)]; %#ok<AGROW>
        move.newValues = [move.newValues; true(numel(addIdx), 1)]; %#ok<AGROW>
    end
    if ~isempty(poreIdx)
        removeIdx = poreIdx(randperm(numel(poreIdx), min(removeCount, numel(poreIdx))));
        move.linearIdx = [move.linearIdx; removeIdx(:)]; %#ok<AGROW>
        move.oldValues = [move.oldValues; true(numel(removeIdx), 1)]; %#ok<AGROW>
        move.newValues = [move.newValues; false(numel(removeIdx), 1)]; %#ok<AGROW>
    end
    if ~isempty(move.linearIdx)
        [move.linearIdx, uniqueIdx] = unique(move.linearIdx, 'stable');
        move.oldValues = move.oldValues(uniqueIdx);
        move.newValues = move.newValues(uniqueIdx);
    end
end
function move = generateMorphologyPreservingMove(model, optParams)
    % 生成保持形态的移动
    move = struct();
    
    % 找到边界体素
    boundary = bwperim(model, 26);
    boundaryIdx = find(boundary);
    
    if isempty(boundaryIdx)
        move = generateLocalMove(model, 3);
        return;
    end
    
    % 选择一些边界点
    nSelect = min(10, length(boundaryIdx));
    selectedIdx = boundaryIdx(randperm(length(boundaryIdx), nSelect));
    
    move.linearIdx = [];
    move.oldValues = [];
    move.newValues = [];
    
    % 根据目标形态特征决定操作
    targetSpher = mean(optParams.morphologyFeatures.sphericity);
    
    for i = 1:length(selectedIdx)
        [x, y, z] = ind2sub(size(model), selectedIdx(i));
        
        % 计算局部曲率
        localCurvature = estimateLocalCurvature(model, x, y, z);
        
        if targetSpher > 0.7 && localCurvature < 0
            % 高球形度目标，填充凹陷
            if ~model(selectedIdx(i))
                move.linearIdx = [move.linearIdx; selectedIdx(i)];
                move.oldValues = [move.oldValues; false];
                move.newValues = [move.newValues; true];
            end
        elseif targetSpher < 0.4 && localCurvature > 0
            % 低球形度目标，增加不规则性
            if model(selectedIdx(i)) && rand() < 0.3
                move.linearIdx = [move.linearIdx; selectedIdx(i)];
                move.oldValues = [move.oldValues; true];
                move.newValues = [move.newValues; false];
            end
        end
    end
    
    if isempty(move.linearIdx)
        move = generateLocalMove(model, 3);
    end
end
function curvature = estimateLocalCurvature(model, x, y, z)
    % 估计局部曲率
    radius = 3;
    [nx, ny, nz] = size(model);
    
    % 提取局部区域
    x1 = max(1, x-radius);
    x2 = min(nx, x+radius);
    y1 = max(1, y-radius);
    y2 = min(ny, y+radius);
    z1 = max(1, z-radius);
    z2 = min(nz, z+radius);
    
    localRegion = model(x1:x2, y1:y2, z1:z2);
    
    % 简单曲率估计：比较中心与平均
    centerValue = model(x, y, z);
    avgValue = mean(localRegion(:));
    curvature = centerValue - avgValue;
end
function move = generateLocalShapeMove(model, lookupTables)
    % 基于局部形状的移动
    move = struct();
    
    % 使用曲率图找到高曲率区域
    if isfield(lookupTables, 'localCurvature')
        curvatureMap = lookupTables.localCurvature;
        highCurvIdx = find(curvatureMap > quantile(curvatureMap(:), 0.9));
        
        if ~isempty(highCurvIdx)
            % 选择一个高曲率区域
            centerIdx = highCurvIdx(randi(length(highCurvIdx)));
            [cx, cy, cz] = ind2sub(size(model), centerIdx);
            
            % 在该区域进行形状调整
            radius = 3;
            move.linearIdx = [];
            move.oldValues = [];
            move.newValues = [];
            
            for dx = -radius:radius
                for dy = -radius:radius
                    for dz = -radius:radius
                        x = cx + dx;
                        y = cy + dy;
                        z = cz + dz;
                        
                        if x >= 1 && x <= size(model,1) && ...
                            y >= 1 && y <= size(model,2) && ...
                            z >= 1 && z <= size(model,3)
                            
                            dist = sqrt(dx^2 + dy^2 + dz^2);
                            if dist <= radius && rand() < exp(-dist/2)
                                idx = sub2ind(size(model), x, y, z);
                                
                                % 平滑高曲率区域
                                if curvatureMap(idx) > 0 && ~model(idx)
                                    move.linearIdx = [move.linearIdx; idx];
                                    move.oldValues = [move.oldValues; false];
                                    move.newValues = [move.newValues; true];
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    
    if isempty(move.linearIdx)
        move = generateLocalMove(model, 3);
    end
end
function move = generateBoundarySmoothingMove(model)
    % 生成边界平滑移动
    move = struct();
    
    % 找到粗糙的边界区域
    boundary = bwperim(model, 26);
    [x, y, z] = ind2sub(size(model), find(boundary));
    
    if isempty(x)
        move = generateLocalMove(model, 3);
        return;
    end
    
    % 计算边界粗糙度
    roughness = zeros(length(x), 1);
    for i = 1:length(x)
        % 计算邻域变化
        neighbors = 0;
        changes = 0;
        
        for dx = -1:1
            for dy = -1:1
                for dz = -1:1
                    if dx == 0 && dy == 0 && dz == 0
                        continue;
                    end
                    
                    nx = x(i) + dx;
                    ny = y(i) + dy;
                    nz = z(i) + dz;
                    
                    if nx >= 1 && nx <= size(model,1) && ...
                        ny >= 1 && ny <= size(model,2) && ...
                        nz >= 1 && nz <= size(model,3)
                        neighbors = neighbors + 1;
                        if model(nx, ny, nz) ~= model(x(i), y(i), z(i))
                            changes = changes + 1;
                        end
                    end
                end
            end
        end
        
        if neighbors > 0
            roughness(i) = changes / neighbors;
        end
    end
    
    % 选择最粗糙的点进行平滑
    [~, roughIdx] = sort(roughness, 'descend');
    nSmooth = min(5, length(roughIdx));
    
    move.linearIdx = [];
    move.oldValues = [];
    move.newValues = [];
    
    for i = 1:nSmooth
        idx = sub2ind(size(model), x(roughIdx(i)), y(roughIdx(i)), z(roughIdx(i)));
        
        % 根据邻域多数决定
        neighborSum = 0;
        neighborCount = 0;
        
        for dx = -1:1
            for dy = -1:1
                for dz = -1:1
                    if dx == 0 && dy == 0 && dz == 0
                        continue;
                    end
                    
                    nx = x(roughIdx(i)) + dx;
                    ny = y(roughIdx(i)) + dy;
                    nz = z(roughIdx(i)) + dz;
                    
                    if nx >= 1 && nx <= size(model,1) && ...
                        ny >= 1 && ny <= size(model,2) && ...
                        nz >= 1 && nz <= size(model,3)
                        neighborSum = neighborSum + model(nx, ny, nz);
                        neighborCount = neighborCount + 1;
                    end
                end
            end
        end
        
        if neighborCount > 0
            majorityValue = neighborSum > neighborCount/2;
            if model(idx) ~= majorityValue
                move.linearIdx = [move.linearIdx; idx];
                move.oldValues = [move.oldValues; model(idx)];
                move.newValues = [move.newValues; majorityValue];
            end
        end
    end
    
    if isempty(move.linearIdx)
        move = generateLocalMove(model, 3);
    end
end
function move = generateClusterShapeMove(model, optParams)
    % 生成簇形状调整移动
    move = struct();
    
    CC = bwconncomp(model, 26);
    if CC.NumObjects == 0
        move = generateLocalMove(model, 3);
        return;
    end
    
    % 选择一个中等大小的簇
    sizes = cellfun(@numel, CC.PixelIdxList);
    midSizeClusters = find(sizes > optParams.targetMin & sizes < optParams.targetMax);
    
    if isempty(midSizeClusters)
        move = generateLocalMove(model, 3);
        return;
    end
    
    selectedCluster = midSizeClusters(randi(length(midSizeClusters)));
    [x, y, z] = ind2sub(size(model), CC.PixelIdxList{selectedCluster});
    
    % 计算簇的形状特征
    coords = [x - mean(x), y - mean(y), z - mean(z)];
    [V, D] = eig(cov(coords));
    eigenvalues = diag(D);
    
    % 根据特征值调整形状
    move.linearIdx = [];
    move.oldValues = [];
    move.newValues = [];
    
    if eigenvalues(1) / eigenvalues(3) > 3
        % 过于伸长，需要增加宽度
        mainDir = V(:, 3);
        perpDir1 = V(:, 1);
        perpDir2 = V(:, 2);
        
        % 在垂直方向添加体素
        nAdd = min(10, round(sizes(selectedCluster) * 0.1));
        for i = 1:nAdd
            % 随机选择一个簇内点
            baseIdx = randi(length(x));
            basePoint = [x(baseIdx), y(baseIdx), z(baseIdx)] - [mean(x), mean(y), mean(z)];
            
            % 在垂直方向扩展
            offset = (perpDir1 + perpDir2) * randi([-2, 2]);
            newPoint = round(basePoint + offset' + [mean(x), mean(y), mean(z)]);
            
            if all(newPoint >= 1) && newPoint(1) <= size(model,1) && ...
                newPoint(2) <= size(model,2) && newPoint(3) <= size(model,3)
                idx = sub2ind(size(model), newPoint(1), newPoint(2), newPoint(3));
                if ~model(idx)
                    move.linearIdx = [move.linearIdx; idx];
                    move.oldValues = [move.oldValues; false];
                    move.newValues = [move.newValues; true];
                end
            end
        end
    else
        % 形状合理，进行小的调整
        clusterMask = false(size(model));
        clusterMask(CC.PixelIdxList{selectedCluster}) = true;
        boundary = bwperim(clusterMask, 26);
        boundaryIdx = find(boundary);
        
        if ~isempty(boundaryIdx)
            nAdjust = min(5, length(boundaryIdx));
            adjustIdx = boundaryIdx(randperm(length(boundaryIdx), nAdjust));
            move.linearIdx = adjustIdx;
            move.oldValues = model(adjustIdx);
            move.newValues = ~move.oldValues;
        end
    end
    
    if isempty(move.linearIdx)
        move = generateLocalMove(model, 3);
    end
end
function move = generateFineMorphologyMove(model, optParams)
    % 生成精细形态学移动
    move = struct();
    
    % 找到需要形态调整的区域
    CC = bwconncomp(model, 26);
    if CC.NumObjects == 0
        move = generateLocalMove(model, 3);
        return;
    end
    
    % 选择一个簇进行精细调整
    sizes = cellfun(@numel, CC.PixelIdxList);
    midSizeClusters = find(sizes > optParams.targetMin & sizes < optParams.targetMax);
    
    if isempty(midSizeClusters)
        move = generateLocalMove(model, 3);
        return;
    end
    
    selectedCluster = midSizeClusters(randi(length(midSizeClusters)));
    clusterMask = false(size(model));
    clusterMask(CC.PixelIdxList{selectedCluster}) = true;
    
    % 使用3D形态学操作找到需要调整的位置
    se = ones(3,3,3); % 3D结构元素
    dilatedMask = imdilate(clusterMask, se);
    morphGrad = dilatedMask & ~clusterMask; % 形态学梯度（外边界）
    
    adjustPoints = find(morphGrad);
    if isempty(adjustPoints)
        move = generateLocalMove(model, 3);
        return;
    end
    
    % 选择少量点进行调整
    nAdjust = min(5, length(adjustPoints));
    selectedPoints = adjustPoints(randperm(length(adjustPoints), nAdjust));
    
    move.linearIdx = selectedPoints;
    move.oldValues = model(selectedPoints);
    move.newValues = ~move.oldValues;
end
function move = generateSpatialAwareMove(model, lookupTables)
    % 生成空间感知移动
    move = struct();
    
    % 使用空间梯度信息
    if isfield(lookupTables, 'spatialGradient')
        gradientMap = lookupTables.spatialGradient;
        
        % 找到梯度变化大的区域
        highGradIdx = find(gradientMap > quantile(gradientMap(:), 0.8));
        
        if ~isempty(highGradIdx)
            % 选择一个高梯度区域
            centerIdx = highGradIdx(randi(length(highGradIdx)));
            [cx, cy, cz] = ind2sub(size(model), centerIdx);
            
            % 沿梯度方向调整
            radius = 4;
            move.linearIdx = [];
            move.oldValues = [];
            move.newValues = [];
            
            % 计算局部梯度方向
            [gx, gy, gz] = gradient(double(model));
            gradDir = [gx(cx,cy,cz), gy(cx,cy,cz), gz(cx,cy,cz)];
            if norm(gradDir) > 0
                gradDir = gradDir / norm(gradDir);
            else
                gradDir = randn(1, 3);
                gradDir = gradDir / norm(gradDir);
            end
            
            % 沿梯度方向进行调整
            for t = -radius:radius
                pos = round([cx, cy, cz] + t * gradDir);
                
                if all(pos >= 1) && pos(1) <= size(model,1) && ...
                    pos(2) <= size(model,2) && pos(3) <= size(model,3)
                    idx = sub2ind(size(model), pos(1), pos(2), pos(3));
                    
                    % 根据位置决定操作
                    if t < 0 && model(idx)
                        % 梯度下降方向：可能移除
                        if rand() < 0.3
                            move.linearIdx = [move.linearIdx; idx];
                            move.oldValues = [move.oldValues; true];
                            move.newValues = [move.newValues; false];
                        end
                    elseif t > 0 && ~model(idx)
                        % 梯度上升方向：可能添加
                        if rand() < 0.3
                            move.linearIdx = [move.linearIdx; idx];
                            move.oldValues = [move.oldValues; false];
                            move.newValues = [move.newValues; true];
                        end
                    end
                end
            end
        end
    end
    
    if isempty(move.linearIdx)
        move = generateLocalMove(model, 4);
    end
end
function move = generateSpatialCorrelationMove(model, optParams)
    % 生成空间相关性移动
    move = struct();
    move.linearIdx = [];
    move.oldValues = [];
    move.newValues = [];
    
    % 计算当前的两点相关性
    if isfield(optParams.spatialFeatures, 'keyDistances')
        keyDistances = optParams.spatialFeatures.keyDistances;
    else
        keyDistances = [1, 3, 5, 10, 15, 20, 30, 40, 50, 70];
    end
    currentTPC = computeFastTwoPointCorrelation(model, keyDistances);
    targetTPC = optParams.spatialFeatures.twoPointCorr;
    minRows = min(size(currentTPC, 1), size(targetTPC, 1));
    if minRows == 0
        return;
    end
    currentTPC = currentTPC(1:minRows, :);
    targetTPC = targetTPC(1:minRows, :);
    
    % 比较相关性
    tpcDiff = mean(currentTPC(:)) - mean(targetTPC(:));
    
    [nx, ny, nz] = size(model);
    
    if tpcDiff > 0.05
        % 当前相关性太高，增加随机性
        % 在随机位置破坏连续性
        nBreaks = 10;
        for i = 1:nBreaks
            % 随机选择一个位置
            x = randi([3, nx-3]);
            y = randi([3, ny-3]);
            z = randi([2, nz-2]);
            
            % 创建局部扰动
            for dx = -1:1
                for dy = -1:1
                    for dz = -1:1
                        px = x + dx;
                        py = y + dy;
                        pz = z + dz;
                        idx = sub2ind(size(model), px, py, pz);
                        if rand() < 0.3
                            move.linearIdx = [move.linearIdx; idx];
                            move.oldValues = [move.oldValues; model(idx)];
                            move.newValues = [move.newValues; ~model(idx)];
                        end
                    end
                end
            end
        end
    elseif tpcDiff < -0.05
        % 当前相关性太低，增加连续性
        % 找到孤立区域并连接
        CC = bwconncomp(model, 26);
        if CC.NumObjects > 1
            sizes = cellfun(@numel, CC.PixelIdxList);
            smallClusters = find(sizes < mean(sizes));
            
            if ~isempty(smallClusters)
                % 扩展小簇
                clusterIdx = smallClusters(randi(length(smallClusters)));
                clusterMask = false(size(model));
                clusterMask(CC.PixelIdxList{clusterIdx}) = true;
                
                boundary = bwperim(clusterMask, 26);
                growthPoints = find(boundary);
                
                for i = 1:min(10, length(growthPoints))
                    [x, y, z] = ind2sub(size(model), growthPoints(i));
                    
                    % 在边界外生长
                    for dx = -1:1
                        for dy = -1:1
                            for dz = -1:1
                                nx = x + dx;
                                ny = y + dy;
                                nz = z + dz;
                                if nx >= 1 && nx <= size(model,1) && ...
                                    ny >= 1 && ny <= size(model,2) && ...
                                    nz >= 1 && nz <= size(model,3)
                                    idx = sub2ind(size(model), nx, ny, nz);
                                    if ~model(idx) && rand() < 0.5
                                        move.linearIdx = [move.linearIdx; idx];
                                        move.oldValues = [move.oldValues; false];
                                        move.newValues = [move.newValues; true];
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    
    if isempty(move.linearIdx)
        move = generateLocalMove(model, 4);
    end
end
function move = generateAnisotropyAdjustMove(model, lookupTables)
    % 生成各向异性调整移动
    move = struct();
    
    % 使用局部各向异性图
    if isfield(lookupTables, 'localAnisotropy')
        anisoMap = lookupTables.localAnisotropy;
        targetAniso = lookupTables.targetSpatialSignature.anisotropy;
        
        % 找到需要调整的区域
        if targetAniso > mean(anisoMap(:))
            % 需要增加各向异性
            lowAnisoIdx = find(anisoMap < quantile(anisoMap(:), 0.2));
        else
            % 需要减少各向异性
            lowAnisoIdx = find(anisoMap > quantile(anisoMap(:), 0.8));
        end
        
        if ~isempty(lowAnisoIdx)
            centerIdx = lowAnisoIdx(randi(length(lowAnisoIdx)));
            [cx, cy, cz] = ind2sub(size(model), centerIdx);
            
            move.linearIdx = [];
            move.oldValues = [];
            move.newValues = [];
            
            if targetAniso > 0.5
                % 增加Z方向的连续性
                for z = max(1, cz-3):min(size(model,3), cz+3)
                    if z ~= cz
                        % 复制当前层的模式
                        for dx = -2:2
                            for dy = -2:2
                                x = cx + dx;
                                y = cy + dy;
                                if x >= 1 && x <= size(model,1) && ...
                                    y >= 1 && y <= size(model,2)
                                    idx1 = sub2ind(size(model), x, y, cz);
                                    idx2 = sub2ind(size(model), x, y, z);
                                    if model(idx1) ~= model(idx2) && rand() < 0.5
                                        move.linearIdx = [move.linearIdx; idx2];
                                        move.oldValues = [move.oldValues; model(idx2)];
                                        move.newValues = [move.newValues; model(idx1)];
                                    end
                                end
                            end
                        end
                    end
                end
            else
                % 减少各向异性，增加XY平面的变化
                for angle = 0:45:315
                    % 在不同方向创建变化
                    dx = round(3 * cosd(angle));
                    dy = round(3 * sind(angle));
                    x = cx + dx;
                    y = cy + dy;
                    
                    if x >= 1 && x <= size(model,1) && ...
                        y >= 1 && y <= size(model,2)
                        idx = sub2ind(size(model), x, y, cz);
                        move.linearIdx = [move.linearIdx; idx];
                        move.oldValues = [move.oldValues; model(idx)];
                        move.newValues = [move.newValues; ~model(idx)];
                    end
                end
            end
        end
    end
    
    if isempty(move.linearIdx)
        move = generateLocalMove(model, 3);
    end
end
function move = generateConnectivityEnhanceMove(model, optParams)
    % 生成增强连通性的移动
    move = struct();
    
    CC = bwconncomp(model, 26);
    if CC.NumObjects <= 1
        move = generateLocalMove(model, 3);
        return;
    end
    
    % 找到较大的非连通组分
    sizes = cellfun(@numel, CC.PixelIdxList);
    [sortedSizes, sortIdx] = sort(sizes, 'descend');
    
    % 尝试连接前两个最大的组分
    if CC.NumObjects >= 2 && sortedSizes(2) > optParams.targetMin
        [dist, point1, point2] = findClosestPoints(model, ...
            CC.PixelIdxList{sortIdx(1)}, CC.PixelIdxList{sortIdx(2)});
        
        if dist < 10 && dist > 2
            % 创建连接路径
            move = createConnectionMove(point1, point2, model);
            return;
        end
    end
    
    move = generateLocalMove(model, 3);
end
function move = createConnectionMove(point1, point2, model)
    % 创建连接两点的移动
    move = struct();
    move.linearIdx = [];
    move.oldValues = [];
    move.newValues = [];
    
    nSteps = ceil(norm(point1 - point2) * 1.5);
    
    for t = 0:1/nSteps:1
        pos = round(point1 * (1-t) + point2 * t);
        
        if pos(1) >= 1 && pos(1) <= size(model, 1) && ...
            pos(2) >= 1 && pos(2) <= size(model, 2) && ...
            pos(3) >= 1 && pos(3) <= size(model, 3)
            
            idx = sub2ind(size(model), pos(1), pos(2), pos(3));
            if ~model(idx)
                move.linearIdx = [move.linearIdx; idx];
                move.oldValues = [move.oldValues; false];
                move.newValues = [move.newValues; true];
            end
        end
    end
    
    if isempty(move.linearIdx)
        move = generateLocalMove(model, 3);
    end
end
function move = generateStructureCoherenceMove(model, lookupTables)
    % 生成结构连贯性移动
    move = struct();
    
    % 使用结构连贯性图找到需要改善的区域
    coherenceMap = lookupTables.structureCoherenceMap;
    lowCoherenceIdx = find(coherenceMap < quantile(coherenceMap(:), 0.2));
    
    if isempty(lowCoherenceIdx)
        move = generateLocalMove(model, 5);
        return;
    end
    
    % 选择一个低连贯性区域
    centerIdx = lowCoherenceIdx(randi(length(lowCoherenceIdx)));
    [cx, cy, cz] = ind2sub(size(model), centerIdx);
    
    % 在该区域进行结构化调整
    radius = 4;
    move.linearIdx = [];
    move.oldValues = [];
    move.newValues = [];
    
    % 计算局部主方向
    localWindow = model(max(1,cx-radius):min(end,cx+radius), ...
        max(1,cy-radius):min(end,cy+radius), ...
        max(1,cz-radius):min(end,cz+radius));
    
    [gx, gy, gz] = gradient(double(localWindow));
    mainDir = [mean(gx(:)), mean(gy(:)), mean(gz(:))];
    mainDir = mainDir / (norm(mainDir) + eps);
    
    % 沿主方向进行调整
    for i = -radius:radius
        pos = round([cx, cy, cz] + i * mainDir);
        
        if all(pos >= 1) && pos(1) <= size(model, 1) && ...
            pos(2) <= size(model, 2) && pos(3) <= size(model, 3)
            
            idx = sub2ind(size(model), pos(1), pos(2), pos(3));
            
            % 根据位置决定操作
            if abs(i) < radius/2 && ~model(idx)
                % 中心区域：添加体素
                move.linearIdx = [move.linearIdx; idx];
                move.oldValues = [move.oldValues; false];
                move.newValues = [move.newValues; true];
            elseif abs(i) > radius/2 && model(idx) && rand() < 0.3
                % 边缘区域：可能移除体素
                move.linearIdx = [move.linearIdx; idx];
                move.oldValues = [move.oldValues; true];
                move.newValues = [move.newValues; false];
            end
        end
    end
    
    if isempty(move.linearIdx)
        move = generateLocalMove(model, 3);
    end
end
function move = generateLargeScaleMove(model, optParams)
    % 生成大规模移动
    move = struct();
    
    % 随机选择一个较大的区域
    [nx, ny, nz] = size(model);
    regionSize = 10;
    
    cx = randi([regionSize, nx-regionSize]);
    cy = randi([regionSize, ny-regionSize]);
    cz = randi([regionSize/2, nz-regionSize/2]);
    
    % 在该区域进行大规模调整
    move.linearIdx = [];
    move.oldValues = [];
    move.newValues = [];
    
    for x = cx-regionSize:cx+regionSize
        for y = cy-regionSize:cy+regionSize
            for z = cz-regionSize/2:cz+regionSize/2
                if x >= 1 && x <= nx && y >= 1 && y <= ny && z >= 1 && z <= nz
                    % 根据距离中心的距离决定翻转概率
                    dist = sqrt((x-cx)^2 + (y-cy)^2 + (z-cz)^2);
                    flipProb = exp(-dist^2 / (2*(regionSize/2)^2)) * 0.3;
                    
                    if rand() < flipProb
                        idx = sub2ind(size(model), x, y, z);
                        move.linearIdx = [move.linearIdx; idx];
                        move.oldValues = [move.oldValues; model(idx)];
                        move.newValues = [move.newValues; ~model(idx)];
                    end
                end
            end
        end
    end
    
    if isempty(move.linearIdx)
        move = generateLocalMove(model, 5);
    end
end
function move = generateFineTuneMove(model)
    % 生成精细调整移动
    move = struct();
    
    % 找到边界区域
    boundary = bwperim(model, 26);
    boundaryIdx = find(boundary);
    
    if isempty(boundaryIdx)
        move = generateLocalMove(model, 2);
        return;
    end
    
    % 选择少量边界点进行精细调整
    nAdjust = min(3, length(boundaryIdx));
    selectedIdx = boundaryIdx(randperm(length(boundaryIdx), nAdjust));
    
    move.linearIdx = selectedIdx;
    move.oldValues = model(selectedIdx);
    move.newValues = ~move.oldValues;
end
function move = generateClusterMove(model, optParams)
    % 生成簇操作移动
    move = struct();
    
    CC = bwconncomp(model, 26);
    if CC.NumObjects == 0
        move = generateLocalMove(model, 4);
        return;
    end
    
    sizes = cellfun(@numel, CC.PixelIdxList);
    
    % 根据簇大小选择操作
    if any(sizes > optParams.targetMax)
        % 缩小过大的簇
        largeClusterIdx = find(sizes > optParams.targetMax, 1);
        clusterMask = false(size(model));
        clusterMask(CC.PixelIdxList{largeClusterIdx}) = true;
        
        % 从边界移除一些体素
        boundary = bwperim(clusterMask, 26);
        removePoints = find(boundary & clusterMask);
        
        if ~isempty(removePoints)
            nRemove = min(10, length(removePoints));
            selectedPoints = removePoints(randperm(length(removePoints), nRemove));
            move.linearIdx = selectedPoints;
            move.oldValues = true(length(selectedPoints), 1);
            move.newValues = false(length(selectedPoints), 1);
        else
            move = generateLocalMove(model, 3);
        end
        
    elseif any(sizes < optParams.targetMin)
        % 扩大过小的簇
        smallClusterIdx = find(sizes < optParams.targetMin, 1);
        clusterMask = false(size(model));
        clusterMask(CC.PixelIdxList{smallClusterIdx}) = true;
        
        % 在边界外添加一些体素
        dilated = imdilate(clusterMask, ones(3,3,3));
        growthPoints = find(dilated & ~model);
        
        if ~isempty(growthPoints)
            nGrow = min(10, length(growthPoints));
            selectedPoints = growthPoints(randperm(length(growthPoints), nGrow));
            move.linearIdx = selectedPoints;
            move.oldValues = false(length(selectedPoints), 1);
            move.newValues = true(length(selectedPoints), 1);
        else
            move = generateLocalMove(model, 3);
        end
        
    else
        % 簇大小合适，进行形状优化
        move = generateClusterShapeMove(model, optParams);
    end
end
function move = generateBoundaryMove(model)
    % 生成边界移动
    move = struct();
    
    % 找到边界
    boundary = bwperim(model, 26);
    boundaryIdx = find(boundary);
    
    if isempty(boundaryIdx)
        move = generateLocalMove(model, 3);
        return;
    end
    
    % 随机选择边界操作类型
    operationType = randi(3);
    
    switch operationType
        case 1
            % 边界生长
            dilated = imdilate(boundary, ones(3,3,3));
            growthPoints = find(dilated & ~model);
            
            if ~isempty(growthPoints)
                nGrow = min(10, length(growthPoints));
                selectedPoints = growthPoints(randperm(length(growthPoints), nGrow));
                move.linearIdx = selectedPoints;
                move.oldValues = false(length(selectedPoints), 1);
                move.newValues = true(length(selectedPoints), 1);
            else
                move = generateLocalMove(model, 3);
            end
            
        case 2
            % 边界收缩
            shrinkPoints = find(boundary & model);
            
            if ~isempty(shrinkPoints)
                nShrink = min(10, length(shrinkPoints));
                selectedPoints = shrinkPoints(randperm(length(shrinkPoints), nShrink));
                move.linearIdx = selectedPoints;
                move.oldValues = true(length(selectedPoints), 1);
                move.newValues = false(length(selectedPoints), 1);
            else
                move = generateLocalMove(model, 3);
            end
            
        case 3
            % 边界平滑
            move = generateBoundarySmoothingMove(model);
    end
end
%% ========== 评估和应用函数 ==========
function deltaEnergies = evaluateComprehensiveBatchMoves(mcmcState, moves, ...
    moveTypes, lookupTables, phase)
    % 评估综合批量移动
    nMoves = length(moves);
    deltaEnergies = zeros(nMoves, 1);
    
    for i = 1:nMoves
        try
            % 根据移动类型和当前阶段评估能量变化
            deltaEnergies(i) = evaluateComprehensiveMoveDelta(mcmcState, moves{i}, ...
                moveTypes{i}, lookupTables, phase);
        catch ME
            % 如果评估失败，使用简化方法
            warning('评估移动 %d 时出错: %s', i, ME.message);
            deltaEnergies(i) = evaluateSimpleMoveDelta(mcmcState, moves{i});
        end
    end
end
function deltaE = evaluateComprehensiveMoveDelta(mcmcState, move, moveType, ...
    lookupTables, phase)
    % 评估综合移动的能量变化
    
    % 基础能量变化
    baseDE = evaluateBaseMoveDelta(mcmcState, move);
    
    % 根据移动类型和阶段调整
    switch moveType
        case {'morphology_preserving', 'local_shape', 'boundary_smooth', ...
                'cluster_shape', 'fine_morphology'}
            % 形态相关移动
            if strcmp(phase, 'morphology')
                deltaE = baseDE - 2.0; % 强烈鼓励
            else
                deltaE = baseDE * 0.8;
            end
            
        case {'spatial_aware', 'spatial_correlation', 'anisotropy_adjust', ...
                'connectivity_enhance'}
            % 空间相关移动
            if strcmp(phase, 'spatial')
                deltaE = baseDE - 2.0; % 强烈鼓励
            else
                deltaE = baseDE * 0.8;
            end
            
        case 'structure_coherence'
            % 结构连贯性移动在所有阶段都有益
            deltaE = baseDE - 0.5;
            
        case 'large_scale'
            % 大规模移动需要谨慎
            deltaE = baseDE * 1.2;
            
        otherwise
            deltaE = baseDE;
    end
    
    % 添加随机扰动避免局部最优
    deltaE = deltaE * (0.9 + 0.2 * rand());
end
function deltaE = evaluateBaseMoveDelta(mcmcState, move)
    % 评估基础移动能量变化
    if isempty(move.linearIdx)
        deltaE = 0;
        return;
    end
    
    % 快速估计能量变化
    nChanged = length(move.linearIdx);
    nFlipped = sum(move.newValues) - sum(move.oldValues);
    
    % 孔隙率变化
    deltaPorosity = nFlipped / numel(mcmcState.model);
    porosityDiff = abs(deltaPorosity);
    targetPorosity = mcmcState.optParams.targetPorosity;
    currentPorosity = mean(mcmcState.model(:));
    
    % 如果变化会使孔隙率偏离目标，增加惩罚
    newPorosity = currentPorosity + deltaPorosity;
    currentError = abs(currentPorosity - targetPorosity);
    newError = abs(newPorosity - targetPorosity);
    
    deltaE_porosity = (newError - currentError) * mcmcState.weights.porosity;
    
    % 表面积变化估计
    boundary = bwperim(mcmcState.model, 26);
    boundaryVec = boundary(:);
    
    % 创建标记向量
    changeMarker = false(numel(mcmcState.model), 1);
    changeMarker(move.linearIdx) = true;
    
    % 计算边界上的变化数量
    boundaryChange = sum(boundaryVec & changeMarker);
    
    % 估计表面积变化
    surfaceChangeEstimate = 0;
    for i = 1:length(move.linearIdx)
        idx = move.linearIdx(i);
        if boundaryVec(idx)
            % 边界体素
            if move.newValues(i)
                surfaceChangeEstimate = surfaceChangeEstimate - 1; % 填充边界减少表面
            else
                surfaceChangeEstimate = surfaceChangeEstimate + 1; % 移除边界增加表面
            end
        else
            % 内部体素，检查是否会创建新边界
            [x, y, z] = ind2sub(size(mcmcState.model), idx);
            neighborCount = 0;
            
            % 检查6邻域
            for dx = [-1, 1]
                nx = x + dx;
                if nx >= 1 && nx <= size(mcmcState.model, 1)
                    neighborCount = neighborCount + mcmcState.model(nx, y, z);
                end
            end
            
            for dy = [-1, 1]
                ny = y + dy;
                if ny >= 1 && ny <= size(mcmcState.model, 2)
                    neighborCount = neighborCount + mcmcState.model(x, ny, z);
                end
            end
            
            for dz = [-1, 1]
                nz = z + dz;
                if nz >= 1 && nz <= size(mcmcState.model, 3)
                    neighborCount = neighborCount + mcmcState.model(x, y, nz);
                end
            end
            
            % 如果翻转会创建新边界
            if move.newValues(i) && neighborCount < 6
                surfaceChangeEstimate = surfaceChangeEstimate + 0.5;
            elseif ~move.newValues(i) && neighborCount > 0
                surfaceChangeEstimate = surfaceChangeEstimate + 0.5;
            end
        end
    end
    
    deltaE_surface = abs(surfaceChangeEstimate) / numel(mcmcState.model) * mcmcState.weights.morphology;
    
    % 簇大小变化的粗略估计
    clusterImpact = 0;
    if nChanged > 20
        clusterImpact = 0.1; % 大变化可能破坏簇结构
    elseif nChanged > 10
        clusterImpact = 0.05;
    end
    
    deltaE_cluster = clusterImpact * mcmcState.weights.cluster;
    
    % 空间相关性影响
    if length(move.linearIdx) > 1
        [xs, ys, zs] = ind2sub(size(mcmcState.model), move.linearIdx);
        spatialSpread = std(xs) + std(ys) + std(zs);
        normalizedSpread = spatialSpread / (size(mcmcState.model, 1) + size(mcmcState.model, 2) + size(mcmcState.model, 3));
        
        % 集中的移动更好
        deltaE_spatial = normalizedSpread * mcmcState.weights.spatial * 0.1;
    else
        deltaE_spatial = 0;
    end
    
    % 总能量变化
    deltaE = deltaE_porosity + deltaE_surface + deltaE_cluster + deltaE_spatial;
end
function deltaE = evaluateSimpleMoveDelta(mcmcState, move)
    % 简单的能量变化评估（用于错误恢复）
    if isempty(move.linearIdx)
        deltaE = 0;
        return;
    end
    
    % 仅考虑孔隙率变化
    nFlipped = sum(move.newValues) - sum(move.oldValues);
    deltaPorosity = nFlipped / numel(mcmcState.model);
    
    currentPorosity = mean(mcmcState.model(:));
    targetPorosity = mcmcState.optParams.targetPorosity;
    
    currentError = abs(currentPorosity - targetPorosity);
    newError = abs(currentPorosity + deltaPorosity - targetPorosity);
    
    deltaE = (newError - currentError) * mcmcState.weights.porosity;
    
    % 添加小的随机项避免停滞
    deltaE = deltaE + 0.01 * randn();
end
function [mcmcState, accepted] = applyBestComprehensiveMoves(mcmcState, moves, ...
    deltaEnergies, temperature, moveTypes)
    % 应用最佳综合移动
    accepted = false;
    
    % 过滤掉无效的能量值
    validIdx = ~isnan(deltaEnergies) & ~isinf(deltaEnergies);
    if ~any(validIdx)
        % 没有有效的移动
        return;
    end
    
    validEnergies = deltaEnergies(validIdx);
    validMoves = moves(validIdx);
    validTypes = moveTypes(validIdx);
    
    % 找到最佳移动
    [minDeltaE, bestIdx] = min(validEnergies);
    bestMove = validMoves{bestIdx};
    bestType = validTypes{bestIdx};
    
    % Metropolis准则
    acceptProb = exp(-minDeltaE/temperature);
    
    if minDeltaE < 0 || rand() < acceptProb
        % 接受移动
        if isfield(bestMove, 'linearIdx') && ~isempty(bestMove.linearIdx)
            % 验证移动的有效性
            validIndices = bestMove.linearIdx > 0 & bestMove.linearIdx <= numel(mcmcState.model);
            if any(validIndices)
                validLinearIdx = bestMove.linearIdx(validIndices);
                validNewValues = bestMove.newValues(validIndices);
                
                % 应用移动
                mcmcState.model(validLinearIdx) = validNewValues;
                
                % 更新能量（使用估计值）
                mcmcState.currentEnergy = mcmcState.currentEnergy + minDeltaE;
                
                % 更新最佳模型
                if mcmcState.currentEnergy < mcmcState.bestEnergy
                    mcmcState.bestModel = mcmcState.model;
                    mcmcState.bestEnergy = mcmcState.currentEnergy;
                end
                
                accepted = true;
                
                % 记录接受的移动类型
                if isfield(mcmcState, 'acceptedMoveTypes')
                    if isfield(mcmcState.acceptedMoveTypes, bestType)
                        mcmcState.acceptedMoveTypes.(bestType) = ...
                            mcmcState.acceptedMoveTypes.(bestType) + 1;
                    else
                        mcmcState.acceptedMoveTypes.(bestType) = 1;
                    end
                end
            end
        end
    end
    
    % 更新接受率
    if ~isfield(mcmcState, 'moveHistory')
        mcmcState.moveHistory = zeros(1, 10);
    end
    mcmcState.moveHistory = [mcmcState.moveHistory(2:end), accepted];
    mcmcState.acceptanceRate = mean(mcmcState.moveHistory);
    
    % 更新能量趋势
    if ~isfield(mcmcState, 'energyTrend')
        mcmcState.energyTrend = zeros(1, 10);
    end
    mcmcState.energyTrend = [mcmcState.energyTrend(2:end), mcmcState.currentEnergy];
    
    % 定期完整更新特征（降低频率以提高性能）
    if rand() < 0.02 % 2%的概率
        mcmcState = updateAllFeatures(mcmcState);
    end
end
%% ========== 性能监控和自适应调整 ==========
function performanceMonitor = updatePerformanceMetrics(performanceMonitor, mcmcState, idx)
    % 更新性能指标
    
    % 空间匹配度
    spatialMatch = calculateSpatialMatch(mcmcState.spatialFeatures, ...
        mcmcState.optParams.spatialFeatures);
    performanceMonitor.spatialMatchHistory(idx) = spatialMatch;
    
    % 形态匹配度
    morphMatch = calculateMorphologyMatch(mcmcState.morphologyFeatures, ...
        mcmcState.optParams.morphologyFeatures);
    performanceMonitor.morphologyMatchHistory(idx) = morphMatch;
    
    % 簇大小统计
    if ~isempty(mcmcState.features.sizes)
        performanceMonitor.clusterSizeHistory(:, idx) = ...
            [mcmcState.features.sizeStats(1); mcmcState.features.sizeStats(2); ...
            mcmcState.features.sizeStats(3)];
    end
    
    % 各向异性
    performanceMonitor.anisotropyHistory(idx) = mcmcState.spatialFeatures.anisotropy;
    
    % 连通性
    if isfield(mcmcState.spatialFeatures, 'connectivity')
        performanceMonitor.connectivityHistory(idx) = ...
            mcmcState.spatialFeatures.connectivity.largestComponentRatio;
    end
end
function mcmcState = comprehensiveAdaptiveAdjustment(mcmcState, performanceMonitor, ...
    iter, phase)
    % 综合自适应调整
    
    % 检查能量停滞
    if std(mcmcState.energyTrend) < 1e-6
        fprintf('  检测到能量停滞，进行自适应调整...\n');
        
        % 根据阶段选择扰动策略
        switch phase
            case 'morphology'
                mcmcState.model = morphologyGuidedPerturbation(mcmcState.model, ...
                    mcmcState.optParams);
            case 'spatial'
                mcmcState.model = largeSpatialPerturbation(mcmcState.model, ...
                    mcmcState.optParams);
            case 'balanced'
                mcmcState.model = balancedPerturbation(mcmcState.model, ...
                    mcmcState.optParams);
        end
        
        % 更新特征
        mcmcState = updateAllFeatures(mcmcState);
    end
    
    % 温度调整
    if mcmcState.acceptanceRate < 0.1
        mcmcState.temperature = mcmcState.temperature * 1.2;
        fprintf('  接受率过低，升温至 %.6f\n', mcmcState.temperature);
    elseif mcmcState.acceptanceRate > 0.9
        mcmcState.temperature = mcmcState.temperature * 0.8;
        fprintf('  接受率过高，降温至 %.6f\n', mcmcState.temperature);
    end
    
    % 检查特定阶段的进展
    if iter > 1000
        checkPhaseProgress(mcmcState, performanceMonitor, phase);
    end
end
function model = morphologyGuidedPerturbation(model, optParams)
    % 形态学引导的扰动
    targetMorph = optParams.morphologyFeatures;
    
    % 选择扰动策略
    if ~isempty(targetMorph.sphericity)
        targetSpher = mean(targetMorph.sphericity);
        
        if targetSpher > 0.7
            % 高球形度目标 - 使用球形扰动
            nSpheres = 5;
            for i = 1:nSpheres
                % 随机位置添加球形结构
                cx = randi([10, size(model,1)-10]);
                cy = randi([10, size(model,2)-10]);
                cz = randi([5, size(model,3)-5]);
                radius = randi([3, 6]);
                
                [x, y, z] = meshgrid(1:size(model,1), 1:size(model,2), 1:size(model,3));
                sphere = sqrt((x-cx).^2 + (y-cy).^2 + (z-cz).^2) <= radius;
                
                if rand() > 0.5
                    model = model | sphere';
                else
                    model = model & ~sphere';
                end
            end
        else
            % 低球形度目标 - 使用伸长结构
            nRods = 3;
            for i = 1:nRods
                % 随机位置和方向的杆状结构
                cx = randi([10, size(model,1)-10]);
                cy = randi([10, size(model,2)-10]);
                cz = randi([5, size(model,3)-5]);
                
                direction = randn(1, 3);
                direction = direction / norm(direction);
                length = randi([10, 20]);
                
                for t = -length/2:length/2
                    pos = round([cx, cy, cz] + t * direction);
                    if all(pos >= 1) & pos(1) <= size(model,1) & ...
                        pos(2) <= size(model,2) & pos(3) <= size(model,3)
                        model(pos(1), pos(2), pos(3)) = true;
                    end
                end
            end
        end
    end
    
    % 保持孔隙率
    model = adjustPorosity(model, optParams.targetPorosity);
end
function model = largeSpatialPerturbation(model, optParams)
    % 大规模空间扰动
    [nx, ny, nz] = size(model);
    
    % 创建具有目标空间特征的扰动场
    targetAniso = optParams.spatialFeatures.anisotropy;
    correlationLength = 10;
    
    % 生成相关噪声场
    noise = randn(nx, ny, nz);
    
    if targetAniso > 0.5
        % 各向异性滤波
        for z = 1:nz
            noise(:,:,z) = imgaussfilt(noise(:,:,z), 3);
        end
    else
        % 各向同性滤波
        noise = imgaussfilt3(noise, 3);
    end
    
    % 应用扰动
    threshold = quantile(noise(:), 0.7);
    perturbMask = noise > threshold;
    
    % 混合扰动
    alpha = 0.3;
    model = (1-alpha) * double(model) + alpha * double(perturbMask);
    model = model > 0.5;
    
    % 保持孔隙率
    model = adjustPorosity(model, optParams.targetPorosity);
end
function model = balancedPerturbation(model, optParams)
    % 平衡扰动
    % 结合形态和空间特征进行扰动
    nRegions = 3;
    [nx, ny, nz] = size(model);
    
    for r = 1:nRegions
        % 随机选择扰动类型
        perturbType = randi(3);
        
        switch perturbType
            case 1
                % 形态扰动
                model = localMorphologyPerturbation(model, optParams);
                
            case 2
                % 空间扰动
                model = localSpatialPerturbation(model, optParams);
                
            case 3
                % 混合扰动
                regionSize = 15;
                cx = randi([regionSize, nx-regionSize]);
                cy = randi([regionSize, ny-regionSize]);
                cz = randi([regionSize/2, nz-regionSize/2]);
                
                % 创建目标特征的局部结构
                localStructure = createTargetStructure(regionSize, optParams);
                model = blendStructureIntoModel(model, localStructure, cx, cy, cz);
        end
    end
end
function model = localMorphologyPerturbation(model, optParams)
    % 局部形态扰动
    CC = bwconncomp(model, 26);
    if CC.NumObjects == 0
        return;
    end
    
    % 选择一个簇进行形态调整
    idx = randi(CC.NumObjects);
    clusterMask = false(size(model));
    clusterMask(CC.PixelIdxList{idx}) = true;
    
    % 应用目标形态
    targetSphericity = mean(optParams.morphologyFeatures.sphericity);
    if targetSphericity > 0.6
        se = strel('sphere', 1);
        adjusted = imclose(clusterMask, se);
    else
        se = strel('line', 3, 0);
        adjusted = imerode(clusterMask, se);
        adjusted = imdilate(adjusted, se);
    end
    
    % 混合调整结果
    blendMask = rand(size(model)) < 0.5;
    model(blendMask & clusterMask) = adjusted(blendMask & clusterMask);
end
function model = localSpatialPerturbation(model, optParams)
    % 局部空间扰动
    targetAniso = optParams.spatialFeatures.anisotropy;
    [nx, ny, nz] = size(model);
    
    % 选择一个区域
    regionSize = 20;
    cx = randi([regionSize, nx-regionSize]);
    cy = randi([regionSize, ny-regionSize]);
    cz = randi([regionSize/2, nz-regionSize/2]);
    
    % 在该区域应用方向性操作
    if targetAniso > 0.5
        % 增强Z方向的连续性
        for z = cz-regionSize/2:cz+regionSize/2
            if z >= 1 && z <= nz
                slice = model(cx-regionSize:cx+regionSize, ...
                    cy-regionSize:cy+regionSize, z);
                if z > 1
                    prevSlice = model(cx-regionSize:cx+regionSize, ...
                        cy-regionSize:cy+regionSize, z-1);
                    % 增加与前一层的相似性
                    similarity = rand(size(slice)) < 0.7;
                    slice(similarity) = prevSlice(similarity);
                    model(cx-regionSize:cx+regionSize, ...
                        cy-regionSize:cy+regionSize, z) = slice;
                end
            end
        end
    else
        % 增加各向同性
        region = model(cx-regionSize:cx+regionSize, ...
            cy-regionSize:cy+regionSize, ...
            cz-regionSize/2:cz+regionSize/2);
        region = medfilt3(double(region), [3, 3, 3]) > 0.5;
        model(cx-regionSize:cx+regionSize, ...
            cy-regionSize:cy+regionSize, ...
            cz-regionSize/2:cz+regionSize/2) = region;
    end
end
function checkPhaseProgress(mcmcState, performanceMonitor, phase)
    % 检查阶段进展
    recentIdx = max(1, length(performanceMonitor.morphologyMatchHistory)-5):...
        length(performanceMonitor.morphologyMatchHistory);
    
    if isempty(recentIdx)
        return;
    end
    
    switch phase
        case 'morphology'
            recentMorphMatch = performanceMonitor.morphologyMatchHistory(recentIdx);
            if std(recentMorphMatch) < 0.01 && mean(recentMorphMatch) < 0.6
                fprintf('  形态匹配进展缓慢，考虑调整策略...\n');
            end
            
        case 'spatial'
            recentSpatialMatch = performanceMonitor.spatialMatchHistory(recentIdx);
            if std(recentSpatialMatch) < 0.01 && mean(recentSpatialMatch) < 0.6
                fprintf('  空间匹配进展缓慢，考虑调整策略...\n');
            end
            
        case 'balanced'
            morphProgress = mean(performanceMonitor.morphologyMatchHistory(recentIdx));
            spatialProgress = mean(performanceMonitor.spatialMatchHistory(recentIdx));
            if morphProgress < 0.7 || spatialProgress < 0.7
                fprintf('  综合匹配未达预期 (形态: %.3f, 空间: %.3f)\n', ...
                    morphProgress, spatialProgress);
            end
    end
end
%% ========== 进度显示和可视化 ==========
function printComprehensiveProgress(iter, mcmcState, elapsedTime, phase)
    % 打印综合进度
    fprintf('\n========== 迭代 %d ==========\n', iter);
    fprintf('优化阶段: %s\n', phase);
    fprintf('当前能量: %.6f | 最佳能量: %.6f\n', ...
        mcmcState.currentEnergy, mcmcState.bestEnergy);
    fprintf('温度: %.6f | 接受率: %.2f%%\n', ...
        mcmcState.temperature, mcmcState.acceptanceRate * 100);
    fprintf('已用时间: %.2f秒\n', elapsedTime);
    
    % 显示匹配度
    morphMatch = calculateMorphologyMatch(mcmcState.morphologyFeatures, ...
        mcmcState.optParams.morphologyFeatures);
    spatialMatch = calculateSpatialMatch(mcmcState.spatialFeatures, ...
        mcmcState.optParams.spatialFeatures);
    
    fprintf('\n匹配度指标：\n');
    fprintf('  形态匹配度: %.3f\n', morphMatch);
    fprintf('  空间匹配度: %.3f\n', spatialMatch);
    
    % 显示关键特征
    if ~isempty(mcmcState.features.sizes)
        fprintf('\n簇统计：\n');
        fprintf('  簇数量: %d\n', mcmcState.features.numClusters);
        fprintf('  簇大小: [%d, %d] (目标: [%d, %d])\n', ...
            mcmcState.features.sizeStats(1), mcmcState.features.sizeStats(2), ...
            mcmcState.optParams.targetMin, mcmcState.optParams.targetMax);
    end
    fprintf('  当前孔隙率: %.4f (目标: %.4f)\n', ...
        mean(mcmcState.model(:)), mcmcState.optParams.targetPorosity);
end
function visualizeComprehensiveModelSlices(model, iter, phase)
    % 综合可视化
    try
        figure(300); clf;
        
        % 设置标题
        sgtitle(sprintf('迭代 %d - %s阶段', iter, phase));
        
        % 显示三个正交切片
        [nx, ny, nz] = size(model);
        
        subplot(2,3,1);
        imshow(squeeze(model(round(nx/2),:,:)), []);
        title('YZ切面');
        
        subplot(2,3,2);
        imshow(squeeze(model(:,round(ny/2),:)), []);
        title('XZ切面');
        
        subplot(2,3,3);
        imshow(model(:,:,round(nz/2)), []);
        title('XY切面');
        
        % 3D渲染（如果可用）
        subplot(2,3,4:6);
        try
            isosurface(model, 0.5);
            axis equal;
            view(3);
            camlight;
            lighting gouraud;
            title('3D结构');
        catch
            % 如果3D渲染失败，显示投影
            projXY = sum(model, 3);
            imagesc(projXY);
            colormap gray;
            axis equal;
            title('XY投影');
        end
        
        drawnow;
        
    catch
        % 忽略可视化错误
    end
end
%% ========== 改进的后处理函数 ==========
function finalModel = comprehensivePostProcessWithSmallPores(model, optParams)
    % 综合后处理 - 改进版，更好地匹配原始孔隙形态
    fprintf('执行综合后处理（形态保持版本）...\n');
    
    % 保存原始孔隙率
    originalPorosity = mean(model(:));
    
    % 1. 强化噪声去除
    fprintf('  1. 强化噪声去除...\n');
    model = enhancedNoiseRemoval(model, optParams);
    
    % 2. 形态学重建
    fprintf('  2. 形态学重建...\n');
    model = morphologicalReconstruction(model, optParams);
    
    % 3. 基于参考的形态恢复
    fprintf('  3. 基于参考的形态恢复...\n');
    model = referenceBasedMorphologyRestoration(model, optParams.originalBinaryModel);
    
    % 4. 多尺度边界平滑
    fprintf('  4. 多尺度边界平滑...\n');
    model = multiScaleBoundarySmoothing(model);
    
    % 5. 结构优化
    fprintf('  5. 结构优化...\n');
    model = structureOptimization(model, optParams);
    
    % 6. 最终孔隙率调整
    fprintf('  6. 恢复目标孔隙率...\n');
    model = smartPorosityAdjustment(model, originalPorosity, optParams);
    
    % 7. 最终质量检查和微调
    fprintf('  7. 最终质量检查...\n');
    model = finalQualityRefinement(model, optParams);
    model = enforceTargetClusterCount(model, optParams);
    
    % 统计最终结果
    reportFinalStatistics(model, optParams);
    
    finalModel = model;
    fprintf('综合后处理完成\n');
end
function finalModel = comprehensivePostProcess(model, optParams)
    % 综合后处理 - 标准平滑版本
    fprintf('执行综合后处理（标准平滑版本）...\n');
    
    % 保存原始孔隙率
    originalPorosity = mean(model(:));
    
    % 1. 激进的噪声去除
    fprintf('  1. 激进噪声去除...\n');
    model = aggressiveNoiseRemoval(model, optParams);
    
    % 2. 强化形态学操作
    fprintf('  2. 强化形态学操作...\n');
    model = strongMorphologicalOperations(model);
    
    % 3. 深度边界平滑
    fprintf('  3. 深度边界平滑...\n');
    model = deepBoundarySmoothing(model);
    
    % 4. 结构简化
    fprintf('  4. 结构简化...\n');
    model = structureSimplification(model, optParams);
    
    % 5. 最终孔隙率调整
    fprintf('  5. 恢复目标孔隙率...\n');
    model = adjustPorosity(model, originalPorosity);
    model = enforceTargetClusterCount(model, optParams);
    
    finalModel = model;
    fprintf('综合后处理完成（平滑版本）\n');
end
%% ========== 核心后处理函数 ==========
function model = enhancedNoiseRemoval(model, optParams)
    % 强化的噪声去除
    % 移除小于目标最小值一定比例的所有小孔隙
    minThreshold = max(10, optParams.targetMin / 5);
    
    % 第一轮：去除极小孔隙
    CC = bwconncomp(model, 26);
    if CC.NumObjects > 0
        sizes = cellfun(@numel, CC.PixelIdxList);
        removeIdx = find(sizes < minThreshold);
        fprintf('    移除%d个小孔隙（<%d体素）\n', length(removeIdx), minThreshold);
        for i = 1:length(removeIdx)
            model(CC.PixelIdxList{removeIdx(i)}) = false;
        end
    end
    
    % 第二轮：填充小孔洞
    invertedModel = ~model;
    CC_inv = bwconncomp(invertedModel, 26);
    if CC_inv.NumObjects > 0
        sizes_inv = cellfun(@numel, CC_inv.PixelIdxList);
        fillIdx = find(sizes_inv < minThreshold);
        fprintf('    填充%d个小孔洞（<%d体素）\n', length(fillIdx), minThreshold);
        for i = 1:length(fillIdx)
            model(CC_inv.PixelIdxList{fillIdx(i)}) = true;
        end
    end
    
    % 第三轮：3D中值滤波去除椒盐噪声
    fprintf('    应用3D中值滤波...\n');
    model = medfilt3(double(model), [3, 3, 3]) > 0.5;
end
function model = morphologicalReconstruction(model, optParams)
    % 形态学重建 - 恢复主要孔隙形态
    fprintf('    执行形态学重建...\n');
    
    % 标记主要孔隙（大于平均大小的孔隙）
    CC = bwconncomp(model, 26);
    if CC.NumObjects == 0
        return;
    end
    
    sizes = cellfun(@numel, CC.PixelIdxList);
    avgSize = mean(sizes);
    majorPoresIdx = find(sizes >= avgSize);
    
    % 创建种子图像（只包含主要孔隙）
    seeds = false(size(model));
    for i = 1:length(majorPoresIdx)
        seeds(CC.PixelIdxList{majorPoresIdx(i)}) = true;
    end
    
    % 形态学重建
    fprintf('    重建主要孔隙结构...\n');
    reconstructed = imreconstruct(seeds, model, 26);
    
    % 混合原始和重建结果（保留一些细节）
    alpha = 0.8; % 重建权重
    model = alpha * reconstructed + (1-alpha) * double(model);
    model = model > 0.5;
    
    % 恢复一些中等大小的孔隙
    mediumPoresIdx = find(sizes >= optParams.targetMin & sizes < avgSize);
    nRestore = min(10, round(length(mediumPoresIdx) * 0.3));
    if nRestore > 0
        restoreIdx = mediumPoresIdx(randperm(length(mediumPoresIdx), nRestore));
        for i = 1:length(restoreIdx)
            model(CC.PixelIdxList{restoreIdx(i)}) = true;
        end
        fprintf('    恢复了%d个中等孔隙\n', nRestore);
    end
end
function model = referenceBasedMorphologyRestoration(model, referenceModel)
    % 基于参考模型的形态恢复
    fprintf('    基于参考进行形态恢复...\n');
    
    % 计算参考模型的形态特征
    refBoundary = bwperim(referenceModel, 26);
    refDistMap = bwdist(~referenceModel);
    
    % 计算当前模型的距离图
    currentDistMap = bwdist(~model);
    
    % 识别形态差异较大的区域
    diffMap = abs(refDistMap - currentDistMap);
    highDiffRegions = diffMap > quantile(diffMap(:), 0.8);
    
    % 在高差异区域进行局部调整
    [nx, ny, nz] = size(model);
    windowSize = 7;
    halfWindowSize = floor(windowSize/2);
    
    for z = 1:5:nz
        for y = 1:5:ny
            for x = 1:5:nx
                % 检查是否在高差异区域
                if x > windowSize && x <= nx-windowSize && ...
                    y > windowSize && y <= ny-windowSize && ...
                    z > halfWindowSize && z <= nz-halfWindowSize
                    
                    localDiff = highDiffRegions(x-3:x+3, y-3:y+3, z-1:z+1);
                    if sum(localDiff(:)) > numel(localDiff) * 0.3
                        % 提取局部窗口
                        localRef = referenceModel(x-windowSize:x+windowSize, ...
                            y-windowSize:y+windowSize, ...
                            z-halfWindowSize:z+halfWindowSize);
                        localCurrent = model(x-windowSize:x+windowSize, ...
                            y-windowSize:y+windowSize, ...
                            z-halfWindowSize:z+halfWindowSize);
                        
                        % 局部形态匹配
                        matched = matchLocalMorphology(localCurrent, localRef);
                        
                        % 更新模型
                        model(x-windowSize:x+windowSize, ...
                            y-windowSize:y+windowSize, ...
                            z-halfWindowSize:z+halfWindowSize) = matched;
                    end
                end
            end
        end
    end
end
function matched = matchLocalMorphology(current, reference)
    % 匹配局部形态
    % 使用形态学操作使当前区域更接近参考
    
    % 计算参考的形态特征
    refSolidity = sum(reference(:)) / numel(reference);
    currentSolidity = sum(current(:)) / numel(current);
    
    if abs(refSolidity - currentSolidity) > 0.1
        % 需要调整
        if refSolidity > currentSolidity
            % 参考更密实，需要填充
            se = strel('sphere', 1);
            matched = imclose(current, se);
        else
            % 参考更稀疏，需要侵蚀
            se = strel('sphere', 1);
            matched = imopen(current, se);
        end
    else
        % 差异不大，轻微平滑
        matched = medfilt3(double(current), [3, 3, 3]) > 0.5;
    end
end
function model = multiScaleBoundarySmoothing(model)
    % 多尺度边界平滑
    fprintf('    执行多尺度边界平滑...\n');
    
    % 尺度1：精细平滑
    se1 = strel('sphere', 1);
    model1 = imclose(imopen(model, se1), se1);
    
    % 尺度2：中等平滑
    fprintf('    中等尺度平滑...\n');
    smoothed2 = imgaussfilt3(double(model1), 1.0);
    model2 = smoothed2 > 0.5;
    
    % 尺度3：大尺度平滑（只应用于大孔隙）
    fprintf('    大尺度选择性平滑...\n');
    CC = bwconncomp(model2, 26);
    if CC.NumObjects > 0
        sizes = cellfun(@numel, CC.PixelIdxList);
        largePoresIdx = find(sizes > quantile(sizes, 0.75));
        
        largePoresMask = false(size(model));
        for i = 1:length(largePoresIdx)
            largePoresMask(CC.PixelIdxList{largePoresIdx(i)}) = true;
        end
        
        if any(largePoresMask(:))
            % 对大孔隙进行更强的平滑
            se3 = strel('sphere', 2);
            smoothedLarge = imclose(imopen(largePoresMask, se3), se3);
            
            % 更新大孔隙
            model2(largePoresMask) = false;
            model2 = model2 | smoothedLarge;
        end
    end
    
    model = model2;
    
    % 最终的细节恢复
    fprintf('    最终细节优化...\n');
    model = medfilt3(double(model), [3, 3, 3]) > 0.5;
end
function model = structureOptimization(model, optParams)
    % 结构优化 - 确保孔隙结构合理
    fprintf('    优化孔隙结构...\n');
    
    % 1. 合并过于接近的小孔隙
    model = mergeCloseSmallPores(model, optParams);
    
    % 2. 调整过大的孔隙
    model = reshapeOversizedPores(model, optParams);
    
    % 3. 修复细颈结构
    model = repairThinNecks(model);
    
    % 4. 确保连通性
    model = ensureConnectivity(model, optParams);
end
function model = mergeCloseSmallPores(model, optParams)
    % 合并接近的小孔隙
    CC = bwconncomp(model, 26);
    if CC.NumObjects <= 1
        return;
    end
    
    sizes = cellfun(@numel, CC.PixelIdxList);
    smallPoresIdx = find(sizes < optParams.targetMin);
    
    if length(smallPoresIdx) >= 2
        % 计算小孔隙之间的距离
        merged = false(length(smallPoresIdx));
        
        for i = 1:length(smallPoresIdx)-1
            if merged(i)
                continue;
            end
            
            for j = i+1:length(smallPoresIdx)
                if merged(j)
                    continue;
                end
                
                % 计算距离
                [dist, ~, ~] = findClosestPoints(model, ...
                    CC.PixelIdxList{smallPoresIdx(i)}, ...
                    CC.PixelIdxList{smallPoresIdx(j)});
                
                if dist < 5
                    % 合并这两个孔隙
                    model = connectClusters(model, ...
                        CC.PixelIdxList{smallPoresIdx(i)}, ...
                        CC.PixelIdxList{smallPoresIdx(j)});
                    merged([i, j]) = true;
                end
            end
        end
    end
end
function model = reshapeOversizedPores(model, optParams)
    % 调整过大的孔隙使其更符合目标大小而非直接分裂
    CC = bwconncomp(model, 26);
    if CC.NumObjects == 0
        return;
    end
    sizes = cellfun(@numel, CC.PixelIdxList);
    threshold = optParams.targetMax * 2;
    se = strel('sphere', 1);
    for i = 1:length(sizes)
        if sizes(i) <= threshold
            continue;
        end
        clusterMask = false(size(model));
        clusterMask(CC.PixelIdxList{i}) = true;
        distMap = bwdist(~clusterMask);
        shrinkRatio = min(0.4, max(0.05, (sizes(i) - optParams.targetMax) / sizes(i)));
        distances = distMap(clusterMask);
        cutoff = quantile(distances, shrinkRatio);
        coreRegion = clusterMask & (distMap > cutoff);
        coreRegion = imclose(coreRegion, se);
        coreRegion = imopen(coreRegion, se);
        boundary = clusterMask & ~coreRegion;
        reinforcement = imdilate(coreRegion, se) & clusterMask;
        adjusted = coreRegion | (boundary & reinforcement);
        if ~any(adjusted(:))
            adjusted = clusterMask;
        end
        model(CC.PixelIdxList{i}) = false;
        model = model | adjusted;
    end
end
function model = repairThinNecks(model)
    % 修复细颈结构
    % 使用形态学操作加粗细颈
    distMap = bwdist(~model);
    
    % 识别细颈（距离值小的孔隙区域）
    thinRegions = model & (distMap < 2);
    
    if any(thinRegions(:))
        % 局部膨胀
        se = strel('sphere', 1);
        dilated = imdilate(thinRegions, se);
        
        % 只在原始孔隙区域内膨胀
        model = model | (dilated & ~model & imdilate(model, se));
    end
end
function model = ensureConnectivity(model, optParams)
    % 确保主要孔隙的连通性
    CC = bwconncomp(model, 26);
    if CC.NumObjects <= 1
        return;
    end
    
    sizes = cellfun(@numel, CC.PixelIdxList);
    [sortedSizes, sortIdx] = sort(sizes, 'descend');
    
    % 保持最大的组分，考虑连接次大的组分
    if CC.NumObjects >= 2 && sortedSizes(2) > optParams.targetMin
        [dist, p1, p2] = findClosestPoints(model, ...
            CC.PixelIdxList{sortIdx(1)}, CC.PixelIdxList{sortIdx(2)});
        
        if dist < 8 && dist > 2
            % 创建连接
            model = createSmoothConnection(model, p1, p2);
        end
    end
end
function model = createSmoothConnection(model, p1, p2)
    % 创建平滑的连接路径
    nSteps = ceil(norm(p1 - p2) * 2);
    radius = 2;
    
    for t = 0:1/nSteps:1
        pos = round(p1 * (1-t) + p2 * t);
        
        % 在路径上创建球形区域
        for dx = -radius:radius
            for dy = -radius:radius
                for dz = -radius:radius
                    if sqrt(dx^2 + dy^2 + dz^2) <= radius
                        x = pos(1) + dx;
                        y = pos(2) + dy;
                        z = pos(3) + dz;
                        
                        if x >= 1 && x <= size(model,1) && ...
                            y >= 1 && y <= size(model,2) && ...
                            z >= 1 && z <= size(model,3)
                            model(x, y, z) = true;
                        end
                    end
                end
            end
        end
    end
end
function model = smartPorosityAdjustment(model, targetPorosity, optParams)
    % 智能孔隙率调整 - 保持形态特征
    currentPorosity = mean(model(:));
    diff = targetPorosity - currentPorosity;
    
    if abs(diff) < 0.001
        return;
    end
    
    % 获取当前孔隙结构
    CC = bwconncomp(model, 26);
    if CC.NumObjects == 0
        return;
    end
    
    sizes = cellfun(@numel, CC.PixelIdxList);
    
    if diff > 0
        % 需要增加孔隙 - 优先扩大现有孔隙
        fprintf('    增加孔隙率...\n');
        
        % 选择中等大小的孔隙进行扩展
        mediumPoresIdx = find(sizes > optParams.targetMin & sizes < optParams.targetMax);
        if isempty(mediumPoresIdx)
            mediumPoresIdx = 1:CC.NumObjects;
        end
        
        nVoxelsToAdd = round(abs(diff) * numel(model));
        voxelsAdded = 0;
        
        for i = 1:length(mediumPoresIdx)
            if voxelsAdded >= nVoxelsToAdd
                break;
            end
            
            % 获取孔隙边界
            clusterMask = false(size(model));
            clusterMask(CC.PixelIdxList{mediumPoresIdx(i)}) = true;
            boundary = imdilate(clusterMask, ones(3,3,3)) & ~clusterMask & ~model;
            boundaryIdx = find(boundary);
            
            if ~isempty(boundaryIdx)
                nAdd = min(round(nVoxelsToAdd/length(mediumPoresIdx)), length(boundaryIdx));
                addIdx = boundaryIdx(randperm(length(boundaryIdx), nAdd));
                model(addIdx) = true;
                voxelsAdded = voxelsAdded + nAdd;
            end
        end
        
    else
        % 需要减少孔隙 - 优先从大孔隙边界收缩
        fprintf('    减少孔隙率...\n');
        
        largePoresIdx = find(sizes > optParams.targetMax);
        if isempty(largePoresIdx)
            largePoresIdx = find(sizes > mean(sizes));
        end
        
        nVoxelsToRemove = round(abs(diff) * numel(model));
        voxelsRemoved = 0;
        
        for i = 1:length(largePoresIdx)
            if voxelsRemoved >= nVoxelsToRemove
                break;
            end
            
            % 获取孔隙边界
            clusterMask = false(size(model));
            clusterMask(CC.PixelIdxList{largePoresIdx(i)}) = true;
            innerBoundary = clusterMask & ~imerode(clusterMask, ones(3,3,3));
            boundaryIdx = find(innerBoundary);
            
            if ~isempty(boundaryIdx)
                nRemove = min(round(nVoxelsToRemove/length(largePoresIdx)), length(boundaryIdx));
                removeIdx = boundaryIdx(randperm(length(boundaryIdx), nRemove));
                model(removeIdx) = false;
                voxelsRemoved = voxelsRemoved + nRemove;
            end
        end
    end
    
    % 最终微调
    finalPorosity = mean(model(:));
    if abs(finalPorosity - targetPorosity) > 0.001
        % 使用随机调整进行最终微调
        model = adjustPorosity(model, targetPorosity);
    end
end
function model = finalQualityRefinement(model, optParams)
    % 最终质量细化
    fprintf('    执行最终质量细化...\n');
    
    % 1. 最后一次去除极小特征
    CC = bwconncomp(model, 26);
    if CC.NumObjects > 0
        sizes = cellfun(@numel, CC.PixelIdxList);
        tinyIdx = find(sizes < 5);
        for i = 1:length(tinyIdx)
            model(CC.PixelIdxList{tinyIdx(i)}) = false;
        end
    end
    
    % 2. 最终边界平滑
    se = strel('sphere', 1);
    model = imclose(imopen(model, se), se);
    
    % 3. 确保没有孤立的单体素
    model = bwareaopen(model, 3, 26);
    
    % 4. 最后的中值滤波
    model = medfilt3(double(model), [3, 3, 3]) > 0.5;
end
%% ========== 激进后处理函数（用于平滑版本） ==========
function model = aggressiveNoiseRemoval(model, optParams)
    % 激进的噪声去除
    threshold = max(20, optParams.targetMin / 3);
    
    % 移除所有小孔隙
    CC = bwconncomp(model, 26);
    if CC.NumObjects > 0
        sizes = cellfun(@numel, CC.PixelIdxList);
        removeIdx = find(sizes < threshold);
        fprintf('    移除%d个小孔隙（<%d体素）\n', length(removeIdx), threshold);
        for i = 1:length(removeIdx)
            model(CC.PixelIdxList{removeIdx(i)}) = false;
        end
    end
    
    % 填充所有小孔洞
    invertedModel = ~model;
    CC_inv = bwconncomp(invertedModel, 26);
    if CC_inv.NumObjects > 0
        sizes_inv = cellfun(@numel, CC_inv.PixelIdxList);
        fillIdx = find(sizes_inv < threshold);
        for i = 1:length(fillIdx)
            model(CC_inv.PixelIdxList{fillIdx(i)}) = true;
        end
    end
    
    % 强力中值滤波
    model = medfilt3(double(model), [5, 5, 5]) > 0.5;
end
function model = strongMorphologicalOperations(model)
    % 强化的形态学操作
    se1 = strel('sphere', 2);
    model = imopen(model, se1);
    model = imclose(model, se1);
    
    se2 = strel('sphere', 3);
    model = imclose(imopen(model, se2), se2);
end
function model = deepBoundarySmoothing(model)
    % 深度边界平滑
    % 多次高斯平滑
    for i = 1:3
        smoothed = imgaussfilt3(double(model), 1.5);
        model = smoothed > 0.5;
    end
    
    % 最终形态学平滑
    se = strel('sphere', 2);
    model = imclose(imopen(model, se), se);
end
function model = structureSimplification(model, optParams)
    % 结构简化
    CC = bwconncomp(model, 26);
    if CC.NumObjects == 0
        return;
    end
    
    % 只保留较大的孔隙
    sizes = cellfun(@numel, CC.PixelIdxList);
    keepIdx = find(sizes >= optParams.targetMin);
    
    newModel = false(size(model));
    for i = 1:length(keepIdx)
        newModel(CC.PixelIdxList{keepIdx(i)}) = true;
    end
    
    model = newModel;
end
%% ========== 辅助函数 ==========
function reportFinalStatistics(model, optParams)
    % 报告最终统计信息
    CC = bwconncomp(model, 26);
    if CC.NumObjects == 0
        fprintf('\n警告：没有检测到孔隙！\n');
        return;
    end
    
    sizes = cellfun(@numel, CC.PixelIdxList);
    
    fprintf('\n最终孔隙分布统计：\n');
    fprintf('  总孔隙数: %d\n', CC.NumObjects);
    if isfield(optParams, 'targetClusterCount') && ~isempty(optParams.targetClusterCount)
        fprintf('  目标孔隙数: %d\n', round(optParams.targetClusterCount));
    end
    fprintf('  孔隙率: %.4f\n', mean(model(:)));
    fprintf('  最小孔隙: %d 体素\n', min(sizes));
    fprintf('  最大孔隙: %d 体素\n', max(sizes));
    fprintf('  平均孔隙: %.1f 体素\n', mean(sizes));
    fprintf('  标准差: %.1f 体素\n', std(sizes));
    
    % 分布统计
    smallCount = sum(sizes < optParams.targetMin);
    mediumCount = sum(sizes >= optParams.targetMin & sizes <= optParams.targetMax);
    largeCount = sum(sizes > optParams.targetMax);
    
    fprintf('\n孔隙大小分布：\n');
    fprintf('  小孔隙 (<%d): %d (%.1f%%)\n', optParams.targetMin, smallCount, 100*smallCount/CC.NumObjects);
    fprintf('  中孔隙 (%d-%d): %d (%.1f%%)\n', optParams.targetMin, optParams.targetMax, mediumCount, 100*mediumCount/CC.NumObjects);
    fprintf('  大孔隙 (>%d): %d (%.1f%%)\n', optParams.targetMax, largeCount, 100*largeCount/CC.NumObjects);
    
    % 形态质量评估
    boundary = bwperim(model, 26);
    smoothness = 1 - sum(boundary(:)) / sum(model(:));
    fprintf('\n形态质量：\n');
    fprintf('  边界平滑度指标: %.3f\n', smoothness);
end
%% ========== 质量检查和结果显示 ==========
function checkComprehensiveModelQuality(finalModel, originalModel, optParams)
    % 综合质量检查
    fprintf('\n===== 模型质量检查 =====\n');
    
    % 1. 形态一致性检查
    fprintf('\n1. 形态一致性：\n');
    origMorph = computeDetailedMorphologyFeatures(originalModel);
    finalMorph = computeDetailedMorphologyFeatures(finalModel);
    
    if ~isempty(origMorph.sphericity) && ~isempty(finalMorph.sphericity)
        spherDiff = abs(mean(origMorph.sphericity) - mean(finalMorph.sphericity));
        elongDiff = abs(mean(origMorph.elongation) - mean(finalMorph.elongation));
        fprintf('  球形度差异: %.3f\n', spherDiff);
        fprintf('  伸长率差异: %.3f\n', elongDiff);
        
        if spherDiff > 0.2 || elongDiff > 0.5
            fprintf('  警告：形态特征差异较大\n');
        else
            fprintf('  形态特征保持良好\n');
        end
    end
    
    % 2. 空间特征检查
    fprintf('\n2. 空间特征：\n');
    origSpatial = computeEnhancedSpatialFeatures(originalModel);
    finalSpatial = computeEnhancedSpatialFeatures(finalModel);

    anisDiff = abs(origSpatial.anisotropy - finalSpatial.anisotropy);
    fprintf('  各向异性差异: %.4f\n', anisDiff);

    if anisDiff > 0.1
        fprintf('  警告：各向异性变化较大\n');
    else
        fprintf('  各向异性保持良好\n');
    end

    if isfield(origSpatial, 'chordLengthDistribution') && ...
            isfield(finalSpatial, 'chordLengthDistribution')
        cldDiff = normalizeDifference(origSpatial.chordLengthDistribution.meanLength, ...
            finalSpatial.chordLengthDistribution.meanLength);
        fprintf('  弦长分布均值差异: %.4f\n', cldDiff);
    end

    if isfield(origSpatial, 'poreSizeDistribution') && ...
            isfield(finalSpatial, 'poreSizeDistribution')
        psdDiff = normalizeDifference(origSpatial.poreSizeDistribution.meanRadius, ...
            finalSpatial.poreSizeDistribution.meanRadius);
        fprintf('  孔隙大小均值差异: %.4f\n', psdDiff);
    end

    if isfield(origSpatial, 'minkowskiFunctionals') && ...
            isfield(finalSpatial, 'minkowskiFunctionals')
        imcDiff = normalizeDifference(origSpatial.minkowskiFunctionals.integralMeanCurvature, ...
            finalSpatial.minkowskiFunctionals.integralMeanCurvature);
        fprintf('  积分平均曲率差异: %.4f\n', imcDiff);
    end

    if isfield(origSpatial, 'linealPathFunction') && ...
            isfield(finalSpatial, 'linealPathFunction')
        lpfDiff = computeLinealPathDifference(origSpatial.linealPathFunction, ...
            finalSpatial.linealPathFunction);
        fprintf('  线性路径函数平均差异: %.4f\n', lpfDiff);
    end

    % 3. 结构完整性检查
    fprintf('\n3. 结构完整性：\n');
    checkStructuralIntegrity(finalModel);
    
    % 4. 簇分布检查
    fprintf('\n4. 簇分布：\n');
    CC = bwconncomp(finalModel, 26);
    if CC.NumObjects > 0
        sizes = cellfun(@numel, CC.PixelIdxList);
        fprintf('  簇数量: %d\n', CC.NumObjects);
        if isfield(optParams, 'targetClusterCount') && ~isempty(optParams.targetClusterCount)
            fprintf('  目标簇数量: %d\n', round(optParams.targetClusterCount));
            if CC.NumObjects == round(optParams.targetClusterCount)
                fprintf('  ✅ 簇数量与目标一致\n');
            else
                fprintf('  ⚠️ 当前簇数量与目标存在偏差\n');
            end
        end
        fprintf('  簇大小范围: [%d, %d]\n', min(sizes), max(sizes));
        fprintf('  目标范围: [%d, %d]\n', optParams.targetMin, optParams.targetMax);
        
        outOfRange = sum(sizes < optParams.targetMin) + sum(sizes > optParams.targetMax);
        if outOfRange > CC.NumObjects * 0.2
            fprintf('  警告：%d个簇超出目标范围\n', outOfRange);
        else
            fprintf('  簇大小分布良好\n');
        end
    end
end
function checkStructuralIntegrity(model)
    % 检查结构完整性
    % 检查是否有断裂或不连续
    CC = bwconncomp(model, 26);
    
    if CC.NumObjects == 1
        fprintf('  模型为单一连通组分\n');
    elseif CC.NumObjects < 5
        fprintf('  模型有%d个连通组分（可接受）\n', CC.NumObjects);
    else
        fprintf('  警告：模型有%d个连通组分（可能过于碎片化）\n', CC.NumObjects);
    end
    
    % 检查方块化程度
    blockiness = checkBlockiness(model);
    if blockiness > 0.8
        fprintf('  警告：模型呈现方块状特征\n');
    else
        fprintf('  模型形状自然\n');
    end
end
function blockiness = checkBlockiness(model)
    % 检查方块化程度（返回0-1的值）
    [gx, gy, gz] = gradient(double(model));
    edges = sqrt(gx.^2 + gy.^2 + gz.^2);
    edgePoints = find(edges > 0.5);
    
    if length(edgePoints) < 100
        blockiness = 0;
        return;
    end
    
    % 采样检查梯度方向
    sampleIdx = edgePoints(randperm(length(edgePoints), min(100, length(edgePoints))));
    alignmentScore = 0;
    
    for i = 1:length(sampleIdx)
        [x, y, z] = ind2sub(size(model), sampleIdx(i));
        gradVec = [gx(x,y,z), gy(x,y,z), gz(x,y,z)];
        gradVec = gradVec / (norm(gradVec) + eps);
        
        % 与坐标轴的对齐程度
        axisAlignment = max(abs(gradVec));
        alignmentScore = alignmentScore + axisAlignment;
    end
    
    blockiness = alignmentScore / length(sampleIdx);
end
function displayComprehensiveFinalResults(originalModel, finalModel, ...
    originalFeatures, originalSpatial, originalMorph, performanceMonitor, optParams)
    % 显示综合最终结果
    fprintf('\n\n===== 综合优化结果总结 =====\n');
    
    % 计算最终特征
    finalFeatures = extractEfficientClusterFeatures(finalModel);
    finalSpatial = computeEnhancedSpatialFeatures(finalModel);
    finalMorph = computeDetailedMorphologyFeatures(finalModel);
    
    % 1. 基本统计
    fprintf('\n1. 基本统计：\n');
    fprintf('  原始孔隙率: %.4f | 最终孔隙率: %.4f\n', ...
        mean(originalModel(:)), mean(finalModel(:)));
    fprintf('  原始簇数: %d | 最终簇数: %d\n', ...
        originalFeatures.numClusters, finalFeatures.numClusters);
    if isfield(optParams, 'targetClusterCount') && ~isempty(optParams.targetClusterCount)
        fprintf('  目标簇数: %d\n', round(optParams.targetClusterCount));
        if finalFeatures.numClusters == round(optParams.targetClusterCount)
            fprintf('  ✅ 簇数量满足目标要求\n');
        else
            fprintf('  ⚠️ 簇数量与目标存在偏差\n');
        end
    end
    
    % 2. 形态特征对比
    fprintf('\n2. 形态特征：\n');
    if ~isempty(originalMorph.sphericity) && ~isempty(finalMorph.sphericity)
        fprintf('  球形度: 原始=%.3f | 最终=%.3f\n', ...
            mean(originalMorph.sphericity), mean(finalMorph.sphericity));
        fprintf('  伸长率: 原始=%.3f | 最终=%.3f\n', ...
            mean(originalMorph.elongation), mean(finalMorph.elongation));
    end
    
    % 3. 空间特征对比
    fprintf('\n3. 空间特征：\n');
    fprintf('  各向异性: 原始=%.4f | 最终=%.4f\n', ...
        originalSpatial.anisotropy, finalSpatial.anisotropy);
    fprintf('  连通性: 原始=%.4f | 最终=%.4f\n', ...
        originalSpatial.connectivity.largestComponentRatio, ...
        finalSpatial.connectivity.largestComponentRatio);
    if isfield(originalSpatial, 'chordLengthDistribution') && ...
            isfield(finalSpatial, 'chordLengthDistribution')
        fprintf('  弦长均值: 原始=%.3f | 最终=%.3f\n', ...
            originalSpatial.chordLengthDistribution.meanLength, ...
            finalSpatial.chordLengthDistribution.meanLength);
    end
    if isfield(originalSpatial, 'poreSizeDistribution') && ...
            isfield(finalSpatial, 'poreSizeDistribution')
        fprintf('  孔隙半径均值: 原始=%.3f | 最终=%.3f\n', ...
            originalSpatial.poreSizeDistribution.meanRadius, ...
            finalSpatial.poreSizeDistribution.meanRadius);
    end
    if isfield(originalSpatial, 'minkowskiFunctionals') && ...
            isfield(finalSpatial, 'minkowskiFunctionals')
        fprintf('  积分平均曲率: 原始=%.3f | 最终=%.3f\n', ...
            originalSpatial.minkowskiFunctionals.integralMeanCurvature, ...
            finalSpatial.minkowskiFunctionals.integralMeanCurvature);
    end
    if isfield(originalSpatial, 'linealPathFunction') && ...
            isfield(finalSpatial, 'linealPathFunction')
        lpfDiff = computeLinealPathDifference(originalSpatial.linealPathFunction, ...
            finalSpatial.linealPathFunction);
        fprintf('  线性路径函数平均差异: %.4f\n', lpfDiff);
    end

    % 4. 优化性能
    fprintf('\n4. 优化性能：\n');
    fprintf('  初始能量: %.6f\n', performanceMonitor.energyHistory(1));
    fprintf('  最终能量: %.6f\n', performanceMonitor.energyHistory(end));
    fprintf('  能量降低: %.2f%%\n', ...
        (performanceMonitor.energyHistory(1) - performanceMonitor.energyHistory(end)) / ...
        abs(performanceMonitor.energyHistory(1)) * 100);
    fprintf('  平均接受率: %.2f%%\n', mean(performanceMonitor.acceptanceRatio) * 100);
    
    % 5. 综合评分
    morphMatch = calculateMorphologyMatch(finalMorph, originalMorph);
    spatialMatch = calculateSpatialMatch(finalSpatial, originalSpatial);
    overallScore = 0.5 * morphMatch + 0.5 * spatialMatch;
    
    fprintf('\n5. 综合评分：\n');
    fprintf('  形态匹配度: %.3f\n', morphMatch);
    fprintf('  空间匹配度: %.3f\n', spatialMatch);
    fprintf('  综合得分: %.3f (1.0为完美匹配)\n', overallScore);
    
    % 可视化最终对比
    visualizeFinalComparison(originalModel, finalModel, performanceMonitor);
end
function visualizeFinalComparison(originalModel, finalModel, performanceMonitor)
    % 最终对比可视化
    try
        figure(400); clf;
        
        % 模型对比
        subplot(2,3,1);
        imshow(squeeze(originalModel(:,:,round(size(originalModel,3)/2))), []);
        title('原始模型');
        
        subplot(2,3,2);
        imshow(squeeze(finalModel(:,:,round(size(finalModel,3)/2))), []);
        title('最终模型');
        
        subplot(2,3,3);
        diff = double(originalModel) - double(finalModel);
        imagesc(squeeze(diff(:,:,round(size(diff,3)/2))));
        colormap(subplot(2,3,3), 'jet');
        colorbar;
        title('差异图');
        
        % 能量历史
        subplot(2,3,4);
        plot(performanceMonitor.energyHistory);
        xlabel('迭代');
        ylabel('能量');
        title('能量演化');
        grid on;
        
        % 匹配度历史
        subplot(2,3,5);
        plot(performanceMonitor.morphologyMatchHistory, 'b-', 'LineWidth', 2);
        hold on;
        plot(performanceMonitor.spatialMatchHistory, 'r-', 'LineWidth', 2);
        xlabel('迭代 (×100)');
        ylabel('匹配度');
        legend('形态', '空间');
        title('匹配度演化');
        grid on;
        
        % 簇大小演化
        subplot(2,3,6);
        plot(performanceMonitor.clusterSizeHistory(1,:), 'g-', 'LineWidth', 2);
        hold on;
        plot(performanceMonitor.clusterSizeHistory(2,:), 'r-', 'LineWidth', 2);
        plot(performanceMonitor.clusterSizeHistory(3,:), 'b-', 'LineWidth', 2);
        xlabel('迭代 (×100)');
        ylabel('簇大小');
        legend('最小', '最大', '平均');
        title('簇大小演化');
        grid on;
        
        drawnow;
        
    catch
        % 忽略可视化错误
        fprintf('可视化时出现错误，跳过...\n');
    end
end
%% ========== 辅助计算函数 ==========
function match = calculateSpatialMatch(currentFeatures, targetFeatures)
    % 计算空间特征匹配度
    match = 0;
    nTerms = 0;
    
    % 各向异性匹配
    if isfield(currentFeatures, 'anisotropy') && isfield(targetFeatures, 'anisotropy')
        diff = abs(currentFeatures.anisotropy - targetFeatures.anisotropy) / ...
            (targetFeatures.anisotropy + 0.1);
        match = match + (1 - min(diff, 1));
        nTerms = nTerms + 1;
    end
    
    % 两点相关函数匹配
    if isfield(currentFeatures, 'twoPointCorr') && isfield(targetFeatures, 'twoPointCorr')
        % 确保尺寸匹配
        minSize = min(size(currentFeatures.twoPointCorr, 1), ...
            size(targetFeatures.twoPointCorr, 1));
        if minSize > 0
            currentTPC = currentFeatures.twoPointCorr(1:minSize, :);
            targetTPC = targetFeatures.twoPointCorr(1:minSize, :);
            diff = mean(abs(currentTPC(:) - targetTPC(:)));
            match = match + (1 - min(diff, 1));
            nTerms = nTerms + 1;
        end
    end
    
    % 连通性匹配
    if isfield(currentFeatures, 'connectivity') && isfield(targetFeatures, 'connectivity')
        if isfield(currentFeatures.connectivity, 'largestComponentRatio') && ...
            isfield(targetFeatures.connectivity, 'largestComponentRatio')
            diff = abs(currentFeatures.connectivity.largestComponentRatio - ...
                targetFeatures.connectivity.largestComponentRatio);
            match = match + (1 - min(diff, 1));
            nTerms = nTerms + 1;
        end
    end

    % 弦长分布匹配
    if isfield(currentFeatures, 'chordLengthDistribution') && ...
            isfield(targetFeatures, 'chordLengthDistribution')
        currentCLD = currentFeatures.chordLengthDistribution;
        targetCLD = targetFeatures.chordLengthDistribution;
        distDiff = 0.5 * computeDistributionDifference(currentCLD.binCenters, currentCLD.probability, ...
            targetCLD.binCenters, targetCLD.probability) + ...
            0.5 * normalizeDifference(currentCLD.meanLength, targetCLD.meanLength);
        match = match + (1 - min(distDiff, 1));
        nTerms = nTerms + 1;
    end

    % 孔隙大小分布匹配
    if isfield(currentFeatures, 'poreSizeDistribution') && ...
            isfield(targetFeatures, 'poreSizeDistribution')
        currentPSD = currentFeatures.poreSizeDistribution;
        targetPSD = targetFeatures.poreSizeDistribution;
        distDiff = 0.5 * computeDistributionDifference(currentPSD.binCenters, currentPSD.probability, ...
            targetPSD.binCenters, targetPSD.probability) + ...
            0.5 * normalizeDifference(currentPSD.meanRadius, targetPSD.meanRadius);
        match = match + (1 - min(distDiff, 1));
        nTerms = nTerms + 1;
    end

    % Minkowski 积分平均曲率匹配
    if isfield(currentFeatures, 'minkowskiFunctionals') && ...
            isfield(targetFeatures, 'minkowskiFunctionals')
        if isfield(currentFeatures.minkowskiFunctionals, 'integralMeanCurvature') && ...
                isfield(targetFeatures.minkowskiFunctionals, 'integralMeanCurvature')
            diff = normalizeDifference(currentFeatures.minkowskiFunctionals.integralMeanCurvature, ...
                targetFeatures.minkowskiFunctionals.integralMeanCurvature);
            match = match + (1 - min(diff, 1));
            nTerms = nTerms + 1;
        end
    end

    % 线性路径函数匹配
    if isfield(currentFeatures, 'linealPathFunction') && ...
            isfield(targetFeatures, 'linealPathFunction')
        diff = computeLinealPathDifference(currentFeatures.linealPathFunction, ...
            targetFeatures.linealPathFunction);
        match = match + (1 - min(diff, 1));
        nTerms = nTerms + 1;
    end

    % 归一化
    if nTerms > 0
        match = match / nTerms;
    else
        match = 0.5; % 默认值
    end
    
    % 确保在[0,1]范围
    match = max(0, min(1, match));
end
function match = calculateMorphologyMatch(currentMorph, targetMorph)
    % 计算形态学特征匹配度
    match = 0;
    nTerms = 0;
    
    % 球形度匹配
    if ~isempty(currentMorph.sphericity) && ~isempty(targetMorph.sphericity)
        diff = abs(mean(currentMorph.sphericity) - mean(targetMorph.sphericity));
        match = match + (1 - min(diff, 1));
        nTerms = nTerms + 1;
    end
    
    % 伸长率匹配
    if ~isempty(currentMorph.elongation) && ~isempty(targetMorph.elongation)
        diff = abs(mean(currentMorph.elongation) - mean(targetMorph.elongation)) / ...
            (mean(targetMorph.elongation) + 0.1);
        match = match + (1 - min(diff, 1));
        nTerms = nTerms + 1;
    end
    
    % 网络密度匹配
    if isfield(currentMorph, 'poreNetworkDensity') && ...
            isfield(targetMorph, 'poreNetworkDensity')
        diff = abs(currentMorph.poreNetworkDensity - targetMorph.poreNetworkDensity) / ...
            (targetMorph.poreNetworkDensity + 0.01);
        match = match + (1 - min(diff, 1));
        nTerms = nTerms + 1;
    end
    
    % 归一化
    if nTerms > 0
        match = match / nTerms;
    else
        match = 0.5; % 默认值
    end
    
    % 确保在[0,1]范围
    match = max(0, min(1, match));
end
%% 主函数结束