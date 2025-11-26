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
% 可调配置（集中在一个结构体方便快速实验）
config = struct( ...
    'forceParallelPool', true, ...                 % 强制启动并行池以提升 CPU 利用率
    'desiredPoolSize', 0, ...                      % 0 表示让 MATLAB 自适应，否则可指定并行线程数
    'defaultBatchSize', 32, ...                    % 批量候选移动默认数量（从 12 提升，填满多核）
    'maxBatchSize', 64, ...                        % 并行时的批量上限，兼顾内存
    'constraintInterval', 180, ...                 % 约束调用用于纠偏而非主导优化，降低频率以避免过度重写拓扑
    'checkpointEnabled', true, ...                 % 是否启用迭代回退
    'checkpointTolerance', 0.02, ...               % 回退触发的匹配度容忍度（放宽至 0.02）
    'morphologyPhaseRatio', 0.55, ...              % 形态阶段比例（延长形态阶段）
    'spatialPhaseRatio', 0.85, ...                 % 空间阶段结束比例
    'diagnosticMatchRaw', false, ...               % 默认关闭诊断模式，完全尊重用户设定孔隙率
    'directOverlapWeight', 8.0, ...                % 直接重叠能量权重
    'shapeFirstMode', false);                      % 形态优先模式（孔隙率接近时自动开启）
validateConfig(config);
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
% 如有需要，预先启动并行池，避免迭代过程中频繁创建导致 CPU 闲置
ensureParallelPool(config);
fileName = 'DATA1.raw'; % 输入文件
% 读取原始模型（使用原始尺寸）
rawModel = readRawModel(fileName, originalDims);
% 如果需要，调整原始模型大小
if ~isequal(dims, originalDims)
    fprintf('\n正在调整原始模型大小...\n');
    rawModel = resizeModel(rawModel, originalDims, dims);
end
pore_threshold = input('请输入用户设定的孔隙阈值 pore_threshold: ');
[referenceBinaryModel, referencePorosity] = binarizeWithUserThreshold(rawModel, pore_threshold);
raw_porosity = referencePorosity;
fprintf('原始灰度模型孔隙率：%.4f\n', raw_porosity);
fprintf('参考二值模型孔隙率：%.4f\n', referencePorosity);
saveRawModel(referenceBinaryModel, 'originalDataModel.raw');
% 提取参考模型特征（只计算一次）
fprintf('正在提取参考模型的综合特征...\n');
tic;
originalFeatures = extractEfficientClusterFeatures(referenceBinaryModel);
spatialFeatures = computeEnhancedSpatialFeatures(referenceBinaryModel);
morphologyFeatures = computeDetailedMorphologyFeatures(referenceBinaryModel);
multiScaleSpatialFeatures = computeMultiScaleSpatialFeatures(referenceBinaryModel);
fprintf('特征提取完成，用时：%.2f秒\n', toc);
% 显示特征
displayClusterFeatures(originalFeatures);
displaySpatialFeatures(spatialFeatures);
displayMorphologyFeatures(morphologyFeatures);
displayMultiScaleSpatialFeatures(multiScaleSpatialFeatures);
%% 2. 构建孔隙率-空间特征梯度模型并收集用户目标
fprintf('正在分析孔隙率与空间特征之间的关系...\n');
spatialGradientModel = computeSpatialParameterGradients(referenceBinaryModel, spatialFeatures, referencePorosity);
target_porosity = input('请输入目标孔隙率（0-1之间）: ');
if isempty(target_porosity) || ~isfinite(target_porosity)
    warning('未提供有效的目标孔隙率，默认使用原始模型的孔隙率。');
    target_porosity = raw_porosity;
end
target_porosity = max(0, min(1, target_porosity));
% 基于孔隙率梯度模型推断目标空间特征
[new_target_spatial_features, inferredSpatialSummary] = predictSpatialFeaturesFromGradients( ...
    spatialGradientModel, target_porosity, spatialFeatures, prod(dims));
% 当孔隙率变化幅度很小，直接锁定原始空间与形态特征，避免梯度外推造成不合理偏移
porDiff = abs(target_porosity - raw_porosity);
gradientLockThreshold = 0.02;
if porDiff < gradientLockThreshold
    new_target_spatial_features = spatialFeatures;
    [~, ~, inferredSpatialSummary] = summarizeSpatialFeaturesForRegression(spatialFeatures, prod(dims));
    optParams.morphologyFeatures = morphologyFeatures;
end
% 自动启用形态优先模式：当目标孔隙率与参考孔隙率差异极小，优先保持原始形态
config.shapeFirstMode = config.shapeFirstMode || abs(target_porosity - referencePorosity) < 0.02;
% 诊断模式：仅对齐空间/形态特征，孔隙率保持用户设定
if config.diagnosticMatchRaw
    fprintf('\n已启用诊断模式：空间/形态特征对齐原始模型，孔隙率仍使用用户设定值 %.4f。\n', target_porosity);
    new_target_spatial_features = spatialFeatures;
    morphologyFeatures = computeDetailedMorphologyFeatures(referenceBinaryModel); % 强化对原始形态的匹配
end
fprintf('\n小孔隙保留选项：\n');
fprintf('1. 保留小孔隙（推荐，使结果更接近原始模型）\n');
fprintf('2. 去除小孔隙（获得更平滑的结果）\n');
preserve_small_pores = input('请选择 (1 或 2): ') == 1;
% 预计算优化参数
optParams = struct();
optParams.targetPorosity = target_porosity;
optParams.originalFeatures = originalFeatures;
optParams.spatialFeatures = new_target_spatial_features;
optParams.spatialGradientModel = spatialGradientModel;
optParams.inferredSpatialSummary = inferredSpatialSummary;
baseMorphology = morphologyFeatures;
optParams.morphologyFeatures = buildMorphologyTargetForPorosity(baseMorphology, referencePorosity, target_porosity);
optParams.multiScaleSpatialFeatures = multiScaleSpatialFeatures;
optParams.modelSize = dims;
optParams.originalBinaryModel = referenceBinaryModel; % 统一参考二值模型
optParams.referenceModel = referenceBinaryModel; % 用于 directOverlap 能量项
optParams.preserveSmallPores = preserve_small_pores; % 添加小孔隙保留标志
optParams.shapeFirstMode = config.shapeFirstMode;
optParams.directionalPorosityProfile = computeDirectionalPorosityProfile(referenceBinaryModel);
optParams.referenceDensityMap = constructReferenceDensityMap(referenceBinaryModel, multiScaleSpatialFeatures);
optParams.clusterLibrary = sampleRepresentativeClusters(referenceBinaryModel, morphologyFeatures);
optParams.clusterReference = buildClusterReference(originalFeatures, prod(dims));
optParams.clusterTarget = buildAdaptiveClusterTargets(originalFeatures, referencePorosity, target_porosity, dims);
optParams.energyWeights = getDefaultEnergyWeights();
optParams.energyWeights.directOverlap = config.directOverlapWeight; % 新增直接重叠能量权重
if optParams.clusterReference.meanSize > 0
    optParams.targetMin = max(1, round(optParams.clusterReference.meanSize * 0.5));
    optParams.targetMax = max(optParams.targetMin + 1, round(optParams.clusterReference.meanSize + 2 * optParams.clusterReference.stdSize));
    optParams.targetClusterCount = max(1, round(optParams.clusterReference.numClusters));
else
    optParams.targetMin = [];
    optParams.targetMax = [];
    optParams.targetClusterCount = [];
end
validateOptParams(optParams);
% 生成快速查找表
fprintf('正在生成快速查找表...\n');
lookupTables = generateComprehensiveLookupTables(referenceBinaryModel, originalFeatures, optParams.spatialFeatures, morphologyFeatures);
%% 3. 生成初始模型 - 综合版
fprintf('正在生成综合优化的初始模型...\n');
mcmcModel = [];
if config.shapeFirstMode && isequal(size(referenceBinaryModel), dims)
    fprintf('启用形态优先模式：从参考模型出发，并仅在边界进行轻微扰动。\n');
    mcmcModel = applyBoundaryPerturbation(referenceBinaryModel, 0.002, 2);
end
if isempty(mcmcModel)
    mcmcModel = generateComprehensiveInitialModel(dims, target_porosity, originalFeatures, ...
        optParams.spatialFeatures, morphologyFeatures, optParams);
end
initial_porosity = computeGlobalPorosity(mcmcModel);
%% 4. MCMC 迭代参数设置
maxIterations = 1500; % 增加迭代次数
batchSize = max(config.defaultBatchSize, 16);
parallelEnabled = shouldUseParallel(batchSize, numel(mcmcModel)) || config.forceParallelPool;
if parallelEnabled
    batchSize = min(config.maxBatchSize, max(batchSize, config.defaultBatchSize));
    fprintf('检测到并行资源，批量移动数调整为 %d 并启用并行评估。\n', batchSize);
end
featureParallelEnabled = shouldUseParallel(max(8, batchSize), numel(mcmcModel)) || parallelEnabled;
if parallelEnabled && ~featureParallelEnabled
    % 如果批量评估已启用并行，但特征更新尚未启用，强制共享池
    featureParallelEnabled = true;
end
% 基于初始能量归一化权重，强化孔隙率约束
initialCluster = extractEfficientClusterFeatures(mcmcModel);
initialSpatial = computeEnhancedSpatialFeatures(mcmcModel);
initialMorph = computeDetailedMorphologyFeatures(mcmcModel);
initialMulti = computeMultiScaleSpatialFeatures(mcmcModel);
initialComponents = evaluateEnergyComponents(mcmcModel, initialCluster, ...
    initialSpatial, initialMorph, initialMulti, optParams);
if config.shapeFirstMode || abs(target_porosity - referencePorosity) < 0.03
    targetRatios = struct( ...
        'porosity',          0.24, ...
        'morphology',        0.16, ...
        'spatial',           0.14, ...
        'cluster',           0.16, ...
        'connectivity',      0.14, ...
        'multiScale',        0.04, ...
        'shapePreservation', 0.04, ...
        'structureCoherence',0.03, ...
        'island',            0.05, ...
        'directOverlap',     0.24); % 直接重叠占据主导，强调形态一致
else
    targetRatios = struct( ...
        'porosity',          0.26, ...
        'morphology',        0.17, ...
        'spatial',           0.16, ...
        'cluster',           0.15, ...
        'connectivity',      0.14, ...
        'multiScale',        0.05, ...
        'shapePreservation', 0.03, ...
        'structureCoherence',0.02, ...
        'island',            0.02, ...
        'directOverlap',     0.20);
end
weights = initializeNormalizedEnergyWeights(initialComponents, targetRatios);
optParams.energyWeights = weights;
% 初始化MCMC状态
mcmcState = initializeComprehensiveMCMCState(mcmcModel, optParams, weights, featureParallelEnabled);
initialEnergy = mcmcState.currentEnergy;
% 自适应温度控制
T0 = 0.4;
cooling_rate = 0.995;
morphology_annealing_rate = 0.995;
spatial_annealing_rate = 0.995;
% 性能监控
performanceMonitor = initializeComprehensivePerformanceMonitor(maxIterations);
%% 5. 综合MCMC主循环
fprintf('开始综合形态-空间特征MCMC优化...\n');
startTime = tic;
% 优化阶段控制
optimization_phase = 'morphology'; % 初始阶段
morphology_phase_end = round(maxIterations * config.morphologyPhaseRatio);
spatial_phase_end = round(maxIterations * config.spatialPhaseRatio);
for iter = 1:maxIterations
    iterStart = tic;
    if mod(iter, 25) == 1
        pool = [];
        try
            pool = gcp('nocreate');
        catch
            pool = [];
        end
        if isempty(pool)
            featureParallelEnabled = shouldUseParallel(max(8, batchSize), numel(mcmcState.model));
        else
            featureParallelEnabled = true;
        end
    end
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
    
    % 定期强制检查（降低频率以减少全局重算开销）
    if mod(iter, config.constraintInterval) == 0
        mcmcState = enforceComprehensiveConstraints(mcmcState, optParams, optimization_phase, featureParallelEnabled); % 减少全局重算频率以提升吞吐
        % 重新计算所有特征
        mcmcState = updateAllFeatures(mcmcState, featureParallelEnabled);
    end
    
    % 刷新边界上下文，限制移动在簇边界附近
    mcmcState = refreshBoundaryContext(mcmcState);
    % 自适应选择移动策略（用于占比控制）
    moveStrategy = selectComprehensiveAdaptiveMoveStrategy(iter, maxIterations, ...
        mcmcState, optimization_phase);
    % 生成批量候选移动（基于边界的交换与微调为主）
    [moves, moveTypes] = generateComprehensiveBatchMoves(mcmcState, ...
        batchSize, optParams, lookupTables, parallelEnabled, moveStrategy);
    
    % 评估移动
    deltaEnergies = evaluateComprehensiveBatchMoves(mcmcState, moves, ...
        moveTypes, lookupTables, optimization_phase, parallelEnabled);
    
    % 应用最佳移动
    [mcmcState, accepted] = applyBestComprehensiveMoves(mcmcState, moves, ...
        deltaEnergies, mcmcState.temperature, moveTypes, featureParallelEnabled);
    % 自适应温度调节，保持合理接受率
    if mcmcState.acceptanceRate > 0.6
        mcmcState.temperature = mcmcState.temperature * 0.97;
    elseif mcmcState.acceptanceRate < 0.2
        mcmcState.temperature = mcmcState.temperature * 1.03;
    end
    % 更新匹配度记录
    currentMorphMatch = calculateMorphologyMatch(mcmcState.morphologyFeatures, ...
        mcmcState.optParams.morphologyFeatures);
    currentSpatialMatch = calculateSpatialMatch(mcmcState.spatialFeatures, ...
        mcmcState.optParams.spatialFeatures);
    mcmcState.currentMorphMatch = currentMorphMatch;
    mcmcState.currentSpatialMatch = currentSpatialMatch;
    matchTolerance = config.checkpointTolerance; % 放宽回退容忍度以保留“先降后升”路径
    % 更新最佳匹配模型
    improvedMorph = currentMorphMatch > mcmcState.bestMorphMatch + matchTolerance;
    improvedSpatial = currentSpatialMatch > mcmcState.bestSpatialMatch + matchTolerance;
    if improvedMorph || improvedSpatial
        mcmcState.bestMorphMatch = max(mcmcState.bestMorphMatch, currentMorphMatch);
        mcmcState.bestSpatialMatch = max(mcmcState.bestSpatialMatch, currentSpatialMatch);
        mcmcState.bestMatchModel = mcmcState.model;
    end
    % 记录优化历史
    if isfield(mcmcState, 'optimizationHistory')
        mcmcState.optimizationHistory.morphologyMatches(end+1) = currentMorphMatch; %#ok<AGROW>
        mcmcState.optimizationHistory.spatialMatches(end+1) = currentSpatialMatch; %#ok<AGROW>
        mcmcState.optimizationHistory.iterations(end+1) = iter; %#ok<AGROW>
    end
    % 检查500次迭代退回条件
    checkpointInterval = config.checkpointEnabled * mcmcState.checkpointInterval;
    if checkpointInterval > 0 && mod(iter, checkpointInterval) == 0
        if (currentMorphMatch < mcmcState.checkpointBaselineMorph - matchTolerance) && ...
                (currentSpatialMatch < mcmcState.checkpointBaselineSpatial - matchTolerance)
            fprintf('  检测到形态/空间匹配度显著下降，回退到历史最佳模型 (迭代 %d)。\n', iter); % 放宽回退阈值避免频繁撤销进展
            mcmcState.model = mcmcState.bestMatchModel;
            mcmcState = updateAllFeatures(mcmcState, featureParallelEnabled);
            mcmcState.currentMorphMatch = calculateMorphologyMatch(mcmcState.morphologyFeatures, ...
                mcmcState.optParams.morphologyFeatures);
            mcmcState.currentSpatialMatch = calculateSpatialMatch(mcmcState.spatialFeatures, ...
                mcmcState.optParams.spatialFeatures);
            mcmcState.bestMorphMatch = max(mcmcState.bestMorphMatch, mcmcState.currentMorphMatch);
            mcmcState.bestSpatialMatch = max(mcmcState.bestSpatialMatch, mcmcState.currentSpatialMatch);
            mcmcState.bestMatchModel = mcmcState.model;
            mcmcState.bestEnergy = min(mcmcState.bestEnergy, mcmcState.currentEnergy);
            mcmcState.bestModel = mcmcState.bestMatchModel;
        end
        mcmcState.checkpointBaselineMorph = mcmcState.bestMorphMatch;
        mcmcState.checkpointBaselineSpatial = mcmcState.bestSpatialMatch;
        mcmcState.lastCheckpointIter = iter;
    end
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
            iter, optimization_phase, featureParallelEnabled);
    end
end
totalTime = toc(startTime);
fprintf('MCMC优化完成，总用时：%.2f秒\n', totalTime);
fprintf('平均每次迭代用时：%.4f秒\n', totalTime/maxIterations);
%% 6. 综合后处理
fprintf('正在进行综合后处理...\n');
% 先对最佳模型做孔隙率硬约束，避免后处理偏移目标
mcmcState.bestModel = enforcePorosityHardConstraint(mcmcState.bestModel, target_porosity, 50);
% 形态优先：总是以最佳形态模型作为后处理输入
basePostModel = mcmcState.bestShapeModel;
if isempty(basePostModel)
    basePostModel = mcmcState.bestModel;
end
mcmcState.bestEnergy = min(mcmcState.bestEnergy, computeUnifiedEnergySnapshot(basePostModel, optParams, weights));
if optParams.preserveSmallPores
    fprintf('>>> 使用小孔隙保留+形态保持模式 <<<\n');
    finalModel = comprehensivePostProcessShapePreserving(basePostModel, optParams);
else
    fprintf('>>> 使用平滑模式 <<<\n');
    finalModel = comprehensivePostProcess(basePostModel, optParams);
    % 后处理后再次精确校准孔隙率
    finalModel = enforcePorosityHardConstraint(finalModel, target_porosity, 50);
end
% 检查模型质量（统一参考模型）
checkComprehensiveModelQuality(finalModel, referenceBinaryModel, optParams);
%% 7. 保存结果
outFileName = sprintf('newModel_comprehensive_optimized_%dx%dx%d.raw', dims(1), dims(2), dims(3));
saveRawModel(finalModel, outFileName);
fprintf('优化模型已保存至 %s\n', outFileName);
% 显示最终结果
displayComprehensiveFinalResults(referenceBinaryModel, finalModel, originalFeatures, ...
    optParams.spatialFeatures, morphologyFeatures, performanceMonitor, optParams);
% 统一输出关键信息便于验证
finalPorosity = computeGlobalPorosity(finalModel);
finalPorosityRelErr = abs(finalPorosity - target_porosity) / max(target_porosity, eps);
finalMorphologyMatch = calculateMorphologyMatch(computeDetailedMorphologyFeatures(finalModel), optParams.morphologyFeatures);
finalSpatialMatch = calculateSpatialMatch(computeEnhancedSpatialFeatures(finalModel), optParams.spatialFeatures);
bestEnergy = mcmcState.bestEnergy;
bestEnergy = min(bestEnergy, computeUnifiedEnergySnapshot(mcmcState.bestModel, optParams, weights));
finalEnergy = computeUnifiedEnergySnapshot(finalModel, optParams, weights);
overallBestEnergy = min(bestEnergy, finalEnergy);
relDrop = (initialEnergy - overallBestEnergy) / max(initialEnergy, eps);
fprintf('\n==== 优化结果汇总 ====\n');
fprintf('原始灰度模型孔隙率: %.4f\n', raw_porosity);
fprintf('参考二值模型孔隙率: %.4f\n', referencePorosity);
fprintf('目标孔隙率: %.4f\n', target_porosity);
fprintf('初始模型孔隙率: %.4f\n', initial_porosity);
fprintf('最终模型孔隙率: %.4f (相对误差 %.4f)\n', finalPorosity, finalPorosityRelErr);
fprintf('最终形态匹配度: %.4f\n', finalMorphologyMatch);
fprintf('最终空间匹配度: %.4f\n', finalSpatialMatch);
fprintf('历史最优总能量: %.4f\n', overallBestEnergy);
fprintf('初始能量: %.4f | 最优能量相对下降: %.2f%%%%\n', initialEnergy, relDrop * 100);
fprintf('后处理后能量: %.4f\n', finalEnergy);
end
function [binaryModel, porosity] = binarizeWithUserThreshold(rawModel, pore_threshold)
    if nargin < 2 || isempty(pore_threshold)
        error('必须提供用户设定的孔隙阈值 pore_threshold。');
    end
    binaryModel = rawModel <= pore_threshold;
    porosity = computeGlobalPorosity(binaryModel);
end
function porosity = computeGlobalPorosity(binaryModel)
    porosity = mean(binaryModel(:));
end
function validateConfig(config)
    if ~isstruct(config)
        error('配置必须是结构体。');
    end
    numericFields = {'defaultBatchSize', 'maxBatchSize', 'constraintInterval', ...
        'checkpointTolerance', 'morphologyPhaseRatio', 'spatialPhaseRatio', 'directOverlapWeight'};
    for i = 1:numel(numericFields)
        name = numericFields{i};
        if ~isfield(config, name) || ~isfinite(config.(name))
            error('配置字段 %s 缺失或无效。', name);
        end
    end
end
function validateOptParams(optParams)
    if ~isstruct(optParams)
        error('优化参数必须是结构体。');
    end
    if ~isfield(optParams, 'targetPorosity') || ~isfinite(optParams.targetPorosity)
        error('必须提供有效的 targetPorosity。');
    end
    if ~isfield(optParams, 'modelSize') || numel(optParams.modelSize) ~= 3
        error('modelSize 必须是长度为3的向量。');
    end
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
    features.totalVoxels = sum(features.sizes);
    
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
    features.totalVoxels = 0;
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
    
    % 限制计算的簇数量（扩大到100，提高中小孔隙的可见度）
    nAnalyze = min(CC.NumObjects, 100);
    sizes = cellfun(@numel, CC.PixelIdxList);
    [~, sortIdx] = sort(sizes, 'descend');
    analyzeIdx = sortIdx(1:nAnalyze);
    
    % 初始化特征数组
    elongation = zeros(nAnalyze, 1);
    sphericity = zeros(nAnalyze, 1);
    convexity = zeros(nAnalyze, 1);
    solidity = zeros(nAnalyze, 1);
    modelSize = size(binaryModel);
    useParallel = shouldUseParallel(nAnalyze, numel(binaryModel));
    if useParallel && nAnalyze > 1
        parfor i = 1:nAnalyze
            clusterIdx = analyzeIdx(i);
            pixelIdx = CC.PixelIdxList{clusterIdx};
            [elong, spher, convx, solid] = computeClusterMorphologyMetrics(pixelIdx, modelSize);
            elongation(i) = elong;
            sphericity(i) = spher;
            convexity(i) = convx;
            solidity(i) = solid;
        end
    else
        for i = 1:nAnalyze
            clusterIdx = analyzeIdx(i);
            pixelIdx = CC.PixelIdxList{clusterIdx};
            [elong, spher, convx, solid] = computeClusterMorphologyMetrics(pixelIdx, modelSize);
            elongation(i) = elong;
            sphericity(i) = spher;
            convexity(i) = convx;
            solidity(i) = solid;
        end
    end
    morphologyFeatures.elongation = elongation;
    morphologyFeatures.sphericity = sphericity;
    morphologyFeatures.convexity = convexity;
    morphologyFeatures.solidity = solidity;
    
    % 计算网络特征
    morphologyFeatures.poreNetworkDensity = computePoreNetworkDensity(binaryModel);
    morphologyFeatures.coordinationNumber = computeAverageCoordinationNumber(binaryModel);
    
    % 计算其他形态特征
    morphologyFeatures.lacunarity = computeLacunarity(binaryModel);
    morphologyFeatures.textureFeatures = computeTextureFeatures(binaryModel);
    morphologyFeatures.skeletonFeatures = computeSkeletonFeatures(binaryModel);
end
function [elongation, sphericity, convexity, solidity] = ...
    computeClusterMorphologyMetrics(pixelIdx, modelSize)
    % 计算单个簇的核心形态指标
    [x, y, z] = ind2sub(modelSize, pixelIdx);
    nVoxels = numel(x);
    if nVoxels <= 10
        elongation = 1;
        sphericity = 0.8;
        convexity = 0.8;
        solidity = 0.8;
        return;
    end
    coords = [double(x) - mean(x), double(y) - mean(y), double(z) - mean(z)];
    elongation = 1;
    sphericity = 0.5;
    convexity = 0.5;
    solidity = 0.5;
    try
        C = cov(coords);
        if any(isnan(C(:)))
            C = eye(3);
        end
        eigenvalues = eig(C);
        eigenvalues = sort(max(eigenvalues, eps), 'descend');
        if numel(eigenvalues) < 3
            eigenvalues(3) = eigenvalues(end);
        end
    catch
        eigenvalues = [1; 1; 1];
    end
    elongation = sqrt(eigenvalues(1) / max(eigenvalues(3), eps));
    volume = nVoxels;
    a = sqrt(eigenvalues(1));
    b = sqrt(eigenvalues(2));
    c = sqrt(eigenvalues(3));
    ellipsoidVolume = (4/3) * pi * a * b * c;
    if ellipsoidVolume > 0
        sphericity = min(1, max(0, (volume / ellipsoidVolume)^(1/3)));
    else
        sphericity = 0.5;
    end
    rangeX = double(max(x) - min(x) + 1);
    rangeY = double(max(y) - min(y) + 1);
    rangeZ = double(max(z) - min(z) + 1);
    boundingBoxVolume = rangeX * rangeY * rangeZ;
    if boundingBoxVolume > 0
        convexity = min(1, volume / boundingBoxVolume);
    else
        convexity = 1;
    end
    solidity = convexity;
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
    % 估计迂曲度，优先使用贯通孔隙的测地距离
    [nx, ny, nz] = size(binaryModel);
    % 确认是否存在贯通的孔隙通道
    labelMap = bwlabeln(binaryModel, 26);
    topLabels = unique(labelMap(:,:,1));
    bottomLabels = unique(labelMap(:,:,end));
    percolatingLabels = intersect(topLabels(topLabels>0), bottomLabels(bottomLabels>0));
    if isempty(percolatingLabels)
        tortuosity = 2.0; % 无贯通路径时给出保守估计
        return;
    end
    percolatingMask = ismember(labelMap, percolatingLabels);
    startMask = percolatingMask(:,:,1);
    endMask = percolatingMask(:,:,end);
    startPoints = find(startMask);
    nSamples = min(30, numel(startPoints));
    tortuositySamples = zeros(nSamples, 1);
    for s = 1:nSamples
        startIdx = startPoints(randi(numel(startPoints)));
         % 计算到末端平面的测地距离（使用显式种子掩码避免维度解析错误）
        startSeedMask = false(size(percolatingMask));
        startSeedMask(startIdx) = true;
        % 使用三维“quasi-euclidean”度量等效于 26 邻域，避免将数值误判为 method 参数
        geodesicMap = bwdistgeodesic(percolatingMask, startSeedMask, 'quasi-euclidean');
        reachable = geodesicMap(endMask);
        reachable = reachable(~isnan(reachable));
        if isempty(reachable)
            tortuositySamples(s) = NaN;
            continue;
        end
        shortestPath = min(reachable);
        straightDistance = nz - 1;
        if straightDistance > 0
            tortuositySamples(s) = shortestPath / straightDistance;
        end
    end
    tortuositySamples = tortuositySamples(~isnan(tortuositySamples) & tortuositySamples > 0);
    if isempty(tortuositySamples)
        tortuosity = 2.0;
    else
        tortuosity = mean(tortuositySamples);
    end
    % 合理范围限制
    tortuosity = max(1, min(tortuosity, 5));
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
    % 计算空隙率（多尺度卷积版）
    scales = [3, 5, 9, 15];
    lacunarityValues = zeros(length(scales), 1);
    modelDouble = double(model);
    for i = 1:length(scales)
        k = scales(i);
        kernel = ones(k, k, k);
        massMap = convn(modelDouble, kernel, 'valid');
        mu = mean(massMap(:));
        if mu > 0
            lacunarityValues(i) = var(massMap(:)) / (mu^2);
        else
            lacunarityValues(i) = 0;
        end
    end
    lacunarity = mean(lacunarityValues);
end
function textureFeatures = computeTextureFeatures(model)
    % 计算纹理特征（体素多切片版）
    textureFeatures = struct();
    slices = {
        double(model(:, :, round(size(model, 3)/2))), ...
        double(squeeze(model(:, round(size(model, 2)/2), :))), ...
        double(squeeze(model(round(size(model, 1)/2), :, :)))
    };
    entropyVals = zeros(numel(slices), 1);
    energyVals = zeros(numel(slices), 1);
    contrastVals = zeros(numel(slices), 1);
    homogeneityVals = zeros(numel(slices), 1);
    for i = 1:numel(slices)
        slice = slices{i};
        p = slice(:) / (sum(slice(:)) + eps);
        p(p == 0) = [];
        entropyVals(i) = -sum(p .* log2(p + eps));
        energyVals(i) = sum(slice(:).^2) / numel(slice);
        [gxSlice, gySlice] = gradient(slice);
        contrastVals(i) = mean(sqrt(gxSlice(:).^2 + gySlice(:).^2));
        localStd = stdfilt(slice, ones(5));
        homogeneityVals(i) = 1 / (mean(localStd(:)) + 1);
    end
    % 体素梯度整体能量
    [gx, gy, gz] = gradient(double(model));
    textureFeatures.gradientEnergy = mean(sqrt(gx(:).^2 + gy(:).^2 + gz(:).^2));
    textureFeatures.entropy = mean(entropyVals);
    textureFeatures.energy = mean(energyVals);
    textureFeatures.contrast = mean(contrastVals);
    textureFeatures.homogeneity = mean(homogeneityVals);
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
    % 使用卷积在CPU上实现高度向量化的滑窗平均，显著提升性能
    kernel = ones(windowSize, windowSize, windowSize, 'double');
    % 计算每个位置周围的孔隙体素数量
    localSum = convn(double(binaryModel), kernel, 'same');
    % 计算边界处有效的体素数量，避免零填充造成偏差
    validCount = convn(ones(size(binaryModel), 'double'), kernel, 'same');
    localPorosityMap = localSum ./ max(validCount, 1);
end
function gradientModel = computeSpatialParameterGradients(rawBinaryModel, baseSpatialFeatures, rawPorosity)
    % 构建孔隙率与空间特征之间的梯度/查找表模型
    if nargin < 3 || isempty(rawPorosity)
        rawPorosity = computeGlobalPorosity(rawBinaryModel);
    end
    dims = size(rawBinaryModel);
    regionVolume = prod(dims);
    % 定义重叠区域的窗口和步长
    baseWindow = max(round(min(dims) * 0.5), 20);
    baseWindow = min([baseWindow, dims]);
    windowSize = min(baseWindow);
    stride = max(round(windowSize * 0.5), 10);
    stride = min(stride, max(windowSize - 1, 1));
    stride = max(3, stride);
    % 使用放大的局部孔隙率图作为区域采样的启发
    coarseWindow = max(ceil(windowSize / 2), 6);
    coarseWindow = min([coarseWindow, dims]);
    coarsePorosityMap = computeLocalPorosityMap(rawBinaryModel, coarseWindow);
    % 构造起始索引，确保覆盖整个区域
    xStarts = unique([1:stride:max(1, dims(1) - windowSize + 1), max(1, dims(1) - windowSize + 1)]);
    yStarts = unique([1:stride:max(1, dims(2) - windowSize + 1), max(1, dims(2) - windowSize + 1)]);
    zStarts = unique([1:stride:max(1, dims(3) - windowSize + 1), max(1, dims(3) - windowSize + 1)]);
    % 记录候选区域
    regionRecords = [];
    for ix = 1:length(xStarts)
        for iy = 1:length(yStarts)
            for iz = 1:length(zStarts)
                x1 = xStarts(ix);
                y1 = yStarts(iy);
                z1 = zStarts(iz);
                x2 = min(x1 + windowSize - 1, dims(1));
                y2 = min(y1 + windowSize - 1, dims(2));
                z2 = min(z1 + windowSize - 1, dims(3));
                centerX = round((x1 + x2) / 2);
                centerY = round((y1 + y2) / 2);
                centerZ = round((z1 + z2) / 2);
                centerX = min(max(centerX, 1), dims(1));
                centerY = min(max(centerY, 1), dims(2));
                centerZ = min(max(centerZ, 1), dims(3));
                localPorosity = coarsePorosityMap(centerX, centerY, centerZ);
                regionRecords = [regionRecords; x1, x2, y1, y2, z1, z2, localPorosity]; %#ok<AGROW>
            end
        end
    end
    % 根据孔隙率与全局孔隙率的偏差排序，覆盖不同孔隙率区域
    if ~isempty(regionRecords)
        [~, order] = sort(abs(regionRecords(:,7) - rawPorosity), 'descend');
        regionRecords = regionRecords(order, :);
    end
    dataset = [];
    porositySamples = [];
    summaryNames = {};
    summaryList = {};
    for idx = 1:size(regionRecords, 1)
        bounds = regionRecords(idx, :);
        x1 = bounds(1); x2 = bounds(2);
        y1 = bounds(3); y2 = bounds(4);
        z1 = bounds(5); z2 = bounds(6);
        subVolume = rawBinaryModel(x1:x2, y1:y2, z1:z2);
        localPorosity = mean(subVolume(:));
        if ~isfinite(localPorosity)
            continue;
        end
        localFeatures = computeEnhancedSpatialFeatures(subVolume);
        [values, names, summaryStruct] = summarizeSpatialFeaturesForRegression(localFeatures, numel(subVolume));
        if isempty(summaryNames)
            summaryNames = names;
        end
        if isempty(values) || any(~isfinite(values))
            continue;
        end
        dataset = [dataset; localPorosity, values(:)']; %#ok<AGROW>
        porositySamples = [porositySamples; localPorosity]; %#ok<AGROW>
        summaryList{end+1} = summaryStruct; %#ok<AGROW>
    end
    % 使用原始模型作为基准
    [baseValues, baseNames, baseSummaryStruct] = summarizeSpatialFeaturesForRegression(baseSpatialFeatures, regionVolume);
    if isempty(summaryNames)
        summaryNames = baseNames;
    end
    % 过滤孔隙率范围
    porosityRange = [max(0, rawPorosity - 0.2), min(1, rawPorosity + 0.2)];
    if ~isempty(dataset)
        inRange = dataset(:,1) >= porosityRange(1) & dataset(:,1) <= porosityRange(2);
        if sum(inRange) >= 3
            dataset = dataset(inRange, :);
            porositySamples = porositySamples(inRange);
        end
    end
    models = struct('name', {}, 'coeffs', {}, 'degree', {}, 'rmse', {}, 'r2', {});
    if isempty(dataset)
        % 无足够数据时，退化为常量模型
        dataset = [rawPorosity, baseValues(:)'];
    end
    featureCount = size(dataset, 2) - 1;
    for col = 1:featureCount
        y = dataset(:, col + 1);
        x = dataset(:, 1);
        validMask = isfinite(x) & isfinite(y);
        x = x(validMask);
        y = y(validMask);
        if numel(unique(x)) <= 1
            coeffs = y(1);
            degree = 0;
            yPred = repmat(y(1), size(y));
        else
            if numel(x) >= 6
                degree = 2;
            else
                degree = 1;
            end
            coeffs = polyfit(x, y, degree);
            yPred = polyval(coeffs, x);
        end
        rmse = sqrt(mean((yPred - y).^2));
        if numel(y) > 1
            r2 = 1 - sum((y - yPred).^2) / max(sum((y - mean(y)).^2), eps);
        else
            r2 = 1;
        end
        models(col).name = summaryNames{col}; %#ok<*AGROW>
        models(col).coeffs = coeffs;
        models(col).degree = degree;
        models(col).rmse = rmse;
        models(col).r2 = r2;
    end
    gradientModel = struct();
    gradientModel.porosityRange = porosityRange;
    gradientModel.models = models;
    gradientModel.summaryNames = summaryNames;
    gradientModel.baseSummaryVector = baseValues(:)';
    gradientModel.baseSummaryStruct = baseSummaryStruct;
    gradientModel.dataset = dataset;
    gradientModel.rawPorosity = rawPorosity;
    gradientModel.windowSize = windowSize;
    gradientModel.stride = stride;
    gradientModel.globalVolume = regionVolume;
    gradientModel.sampledSummaries = summaryList;
end
function [values, names, summaryStruct] = summarizeSpatialFeaturesForRegression(features, regionVolume)
    % 将空间特征转换为用于回归的摘要统计量
    if nargin < 2 || isempty(regionVolume)
        regionVolume = 1;
    end
    summaryStruct = struct();
    if isfield(features, 'twoPointCorr') && ~isempty(features.twoPointCorr)
        vals = double(features.twoPointCorr(:));
        summaryStruct.twoPoint_mean = mean(vals);
        summaryStruct.twoPoint_std = std(vals);
    else
        summaryStruct.twoPoint_mean = 0;
        summaryStruct.twoPoint_std = 0;
    end
    if isfield(features, 'anisotropy')
        summaryStruct.anisotropy = double(features.anisotropy);
    else
        summaryStruct.anisotropy = 0;
    end
    if isfield(features, 'porosityGradient')
        summaryStruct.porosityGradient = double(features.porosityGradient);
    else
        summaryStruct.porosityGradient = 0;
    end
    if isfield(features, 'connectivity') && ~isempty(features.connectivity)
        conn = features.connectivity;
        if isfield(conn, 'eulerNumber')
            summaryStruct.connectivity_euler = double(conn.eulerNumber);
        else
            summaryStruct.connectivity_euler = 0;
        end
        if isfield(conn, 'numComponents')
            summaryStruct.connectivity_componentDensity = double(conn.numComponents) / max(regionVolume, 1);
        else
            summaryStruct.connectivity_componentDensity = 0;
        end
        if isfield(conn, 'largestComponentRatio')
            summaryStruct.connectivity_largestRatio = double(conn.largestComponentRatio);
        else
            summaryStruct.connectivity_largestRatio = 0;
        end
        if isfield(conn, 'connectivityDensity')
            summaryStruct.connectivity_density = double(conn.connectivityDensity);
        else
            summaryStruct.connectivity_density = summaryStruct.connectivity_componentDensity;
        end
    else
        summaryStruct.connectivity_euler = 0;
        summaryStruct.connectivity_componentDensity = 0;
        summaryStruct.connectivity_largestRatio = 0;
        summaryStruct.connectivity_density = 0;
    end
    if isfield(features, 'surfaceToVolumeRatio')
        summaryStruct.surfaceToVolumeRatio = double(features.surfaceToVolumeRatio);
    else
        summaryStruct.surfaceToVolumeRatio = 0;
    end
    if isfield(features, 'tortuosityEstimate')
        summaryStruct.tortuosityEstimate = double(features.tortuosityEstimate);
    else
        summaryStruct.tortuosityEstimate = 0;
    end
    if isfield(features, 'spatialAutocorrelation')
        summaryStruct.spatialAutocorrelation = double(features.spatialAutocorrelation);
    else
        summaryStruct.spatialAutocorrelation = 0;
    end
    if isfield(features, 'chordLengthDistribution') && ~isempty(features.chordLengthDistribution)
        cld = features.chordLengthDistribution;
        summaryStruct.chord_meanLength = safeField(cld, 'meanLength');
        summaryStruct.chord_stdLength = safeField(cld, 'stdLength');
    else
        summaryStruct.chord_meanLength = 0;
        summaryStruct.chord_stdLength = 0;
    end
    if isfield(features, 'poreSizeDistribution') && ~isempty(features.poreSizeDistribution)
        psd = features.poreSizeDistribution;
        summaryStruct.pore_meanRadius = safeField(psd, 'meanRadius');
        summaryStruct.pore_stdRadius = safeField(psd, 'stdRadius');
        summaryStruct.pore_maxRadius = safeField(psd, 'maxRadius');
    else
        summaryStruct.pore_meanRadius = 0;
        summaryStruct.pore_stdRadius = 0;
        summaryStruct.pore_maxRadius = 0;
    end
    if isfield(features, 'minkowskiFunctionals') && ~isempty(features.minkowskiFunctionals)
        mk = features.minkowskiFunctionals;
        summaryStruct.minkowski_volumeDensity = safeField(mk, 'volume') / max(regionVolume, 1);
        summaryStruct.minkowski_surfaceDensity = safeField(mk, 'surfaceArea') / max(regionVolume, 1);
        summaryStruct.minkowski_meanBreadthDensity = safeField(mk, 'meanBreadth') / max(regionVolume, 1);
        summaryStruct.minkowski_integralMeanCurvatureDensity = safeField(mk, 'integralMeanCurvature') / max(regionVolume, 1);
        summaryStruct.minkowski_eulerDensity = safeField(mk, 'eulerCharacteristic') / max(regionVolume, 1);
    else
        summaryStruct.minkowski_volumeDensity = 0;
        summaryStruct.minkowski_surfaceDensity = 0;
        summaryStruct.minkowski_meanBreadthDensity = 0;
        summaryStruct.minkowski_integralMeanCurvatureDensity = 0;
        summaryStruct.minkowski_eulerDensity = 0;
    end
    if isfield(features, 'linealPathFunction') && ~isempty(features.linealPathFunction)
        lp = features.linealPathFunction;
        meanVals = double(lp.mean(:));
        meanVals = meanVals(isfinite(meanVals));
        if isempty(meanVals)
            summaryStruct.lineal_meanValue = 0;
            summaryStruct.lineal_shortRange = 0;
            summaryStruct.lineal_longRange = 0;
        else
            summaryStruct.lineal_meanValue = mean(meanVals);
            summaryStruct.lineal_shortRange = meanVals(1);
            summaryStruct.lineal_longRange = meanVals(end);
        end
    else
        summaryStruct.lineal_meanValue = 0;
        summaryStruct.lineal_shortRange = 0;
        summaryStruct.lineal_longRange = 0;
    end
    names = fieldnames(summaryStruct);
    values = zeros(1, numel(names));
    for i = 1:numel(names)
        val = summaryStruct.(names{i});
        if ~isfinite(val)
            val = 0;
        end
        values(i) = val;
    end
end
function val = safeField(structure, fieldName)
    if isfield(structure, fieldName) && ~isempty(structure.(fieldName))
        val = double(structure.(fieldName));
    else
        val = 0;
    end
    if ~isfinite(val)
        val = 0;
    end
end
function [targetSpatialFeatures, summaryStruct] = predictSpatialFeaturesFromGradients(gradientModel, targetPorosity, baseSpatialFeatures, targetVolume)
    % 根据梯度模型预测目标空间特征
    lockThreshold = 0.02;            % 孔隙率变化很小，直接使用原始特征
    maxChangeRatio = 0.20;           % 梯度外推允许的相对变化上限，防止目标偏离物理合理区间
    if nargin < 4 || isempty(targetVolume)
        targetVolume = gradientModel.globalVolume;
    end
    if isempty(gradientModel) || ~isfield(gradientModel, 'models')
        summaryStruct = gradientModel.baseSummaryStruct;
        targetSpatialFeatures = baseSpatialFeatures;
        return;
    end
    targetPorosity = max(0, min(1, targetPorosity));
    porosityRange = gradientModel.porosityRange;
    targetPorosity = max(porosityRange(1), min(porosityRange(2), targetPorosity));
    porDiff = abs(targetPorosity - gradientModel.rawPorosity);
    % 孔隙率变化极小，直接锁定原始特征
    if porDiff < lockThreshold
        summaryStruct = gradientModel.baseSummaryStruct;
        targetSpatialFeatures = baseSpatialFeatures;
        return;
    end
    summaryStruct = struct();
    for i = 1:numel(gradientModel.models)
        model = gradientModel.models(i);
        coeffs = model.coeffs;
        if numel(coeffs) == 1
            value = coeffs;
        else
            value = polyval(coeffs, targetPorosity);
        end
        summaryStruct.(model.name) = value;
    end
    % 补充缺失字段并做限幅，避免梯度外推使特征偏离 ±20%
    baseNames = fieldnames(gradientModel.baseSummaryStruct);
    for i = 1:numel(baseNames)
        name = baseNames{i};
        baseVal = gradientModel.baseSummaryStruct.(name);
        if ~isfield(summaryStruct, name)
            summaryStruct.(name) = baseVal;
        end
        predictedVal = summaryStruct.(name);
        maxDelta = maxChangeRatio * max(abs(baseVal), eps);
        upper = baseVal + maxDelta;
        lower = baseVal - maxDelta;
        if isnan(predictedVal)
            summaryStruct.(name) = baseVal;
        else
            summaryStruct.(name) = min(upper, max(lower, predictedVal));
        end
    end
    targetSpatialFeatures = reconstructSpatialFeaturesFromSummaries( ...
        baseSpatialFeatures, summaryStruct, gradientModel, targetVolume);
end
function targetSpatialFeatures = reconstructSpatialFeaturesFromSummaries(baseSpatialFeatures, summaryStruct, gradientModel, targetVolume)
    % 将摘要统计转换回完整的空间特征结构
    targetSpatialFeatures = baseSpatialFeatures;
    baseSummary = gradientModel.baseSummaryStruct;
    if nargin < 4 || isempty(targetVolume)
        targetVolume = gradientModel.globalVolume;
    end
    % 两点相关函数调整
    if isfield(baseSpatialFeatures, 'twoPointCorr') && isfield(summaryStruct, 'twoPoint_mean')
        baseTPC = double(baseSpatialFeatures.twoPointCorr);
        baseMean = mean(baseTPC(:));
        baseStd = std(baseTPC(:));
        targetMean = summaryStruct.twoPoint_mean;
        targetStd = summaryStruct.twoPoint_std;
        adjusted = baseTPC;
        if isfinite(baseStd) && baseStd > 0 && isfield(summaryStruct, 'twoPoint_std')
            scaleStd = targetStd / max(baseSummary.twoPoint_std, eps);
            adjusted = (adjusted - baseMean) * scaleStd + baseMean;
        end
        adjusted = adjusted + (targetMean - mean(adjusted(:)));
        adjusted = max(0, adjusted);
        targetSpatialFeatures.twoPointCorr = adjusted;
    end
    if isfield(summaryStruct, 'anisotropy')
        targetSpatialFeatures.anisotropy = max(0, summaryStruct.anisotropy);
    end
    if isfield(summaryStruct, 'porosityGradient')
        targetSpatialFeatures.porosityGradient = max(0, summaryStruct.porosityGradient);
    end
    if isfield(targetSpatialFeatures, 'connectivity')
        conn = targetSpatialFeatures.connectivity;
        if isfield(summaryStruct, 'connectivity_euler')
            conn.eulerNumber = summaryStruct.connectivity_euler;
        end
        if isfield(summaryStruct, 'connectivity_density')
            conn.connectivityDensity = max(0, summaryStruct.connectivity_density);
        end
        if isfield(summaryStruct, 'connectivity_largestRatio')
            conn.largestComponentRatio = max(0, min(1, summaryStruct.connectivity_largestRatio));
        end
        if isfield(summaryStruct, 'connectivity_componentDensity')
            conn.numComponents = max(1, round(summaryStruct.connectivity_componentDensity * targetVolume));
        end
        targetSpatialFeatures.connectivity = conn;
    end
    if isfield(summaryStruct, 'surfaceToVolumeRatio')
        targetSpatialFeatures.surfaceToVolumeRatio = max(0, summaryStruct.surfaceToVolumeRatio);
    end
    if isfield(summaryStruct, 'tortuosityEstimate')
        targetSpatialFeatures.tortuosityEstimate = max(1, summaryStruct.tortuosityEstimate);
    end
    if isfield(summaryStruct, 'spatialAutocorrelation')
        targetSpatialFeatures.spatialAutocorrelation = summaryStruct.spatialAutocorrelation;
    end
    if isfield(targetSpatialFeatures, 'chordLengthDistribution') && ~isempty(targetSpatialFeatures.chordLengthDistribution)
        cld = targetSpatialFeatures.chordLengthDistribution;
        baseMean = baseSummary.chord_meanLength;
        if isfield(summaryStruct, 'chord_meanLength') && baseMean > 0
            ratio = summaryStruct.chord_meanLength / max(baseMean, eps);
            cld.binCenters = cld.binCenters * ratio;
            if isfield(cld, 'directionalStats')
                dirNames = fieldnames(cld.directionalStats);
                for i = 1:numel(dirNames)
                    name = dirNames{i};
                    cld.directionalStats.(name).mean = cld.directionalStats.(name).mean * ratio;
                    cld.directionalStats.(name).std = cld.directionalStats.(name).std * ratio;
                end
            end
        end
        if isfield(summaryStruct, 'chord_meanLength')
            cld.meanLength = summaryStruct.chord_meanLength;
        end
        if isfield(summaryStruct, 'chord_stdLength')
            cld.stdLength = max(0, summaryStruct.chord_stdLength);
        end
        targetSpatialFeatures.chordLengthDistribution = cld;
    end
    if isfield(targetSpatialFeatures, 'poreSizeDistribution') && ~isempty(targetSpatialFeatures.poreSizeDistribution)
        psd = targetSpatialFeatures.poreSizeDistribution;
        baseMeanRadius = baseSummary.pore_meanRadius;
        if isfield(summaryStruct, 'pore_meanRadius') && baseMeanRadius > 0
            ratio = summaryStruct.pore_meanRadius / max(baseMeanRadius, eps);
            psd.binCenters = psd.binCenters * ratio;
        end
        if isfield(summaryStruct, 'pore_meanRadius')
            psd.meanRadius = summaryStruct.pore_meanRadius;
        end
        if isfield(summaryStruct, 'pore_stdRadius')
            psd.stdRadius = max(0, summaryStruct.pore_stdRadius);
        end
        if isfield(summaryStruct, 'pore_maxRadius')
            psd.maxRadius = max(0, summaryStruct.pore_maxRadius);
        end
        targetSpatialFeatures.poreSizeDistribution = psd;
    end
    if isfield(targetSpatialFeatures, 'minkowskiFunctionals') && ~isempty(targetSpatialFeatures.minkowskiFunctionals)
        mk = targetSpatialFeatures.minkowskiFunctionals;
        mk.volume = summaryStruct.minkowski_volumeDensity * targetVolume;
        mk.surfaceArea = summaryStruct.minkowski_surfaceDensity * targetVolume;
        mk.meanBreadth = summaryStruct.minkowski_meanBreadthDensity * targetVolume;
        mk.integralMeanCurvature = summaryStruct.minkowski_integralMeanCurvatureDensity * targetVolume;
        mk.eulerCharacteristic = summaryStruct.minkowski_eulerDensity * targetVolume;
        targetSpatialFeatures.minkowskiFunctionals = mk;
    end
    if isfield(targetSpatialFeatures, 'linealPathFunction') && ~isempty(targetSpatialFeatures.linealPathFunction)
        lp = targetSpatialFeatures.linealPathFunction;
        baseMeanVals = lp.mean;
        if ~isempty(baseMeanVals) && isfield(summaryStruct, 'lineal_meanValue')
            baseMean = mean(baseMeanVals(:));
            if baseMean ~= 0
                ratio = summaryStruct.lineal_meanValue / baseMean;
                lp.values = lp.values * ratio;
                lp.mean = lp.mean * ratio;
                if isfield(lp, 'byDirection')
                    dirNames = fieldnames(lp.byDirection);
                    for i = 1:numel(dirNames)
                        lp.byDirection.(dirNames{i}) = lp.byDirection.(dirNames{i}) * ratio;
                    end
                end
            end
            if isfield(summaryStruct, 'lineal_shortRange') && ~isempty(lp.mean)
                offset = summaryStruct.lineal_shortRange - lp.mean(1);
                lp.mean = lp.mean + offset;
                lp.values = lp.values + offset;
                if isfield(lp, 'byDirection')
                    dirNames = fieldnames(lp.byDirection);
                    for i = 1:numel(dirNames)
                        lp.byDirection.(dirNames{i}) = lp.byDirection.(dirNames{i}) + offset;
                    end
                end
            end
        end
        targetSpatialFeatures.linealPathFunction = lp;
    end
end
function clusterReference = buildClusterReference(originalFeatures, targetVolume)
    % 构建用于软约束的簇分布参考
    clusterReference = struct();
    if nargin < 2 || isempty(targetVolume)
        targetVolume = [];
    end
    if isempty(originalFeatures) || ~isfield(originalFeatures, 'sizes') || isempty(originalFeatures.sizes)
        clusterReference.numClusters = 0;
        clusterReference.meanSize = 0;
        clusterReference.stdSize = 0;
        clusterReference.minSize = 0;
        clusterReference.maxSize = 0;
        clusterReference.histEdges = linspace(0, 1, 11);
        clusterReference.histogram = ones(1, 10) / 10;
        clusterReference.totalVoxels = 0;
        return;
    end
    sizes = double(originalFeatures.sizes(:));
    clusterReference.numClusters = numel(sizes);
    clusterReference.meanSize = mean(sizes);
    clusterReference.stdSize = std(sizes);
    clusterReference.minSize = min(sizes);
    clusterReference.maxSize = max(sizes);
    clusterReference.totalVoxels = sum(sizes);
    logSizes = log10(sizes + 1);
    minLog = min(logSizes);
    maxLog = max(logSizes);
    if maxLog - minLog < 1e-6
        maxLog = minLog + 1;
    end
    edges = linspace(minLog, maxLog, 11);
    histogram = histcounts(logSizes, edges, 'Normalization', 'probability');
    if all(histogram == 0)
        histogram = ones(1, numel(histogram)) / numel(histogram);
    end
    clusterReference.histEdges = edges;
    clusterReference.histogram = histogram;
    clusterReference.sizeCDF = cumsum(histcounts(logSizes, edges, 'Normalization', 'cdf'));
    clusterReference.cdfEdges = edges;
    if ~isempty(targetVolume)
        clusterReference.volume = targetVolume;
    end
end
function clusterTarget = buildAdaptiveClusterTargets(clusterStats_ref, referencePorosity, targetPorosity, modelSize)
    % 基于参考簇统计与目标孔隙率自适应构建簇目标
    if nargin < 4 || isempty(modelSize)
        modelSize = size(clusterStats_ref.sizes);
    end
    if isempty(referencePorosity) || referencePorosity <= 0
        referencePorosity = max(sum(clusterStats_ref.sizes) / prod(modelSize), eps);
    end
    r = targetPorosity / max(referencePorosity, eps);
    N0 = clusterStats_ref.numClusters;
    N_target = max(1, round(N0 * r));
    sizes_ref = double(clusterStats_ref.sizes(:));
    if isempty(sizes_ref)
        sizes_ref = 1;
    end
    sizes_scaled = sizes_ref * r;
    sizes_scaled(sizes_scaled < 1) = 1;
    targetVolume = targetPorosity * prod(modelSize);
    k = targetVolume / max(sum(sizes_scaled), eps);
    sizes_target = max(1, round(sizes_scaled * k));
    if numel(sizes_target) > N_target
        sizes_target = sizes_target(1:N_target);
    elseif numel(sizes_target) < N_target
        sizes_target = [sizes_target; ones(N_target - numel(sizes_target), 1)];
    end
    currentVolume = sum(sizes_target);
    if currentVolume > 0
        correction = targetVolume / currentVolume;
        sizes_target = max(1, round(sizes_target * correction));
    end
    s_max_ref = max(clusterStats_ref.sizes);
    s_max_target = s_max_ref * nthroot(r, 3);
    edges = unique(round(logspace(log10(max(1, min(sizes_target))), log10(max(sizes_target)+1), 12)))';
    if numel(edges) < 2
        edges = [0, max(sizes_target)+1];
    end
    hist_target = histcounts(sizes_target, edges, 'Normalization', 'probability');
    clusterTarget = struct();
    clusterTarget.numClusters = N_target;
    clusterTarget.sizeArray = sizes_target(:);
    clusterTarget.meanSize = mean(sizes_target);
    clusterTarget.maxSize = max(sizes_target);
    clusterTarget.maxSizeTarget = max(s_max_target, clusterTarget.maxSize);
    clusterTarget.histEdges = edges;
    clusterTarget.histogram = hist_target;
    clusterTarget.totalVoxels = sum(sizes_target);
end
function [lowerBound, upperBound] = getAdaptiveClusterBounds(optParams)
    % 根据参考分布自适应地估计簇大小范围
    lowerBound = [];
    upperBound = [];
    if isfield(optParams, 'clusterReference') && ~isempty(optParams.clusterReference)
        ref = optParams.clusterReference;
        if ref.meanSize > 0
            lowerBound = max(1, round(ref.meanSize - 0.75 * ref.stdSize));
            lowerBound = max(lowerBound, round(ref.minSize * 0.8));
            upperBound = max(lowerBound + 1, round(ref.meanSize + 2.5 * ref.stdSize));
            upperBound = max(upperBound, round(ref.maxSize * 0.9));
        end
    end
    if isempty(lowerBound) || isempty(upperBound)
        if isfield(optParams, 'targetMin') && isfield(optParams, 'targetMax') && ...
                ~isempty(optParams.targetMin) && ~isempty(optParams.targetMax)
            lowerBound = optParams.targetMin;
            upperBound = optParams.targetMax;
        else
            lowerBound = 0;
            upperBound = 0;
        end
    end
end
function [minCount, maxCount] = getAdaptiveClusterCountBounds(optParams)
    % 根据参考分布估计簇数量的合理范围
    minCount = 0;
    maxCount = Inf;
    if isfield(optParams, 'clusterReference') && ~isempty(optParams.clusterReference)
        refCount = max(1, optParams.clusterReference.numClusters);
        minCount = max(1, round(refCount * 0.4));
        maxCount = round(refCount * 2.5);
    elseif isfield(optParams, 'targetClusterCount') && ~isempty(optParams.targetClusterCount)
        refCount = max(1, round(optParams.targetClusterCount));
        minCount = max(1, round(refCount * 0.5));
        maxCount = round(refCount * 2);
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
    % 生成综合优化的初始模型（基于自适应簇目标与泊松球心分布）
    fprintf('  生成综合优化的初始模型...');
    anisotropyProfile = estimateAnisotropyFromFeatures(spatialFeatures, morphologyFeatures);
    clusterTarget = optParams.clusterTarget;
    [model, clusterSummary] = generateClusterBasedInitialModel(dims, targetPorosity, clusterTarget, anisotropyProfile, optParams);
    spatialSummary = computeEnhancedSpatialFeatures(model);
    morphSummary = computeDetailedMorphologyFeatures(model);
    fprintf('\n    初始簇数: %d (目标 %d)，最大簇: %d / %.0f，孔隙率: %.4f\n', ...
        clusterSummary.numClusters, clusterTarget.numClusters, ...
        max(clusterSummary.sizes), clusterTarget.maxSizeTarget, mean(model(:)));
    fprintf('    空间匹配度: %.3f, 形态匹配度: %.3f\n', ...
        calculateSpatialMatch(spatialSummary, spatialFeatures), ...
        calculateMorphologyMatch(morphSummary, morphologyFeatures));
    mcmcModel = model;
end
function model = applyBoundaryPerturbation(referenceModel, flipRatio, maxRadius)
    % 在参考模型的孔隙-基质边界做极少量扰动，保持整体形态
    if nargin < 2 || isempty(flipRatio)
        flipRatio = 0.002;
    end
    if nargin < 3 || isempty(maxRadius)
        maxRadius = 2;
    end
    model = logical(referenceModel);
    boundary = bwperim(model, 26) | bwperim(~model, 26);
    if ~any(boundary(:))
        return;
    end
    padded = imdilate(boundary, strel('sphere', maxRadius));
    candidateIdx = find(padded);
    nFlip = max(1, round(numel(candidateIdx) * flipRatio));
    chosen = candidateIdx(randperm(numel(candidateIdx), nFlip));
    model(chosen) = ~model(chosen);
end
function [model, summary] = generateClusterBasedInitialModel(dims, targetPorosity, clusterTarget, anisotropyProfile, optParams)
    if nargin < 5
        optParams = struct();
    end
    if nargin < 4 || isempty(anisotropyProfile)
        anisotropyProfile = ones(1, 3);
    end
    minSeparationFactor = 0.8;
    if isfield(optParams, 'minSeparationFactor') && ~isempty(optParams.minSeparationFactor)
        minSeparationFactor = optParams.minSeparationFactor;
    end
    sizes = clusterTarget.sizeArray(:);
    targetVolume = round(targetPorosity * prod(dims));
    if sum(sizes) > 0
        sizes = max(1, round(sizes * targetVolume / sum(sizes)));
    end
    desiredCount = clusterTarget.numClusters;
    if numel(sizes) > round(desiredCount * 1.1)
        sizes = sizes(1:round(desiredCount * 1.1));
    elseif numel(sizes) < round(desiredCount * 0.9)
        sizes = [sizes; repmat(max(1, round(median(sizes))), round(desiredCount * 0.9) - numel(sizes), 1)];
    end
    radii = nthroot((3 * sizes) / (4 * pi), 3);
    centers = placeClusterCentersPoisson(dims, radii, minSeparationFactor, 400);
    model = false(dims);
    for i = 1:numel(sizes)
        axes = sampleAnisotropicAxes(radii(i), anisotropyProfile);
        model = drawAnisotropicCluster(model, centers(i, :), axes, sizes(i));
    end
    model = smoothInitialModel(model, targetPorosity);
    [model, summary] = enforceInitialClusterConstraints(model, clusterTarget, targetPorosity, optParams);
end
function centers = placeClusterCentersPoisson(dims, radii, minFactor, maxAttempts)
    if nargin < 4
        maxAttempts = 200;
    end
    n = numel(radii);
    centers = zeros(n, 3);
    for i = 1:n
        placed = false;
        attempts = 0;
        while ~placed && attempts < maxAttempts
            attempts = attempts + 1;
            candidate = [randi(dims(1)), randi(dims(2)), randi(dims(3))];
            if i == 1
                centers(i, :) = candidate;
                placed = true;
                break;
            end
            dists = sqrt(sum((centers(1:i-1, :) - candidate).^2, 2));
            required = minFactor * (radii(i) + radii(1:i-1));
            if all(dists(:) >= required(:))
                centers(i, :) = candidate;
                placed = true;
            end
        end
        if ~placed
            centers(i, :) = [randi(dims(1)), randi(dims(2)), randi(dims(3))];
        end
    end
end
function axes = sampleAnisotropicAxes(baseRadius, anisotropyProfile)
    ratios = anisotropyProfile(:)';
    if numel(ratios) < 3
        ratios = [1, 1, 1];
    end
    jitter = 0.25 * (rand(1, 3) - 0.5);
    ratios = max(0.6, ratios + jitter);
    ratios = ratios / nthroot(prod(ratios), 3);
    axes = baseRadius * ratios;
end
function model = drawAnisotropicCluster(model, center, axes, targetVoxels)
    dims = size(model);
    radius = ceil(max(axes) * 1.5);
    [xRange, yRange, zRange] = ndgrid( ...
        max(1, center(1)-radius):min(dims(1), center(1)+radius), ...
        max(1, center(2)-radius):min(dims(2), center(2)+radius), ...
        max(1, center(3)-radius):min(dims(3), center(3)+radius));
    coords = [xRange(:) - center(1), yRange(:) - center(2), zRange(:) - center(3)];
    normR2 = (coords(:,1)./axes(1)).^2 + (coords(:,2)./axes(2)).^2 + (coords(:,3)./axes(3)).^2;
    [~, order] = sort(normR2, 'ascend');
    chosen = order(1:min(targetVoxels, numel(order)));
    linearIdx = sub2ind(dims, xRange(chosen), yRange(chosen), zRange(chosen));
    model(linearIdx) = true;
end
function model = smoothInitialModel(model, targetPorosity)
    if nargin < 2
        targetPorosity = mean(model(:));
    end
    se = strel('sphere', 1);
    model = imopen(imclose(model, se), se);
    model = adjustModelPorosity(model, targetPorosity);
end
function [model, summary] = enforceInitialClusterConstraints(model, clusterTarget, targetPorosity, optParams)
    % 在生成后快速修正初始模型，使其满足簇数、最大簇、孔隙率要求
    tolerancePorosity = 0.002;
    maxIter = 8;
    for iter = 1:maxIter
        summary = extractEfficientClusterFeatures(model);
        porosity = mean(model(:));
        numOk = summary.numClusters >= 0.9 * clusterTarget.numClusters && ...
            summary.numClusters <= 1.1 * clusterTarget.numClusters;
        maxOk = max(summary.sizes) <= 1.1 * clusterTarget.maxSize;
        porosityOk = abs(porosity - targetPorosity) <= tolerancePorosity;
        if numOk && maxOk && porosityOk
            return;
        end
        if summary.numClusters < 0.9 * clusterTarget.numClusters
            deficit = round(0.9 * clusterTarget.numClusters) - summary.numClusters;
            model = addSmallClusters(model, clusterTarget, deficit, optParams);
        elseif summary.numClusters > 1.1 * clusterTarget.numClusters
            excess = summary.numClusters - round(1.1 * clusterTarget.numClusters);
            model = removeSmallestClusters(model, excess);
        end
        if max(summary.sizes) > 1.1 * clusterTarget.maxSize
            model = shrinkLargestCluster(model, clusterTarget.maxSize * 1.05);
        end
        if abs(porosity - targetPorosity) > tolerancePorosity
            model = adjustPorosityTowardsTarget(model, targetPorosity, tolerancePorosity);
        end
    end
    summary = extractEfficientClusterFeatures(model);
end
function model = addSmallClusters(model, clusterTarget, count, optParams)
    sizes = clusterTarget.sizeArray(:);
    medianSize = max(1, round(median(sizes)));
    dims = size(model);
    minSeparationFactor = 0.8;
    if nargin >= 4 && isfield(optParams, 'minSeparationFactor')
        minSeparationFactor = optParams.minSeparationFactor;
    end
    radii = repmat(nthroot((3 * medianSize) / (4 * pi), 3), count, 1);
    centers = placeClusterCentersPoisson(dims, radii, minSeparationFactor, 200);
    for i = 1:count
        axes = sampleAnisotropicAxes(radii(i), [1, 1, 1]);
        model = drawAnisotropicCluster(model, centers(i, :), axes, medianSize);
    end
end
function model = removeSmallestClusters(model, count)
    CC = bwconncomp(model, 26);
    if CC.NumObjects == 0
        return;
    end
    sizes = cellfun(@numel, CC.PixelIdxList);
    [~, order] = sort(sizes, 'ascend');
    removeIdx = order(1:min(count, numel(order)));
    voxels = vertcat(CC.PixelIdxList{removeIdx});
    model(voxels) = false;
end
function model = shrinkLargestCluster(model, maxAllowed)
    CC = bwconncomp(model, 26);
    if CC.NumObjects == 0
        return;
    end
    sizes = cellfun(@numel, CC.PixelIdxList);
    [largestSize, idx] = max(sizes);
    if largestSize <= maxAllowed
        return;
    end
    voxels = CC.PixelIdxList{idx};
    [x, y, z] = ind2sub(size(model), voxels);
    center = [mean(x), mean(y), mean(z)];
    dist2 = (x - center(1)).^2 + (y - center(2)).^2 + (z - center(3)).^2;
    [~, order] = sort(dist2, 'descend');
    removeCount = min(largestSize - round(maxAllowed), numel(order));
    toRemove = voxels(order(1:removeCount));
    model(toRemove) = false;
end
function model = adjustPorosityTowardsTarget(model, targetPorosity, tolerance)
    currentPorosity = mean(model(:));
    maxSteps = 5;
    step = 0;
    while abs(currentPorosity - targetPorosity) > tolerance && step < maxSteps
        model = enforcePorosityHardConstraint(model, targetPorosity, 1);
        currentPorosity = mean(model(:));
        step = step + 1;
    end
end
function model = adjustModelPorosity(model, targetPorosity)
    current = mean(model(:));
    diff = targetPorosity - current;
    if abs(diff) < 1e-4
        return;
    end
    mask = bwperim(model, 26);
    boundaryIdx = find(mask);
    if isempty(boundaryIdx)
        boundaryIdx = 1:numel(model);
    end
    nChange = round(abs(diff) * numel(model));
    nChange = min(nChange, numel(boundaryIdx));
    chosen = boundaryIdx(randperm(numel(boundaryIdx), nChange));
    if diff > 0
        model(chosen) = true;
    else
        model(chosen) = false;
    end
end
function anisotropyProfile = estimateAnisotropyFromFeatures(spatialFeatures, morphologyFeatures)
    anisotropyProfile = [1, 1, 1];
    if isstruct(spatialFeatures) && isfield(spatialFeatures, 'anisotropy') && ...
            ~isempty(spatialFeatures.anisotropy)
        vals = spatialFeatures.anisotropy(:)';
        if numel(vals) >= 3
            anisotropyProfile = vals(1:3);
        end
    elseif isstruct(morphologyFeatures) && isfield(morphologyFeatures, 'elongation') && ...
            ~isempty(morphologyFeatures.elongation)
        vals = morphologyFeatures.elongation(1:min(3, numel(morphologyFeatures.elongation)));
        if numel(vals) == 3
            anisotropyProfile = vals(:)';
        end
    end
    anisotropyProfile = max(anisotropyProfile, eps);
    anisotropyProfile = anisotropyProfile / nthroot(prod(anisotropyProfile), 3);
end
function plan = configureInitialModelParallelism(dims, optParams)
    % 根据问题规模和可用许可配置初始模型并行生成计划
    if nargin < 2
        optParams = struct();
    end
    plan = struct('enableParallel', false, 'numCandidates', 1, ...
        'numWorkers', 0, 'seeds', []);
    problemSize = prod(dims);
    candidateHint = max(3, min(ceil(problemSize / 8e4), 10));
    if candidateHint <= 1
        return;
    end
    try
        hasLicense = license('test', 'Distrib_Computing_Toolbox');
    catch
        hasLicense = false;
    end
    if ~hasLicense
        return;
    end
    useParallel = shouldUseParallel(candidateHint, problemSize);
    if ~useParallel
        return;
    end
    pool = [];
    try
        pool = gcp('nocreate');
    catch
        pool = [];
    end
    if isempty(pool)
        try
            parpool('local');
            pool = gcp('nocreate');
        catch
            pool = [];
        end
    end
    if isempty(pool)
        return;
    end
    plan.enableParallel = true;
    plan.numWorkers = pool.NumWorkers;
    plan.numCandidates = max(3, min(2 * pool.NumWorkers, 12));
    previousState = rng;
    rng('shuffle');
    plan.seeds = randi([1, 1e9], plan.numCandidates, 1);
    rng(previousState);
    if isfield(optParams, 'preserveSmallPores') && optParams.preserveSmallPores
        plan.numCandidates = max(plan.numCandidates, 4);
    end
end
function field = synthesizeDirectionalGaussianField(dims, correlationLength, anisotropy, ...
    spatialFeatures, optParams, randomSeed)
    % 生成包含方向性与梯度的相关高斯场
    if nargin < 6
        randomSeed = [];
    end
    if ~isempty(randomSeed)
        rng(randomSeed, 'twister');
    end
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
function blendedField = blendReferenceDirectionalPatterns(baseField, patternField)
    % 将参考密度/纹理模式与基础场融合（兼容二值簇模型）
    if nargin < 2
        patternField = [];
    end
    if islogical(baseField)
        if isempty(patternField)
            blendedField = baseField;
            return;
        end
        microPattern = imgaussfilt3(patternField, 1);
        threshold = prctile(abs(microPattern(:)), 80);
        boundaryMask = bwperim(baseField, 26);
        growMask = boundaryMask & microPattern > threshold;
        shrinkMask = boundaryMask & microPattern < -threshold;
        blendedField = baseField;
        blendedField(growMask) = true;
        blendedField(shrinkMask) = false;
        return;
    end
    if isempty(patternField)
        blendedField = baseField;
        return;
    end
    normalizedMap = patternField - mean(patternField(:));
    blendedField = baseField + 0.5 * normalizedMap;
    lowFreq = imgaussfilt3(normalizedMap, 3);
    blendedField = blendedField + 0.3 * (lowFreq - mean(lowFreq(:)));
    blendedField = (blendedField - mean(blendedField(:))) / (std(blendedField(:)) + eps);
end
function model = reinforceDensityTargets(model, densityMap, targetPorosity)
    % 根据密度引导图强化局部孔隙分布，确保孔隙率稳定在目标值附近
    if isempty(densityMap)
        return;
    end
    normalized = densityMap - min(densityMap(:));
    normalized = normalized / (max(normalized(:)) + eps);
    totalVoxels = numel(model);
    targetCount = round(totalVoxels * targetPorosity);
    currentCount = nnz(model);
    neighborKernel = ones(3, 3, 3);
    neighborKernel(2, 2, 2) = 0;
    neighborCount = convn(double(model), neighborKernel, 'same');
    additionCandidates = find(~model);
    if ~isempty(additionCandidates)
        boundaryAddMask = neighborCount(additionCandidates) > 0;
        if any(boundaryAddMask)
            additionCandidates = additionCandidates(boundaryAddMask);
        end
    end
    removalCandidates = find(model);
    if ~isempty(removalCandidates)
        maxNeighbors = sum(neighborKernel(:));
        boundaryRemoveMask = neighborCount(removalCandidates) < maxNeighbors;
        if any(boundaryRemoveMask)
            removalCandidates = removalCandidates(boundaryRemoveMask);
        end
    end
    if currentCount < targetCount && ~isempty(additionCandidates)
        deficit = targetCount - currentCount;
        addScores = normalized(additionCandidates) + 0.02 * neighborCount(additionCandidates);
        [~, order] = sort(addScores, 'descend');
        chosen = additionCandidates(order(1:min(deficit, numel(order))));
        model(chosen) = true;
    elseif currentCount > targetCount && ~isempty(removalCandidates)
        surplus = currentCount - targetCount;
        removeScores = normalized(removalCandidates) + 0.02 * neighborCount(removalCandidates);
        [~, order] = sort(removeScores, 'ascend');
        chosen = removalCandidates(order(1:min(surplus, numel(order))));
        model(chosen) = false;
    end
    % 清理过小的孤立孔隙，避免生成噪声
    CC = bwconncomp(model, 26);
    if CC.NumObjects > 0
        for i = 1:CC.NumObjects
            if numel(CC.PixelIdxList{i}) < 3
                model(CC.PixelIdxList{i}) = false;
            end
        end
    end
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
    maxEmbed = max(1, ceil(missing * 0.4));
    nEmbed = min([length(library), missing, maxEmbed]);
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
        if overlapRatio < 0.3
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
function model = calibrateInitialModelToTargets(model, targetSpatial, targetMorph, optParams, densityMap)
    % 使用统一的能量函数对初始模型进行系统校准
    if nargin < 5
        densityMap = [];
    end
    if nargin < 4 || isempty(optParams)
        return;
    end
    weights = getEnergyWeights(optParams, optParams.energyWeights);
    targetMultiScale = [];
    if isfield(optParams, 'multiScaleSpatialFeatures')
        targetMultiScale = optParams.multiScaleSpatialFeatures;
    end
    [currentEnergy, ~, featurePack] = computeUnifiedEnergySnapshot(model, optParams, weights);
    bestFeatures = featurePack;
    maxPasses = 8;
    energyTolerance = 1e-4;
    for pass = 1:maxPasses
        candidateModels = {};
        candidateFeatures = {};
        candidateDescriptions = {};
        % 空间纠偏候选
        spatialCandidate = applySpatialCorrection(model, bestFeatures.spatial, targetSpatial, optParams);
        if ~isequal(spatialCandidate, model)
            [~, ~, candFeat] = computeUnifiedEnergySnapshot(spatialCandidate, optParams, weights);
            candidateModels{end+1} = spatialCandidate; %#ok<AGROW>
            candidateFeatures{end+1} = candFeat; %#ok<AGROW>
            candidateDescriptions{end+1} = 'spatial'; %#ok<AGROW>
        end
        % 形态纠偏候选
        morphCandidate = applyMorphologyCorrection(model, bestFeatures.morphology, targetMorph, optParams);
        if ~isequal(morphCandidate, model)
            [~, ~, candFeat] = computeUnifiedEnergySnapshot(morphCandidate, optParams, weights);
            candidateModels{end+1} = morphCandidate; %#ok<AGROW>
            candidateFeatures{end+1} = candFeat; %#ok<AGROW>
            candidateDescriptions{end+1} = 'morphology'; %#ok<AGROW>
        end
        % 多尺度纠偏候选
        if ~isempty(targetMultiScale)
            multiCandidate = applyMultiScaleCorrection(model, bestFeatures.multiScale, targetMultiScale);
            if ~isequal(multiCandidate, model)
                [~, ~, candFeat] = computeUnifiedEnergySnapshot(multiCandidate, optParams, weights);
                candidateModels{end+1} = multiCandidate; %#ok<AGROW>
                candidateFeatures{end+1} = candFeat; %#ok<AGROW>
                candidateDescriptions{end+1} = 'multiscale'; %#ok<AGROW>
            end
        end
        % 孔隙率微调候选
        porosityCandidate = adjustPorosity(model, optParams.targetPorosity, densityMap);
        if ~isequal(porosityCandidate, model)
            [~, ~, candFeat] = computeUnifiedEnergySnapshot(porosityCandidate, optParams, weights);
            candidateModels{end+1} = porosityCandidate; %#ok<AGROW>
            candidateFeatures{end+1} = candFeat; %#ok<AGROW>
            candidateDescriptions{end+1} = 'porosity'; %#ok<AGROW>
        end
        % 选择能量下降最大的候选
        bestEnergy = currentEnergy;
        bestModel = model;
        bestLabel = '';
        for i = 1:numel(candidateModels)
            candModel = candidateModels{i};
            candFeat = candidateFeatures{i};
            candEnergy = computeUnifiedEnergySnapshot(candModel, optParams, weights, ...
                candFeat.features, candFeat.spatial, candFeat.morphology, candFeat.multiScale);
            if candEnergy < bestEnergy - energyTolerance
                bestEnergy = candEnergy;
                bestModel = candModel;
                bestFeatures = candFeat;
                bestLabel = candidateDescriptions{i};
            end
        end
        if isempty(bestLabel)
            % 没有进一步下降，退出
            break;
        end
        % 应用最佳候选
        model = bestModel;
        currentEnergy = bestEnergy;
        % 结合密度和簇的硬约束
        if ~isempty(densityMap)
            model = reinforceDensityTargets(model, densityMap, optParams.targetPorosity);
        end
        model = enforceClusterSizeConstraints(model, optParams);
        model = enforceTargetClusterCount(model, optParams);
        % 更新特征包以供下一次循环使用
        [~, ~, bestFeatures] = computeUnifiedEnergySnapshot(model, optParams, weights);
    end
end
function model = enforcePorosityClusterTargets(model, optParams, densityMap)
    % 强化孔隙率和簇大小，使其贴近用户设定目标
    if nargin < 3
        densityMap = [];
    end
    if nargin < 2 || isempty(optParams)
        return;
    end
    if isfield(optParams, 'targetPorosity') && ~isempty(optParams.targetPorosity)
        targetPorosity = optParams.targetPorosity;
        for iter = 1:4
            previousPorosity = mean(model(:));
            model = adjustPorosity(model, targetPorosity, densityMap);
            currentPorosity = mean(model(:));
            if abs(currentPorosity - targetPorosity) <= 4e-4
                break;
            end
            if abs(currentPorosity - previousPorosity) < 1e-5
                break;
            end
        end
        if ~isempty(densityMap)
            model = reinforceDensityTargets(model, densityMap, targetPorosity);
        end
    end
    model = enforceClusterSizeConstraints(model, optParams);
    model = enforceTargetClusterCount(model, optParams);
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
            % 低球形度：使用三维各向异性结构元素
            seX = createAnisotropicStructuringElement([1, 0, 0], [3, 1, 1]);
            seY = createAnisotropicStructuringElement([0, 1, 0], [3, 1, 1]);
            seZ = createAnisotropicStructuringElement([0, 0, 1], [1, 1, 3]);
            model = imopen(model, seX);
            model = imopen(model, seY);
            model = imopen(model, seZ);
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
    % 处理极端大小的簇（基于软指导值）
    [targetMin, targetMax] = getAdaptiveClusterBounds(optParams);
    if isempty(targetMax) || targetMax <= 0
        targetMax = prctile(double(sizes), 95);
    end
    if isempty(targetMin) || targetMin <= 0
        targetMin = prctile(double(sizes), 10);
    end
    maxAllowed = max(targetMax * 1.3, targetMax + max(15, round(0.1 * targetMax)));
    minAllowed = max(5, floor(targetMin * 0.6));
    volumeSize = size(model);
    % 缩减过大的簇
    largeClusters = find(sizes > maxAllowed);
    if ~isempty(largeClusters)
        if ~isempty(targetMax)
            targetSize = max(1, round(targetMax));
        else
            targetSize = max(1, round(maxAllowed));
        end
        origPixelLists = CC.PixelIdxList(largeClusters);
        refinedPixelLists = cell(numel(largeClusters), 1);
        useParallel = shouldUseParallel(numel(largeClusters), numel(model));
        if useParallel
            parfor i = 1:numel(largeClusters)
                refinedPixelLists{i} = reduceOversizedClusterIndices( ...
                    origPixelLists{i}, volumeSize, targetSize);
            end
        else
            for i = 1:numel(largeClusters)
                refinedPixelLists{i} = reduceOversizedClusterIndices( ...
                    origPixelLists{i}, volumeSize, targetSize);
            end
        end
        originalSizes = cellfun(@numel, origPixelLists);
        refinedSizes = cellfun(@numel, refinedPixelLists);
        reductionRates = 1 - refinedSizes ./ max(originalSizes, 1);
        reductionRates(~isfinite(reductionRates)) = 0;
        avgReduction = mean(reductionRates);
        fprintf('    批量调整%d个大簇尺寸 (平均缩减率%.1f%%) ...\n', ...
            numel(largeClusters), avgReduction * 100);
        for i = 1:numel(largeClusters)
            model(origPixelLists{i}) = false;
        end
        for i = 1:numel(largeClusters)
            model(refinedPixelLists{i}) = true;
        end
    end
    % 处理过小的簇
    tinyClusters = find(sizes < minAllowed);
    if ~isempty(tinyClusters)
        fprintf('    移除或扩展%d个过小的簇...\n', length(tinyClusters));
        pixelLists = CC.PixelIdxList(tinyClusters);
        if isfield(optParams, 'preserveSmallPores') && optParams.preserveSmallPores
            grownPixelLists = cell(numel(tinyClusters), 1);
            useParallel = shouldUseParallel(numel(tinyClusters), numel(model));
            if useParallel
                parfor i = 1:numel(tinyClusters)
                    grownPixelLists{i} = growSmallClusterIndices( ...
                        pixelLists{i}, volumeSize, max(minAllowed, numel(pixelLists{i}) * 2));
                end
            else
                for i = 1:numel(tinyClusters)
                    grownPixelLists{i} = growSmallClusterIndices( ...
                        pixelLists{i}, volumeSize, max(minAllowed, numel(pixelLists{i}) * 2));
                end
            end
            for i = 1:numel(tinyClusters)
                model(pixelLists{i}) = false;
            end
            for i = 1:numel(tinyClusters)
                model(grownPixelLists{i}) = true;
            end
        else
            flatIdx = cellfun(@(idx) idx(:), pixelLists, 'UniformOutput', false);
            flatIdx = vertcat(flatIdx{:});
            model(flatIdx) = false;
        end
    end
end
function refinedIdx = reduceOversizedClusterIndices(clusterIdx, volumeSize, targetSize)
    % 计算缩减后的簇索引
    currentSize = numel(clusterIdx);
    if currentSize <= targetSize
        refinedIdx = clusterIdx;
        return;
    end
    pad = 2;
    [x, y, z] = ind2sub(volumeSize, clusterIdx);
    minX = max(min(x) - pad, 1);
    maxX = min(max(x) + pad, volumeSize(1));
    minY = max(min(y) - pad, 1);
    maxY = min(max(y) + pad, volumeSize(2));
    minZ = max(min(z) - pad, 1);
    maxZ = min(max(z) + pad, volumeSize(3));
    localSize = [maxX - minX + 1, maxY - minY + 1, maxZ - minZ + 1];
    localMask = false(localSize);
    localX = x - minX + 1;
    localY = y - minY + 1;
    localZ = z - minZ + 1;
    localIdx = sub2ind(localSize, localX, localY, localZ);
    localMask(localIdx) = true;
    distMap = bwdist(~localMask);
    reductionRatio = min(0.15, max(0, (currentSize - targetSize) / currentSize));
    if reductionRatio <= 0
        refinedIdx = clusterIdx;
        return;
    end
    distances = distMap(localMask);
    if isempty(distances) || all(distances == 0)
        refinedIdx = clusterIdx;
        return;
    end
    cutoff = quantile(distances, reductionRatio);
    removalMask = localMask & (distMap <= cutoff);
    refinedLocal = localMask;
    refinedLocal(removalMask) = false;
    se = strel('sphere', 1);
    refinedLocal = imclose(refinedLocal, se);
    targetKeep = max(targetSize, round(currentSize * (1 - reductionRatio)));
    localCC = bwconncomp(refinedLocal, 26);
    if localCC.NumObjects > 1 || ~any(refinedLocal(:))
        refinedLocal = localMask;
        while sum(refinedLocal(:)) > targetKeep
            candidate = imerode(refinedLocal, se);
            if ~any(candidate(:))
                break;
            end
            ccCandidate = bwconncomp(candidate, 26);
            if ccCandidate.NumObjects > 1
                break;
            end
            refinedLocal = candidate;
        end
    end
    if sum(refinedLocal(:)) < targetKeep * 0.85
        refinedLocal = localMask;
    end
    [rx, ry, rz] = ind2sub(localSize, find(refinedLocal));
    rx = rx + minX - 1;
    ry = ry + minY - 1;
    rz = rz + minZ - 1;
    refinedIdx = sub2ind(volumeSize, rx, ry, rz);
end
function model = reduceOversizedCluster(model, clusterIdx, optParams)
    % 通过平滑收缩的方式减少大簇体量
    [~, upperBound] = getAdaptiveClusterBounds(optParams);
    if isempty(upperBound) || upperBound <= 0
        targetSize = max(1, round(numel(clusterIdx) * 0.7));
    else
        targetSize = max(1, round(upperBound));
    end
    refinedIdx = reduceOversizedClusterIndices(clusterIdx, size(model), targetSize);
    model(clusterIdx) = false;
    model(refinedIdx) = true;
end
function grownIdx = growSmallClusterIndices(clusterIdx, volumeSize, targetSize)
    % 计算扩展后的小簇索引，尽量逼近目标体量
    if nargin < 3 || isempty(targetSize) || ~isfinite(targetSize)
        targetSize = numel(clusterIdx) * 2;
    end
    targetSize = max(numel(clusterIdx), round(min(targetSize, prod(volumeSize))));
    mask = false(volumeSize);
    mask(clusterIdx) = true;
    grownMask = mask;
    se = strel('sphere', 1);
    maxIterations = 8;
    iter = 0;
    while sum(grownMask(:)) < targetSize && iter < maxIterations
        grownMask = imdilate(grownMask, se);
        iter = iter + 1;
        if all(grownMask(:))
            break;
        end
    end
    currentCount = sum(grownMask(:));
    if currentCount < targetSize
        % 通过距离场补充最近的体素，避免产生孤立孔隙
        outsideMask = ~grownMask;
        if any(outsideMask(:))
            distanceMap = bwdist(grownMask);
            outsideIdx = find(outsideMask);
            distances = distanceMap(outsideIdx);
            [~, order] = sort(distances, 'ascend');
            deficit = min(targetSize - currentCount, numel(order));
            if deficit > 0
                selected = outsideIdx(order(1:deficit));
                grownMask(selected) = true;
            end
        end
    end
    grownIdx = find(grownMask);
end
function model = growSmallCluster(model, clusterIdx)
    % 通过局部膨胀扩展小簇，避免直接删除
    grownIdx = growSmallClusterIndices(clusterIdx, size(model));
    model(clusterIdx) = false;
    model(grownIdx) = true;
end
function model = applySpatialCorrection(model, currentSpatial, targetSpatial, optParams)
    % 基于能量函数的空间修正，避免随机翻转造成的伪结构
    if nargin < 4
        optParams = struct();
    end
    weights = getEnergyWeights(optParams, getDefaultEnergyWeights());
    [baselineEnergy, ~, ~] = computeUnifiedEnergySnapshot(model, optParams, weights, [], currentSpatial);
    modelSize = size(model);
    targetDensityMap = computeTargetDensityMap(targetSpatial, optParams, modelSize);
    kernel = ones(5, 5, 5, 'double');
    kernel = kernel / sum(kernel(:));
    localDensity = convn(double(model), kernel, 'same');
    densityError = localDensity - targetDensityMap;
    boundaryMask = bwperim(model, 26);
    surfaceBand = imdilate(boundaryMask, strel('sphere', 2));
    addCandidates = find(surfaceBand & ~model);
    removeCandidates = find(surfaceBand & model);
    if isempty(addCandidates) && isempty(removeCandidates)
        return;
    end
    maxCandidates = 5;
    addScores = [];
    removeScores = [];
    if ~isempty(addCandidates)
        addScores = densityError(addCandidates);
        [~, order] = sort(addScores, 'ascend');
        addCandidates = addCandidates(order(1:min(maxCandidates, numel(order))));
    end
    if ~isempty(removeCandidates)
        removeScores = densityError(removeCandidates);
        [~, order] = sort(removeScores, 'descend');
        removeCandidates = removeCandidates(order(1:min(maxCandidates, numel(order))));
    end
    bestEnergy = baselineEnergy;
    bestModel = model;
    tolerance = 1e-4;
    for idx = addCandidates(:)'
        move = createSpatialAdjustmentMove(model, idx, 'add', densityError, optParams);
        move = enforceTopologyAwareMove(move, model);
        if isempty(move.linearIdx)
            continue;
        end
        candidateModel = applyMoveToModel(model, move);
        candidateEnergy = computeUnifiedEnergySnapshot(candidateModel, optParams, weights);
        if candidateEnergy < bestEnergy - tolerance
            bestEnergy = candidateEnergy;
            bestModel = candidateModel;
        end
    end
    for idx = removeCandidates(:)'
        move = createSpatialAdjustmentMove(model, idx, 'remove', densityError, optParams);
        move = enforceTopologyAwareMove(move, model);
        if isempty(move.linearIdx)
            continue;
        end
        candidateModel = applyMoveToModel(model, move);
        candidateEnergy = computeUnifiedEnergySnapshot(candidateModel, optParams, weights);
        if candidateEnergy < bestEnergy - tolerance
            bestEnergy = candidateEnergy;
            bestModel = candidateModel;
        end
    end
    model = bestModel;
end
function model = applyMorphologyCorrection(model, currentMorph, targetMorph, optParams)
    % 应用形态学修正
    % 根据当前和目标形态特征的差异进行调整
    if nargin < 4
        optParams = struct();
    end
    smallSphere = strel('sphere', 1);
    % 球形度修正
    if ~isempty(currentMorph.sphericity) && ~isempty(targetMorph.sphericity)
        currentSpher = mean(currentMorph.sphericity);
        targetSpher = mean(targetMorph.sphericity);
        if abs(currentSpher - targetSpher) > 0.1
            if targetSpher > currentSpher
                % 需要增加球形度 - 使用形态学闭操作
                model = imclose(imopen(model, smallSphere), smallSphere);
            else
                % 需要减少球形度 - 使用三维各向异性腐蚀
                seX = createAnisotropicStructuringElement([1, 0, 0], [2, 1, 1]);
                seY = createAnisotropicStructuringElement([0, 1, 0], [2, 1, 1]);
                seZ = createAnisotropicStructuringElement([0, 0, 1], [1, 1, 2]);
                model = imerode(model, seX);
                model = imerode(model, seY);
                model = imerode(model, seZ);
                model = imdilate(model, smallSphere);
            end
        end
    end
    % 伸长率修正
    if ~isempty(currentMorph.elongation) && ~isempty(targetMorph.elongation)
        currentElong = mean(currentMorph.elongation);
        targetElong = mean(targetMorph.elongation);
        if abs(currentElong - targetElong) > 0.5
            if targetElong > currentElong
                % 需要增加伸长率 - 使用三维定向膨胀并保持体积
                axes = {[1, 0, 0], [0, 1, 0], [0, 0, 1]};
                for a = 1:numel(axes)
                    se = createAnisotropicStructuringElement(axes{a}, [3, 1, 1]);
                    model = imdilate(model, se);
                end
                model = imerode(model, smallSphere);
            else
                % 需要减少伸长率 - 各向同性收缩
                se = strel('sphere', 2);
                model = imopen(imclose(model, se), smallSphere);
            end
        end
    end
end
function model = adjustPorosity(model, targetPorosity, densityMap)
    % 调整孔隙率到目标值，支持密度引导并提供精细兜底
    if nargin < 3
        densityMap = [];
    end
    if isempty(targetPorosity) || ~isfinite(targetPorosity)
        return;
    end
    totalVoxels = numel(model);
    tolerance = 1e-4;
    maxIterations = 8;
    neighborKernel = ones(3, 3, 3);
    neighborKernel(2, 2, 2) = 0;
    % 分阶段迭代逼近目标孔隙率
    for iter = 1:maxIterations
        currentPorosity = mean(model(:));
        diff = targetPorosity - currentPorosity;
        if abs(diff) <= tolerance
            break;
        end
        stepFraction = min(0.02, max(0.0005, abs(diff)));
        voxelsToAdjust = max(1, round(stepFraction * totalVoxels));
        neighborCount = convn(double(model), neighborKernel, 'same');
        if diff > 0
            boundaryMask = imdilate(model, ones(3, 3, 3)) & ~model;
            candidateIdx = find(boundaryMask);
            if isempty(candidateIdx)
                candidateIdx = find(~model);
            end
            if isempty(candidateIdx)
                break;
            end
            scores = normalizeScores(neighborCount(candidateIdx));
            if ~isempty(densityMap)
                densityVals = normalizeScores(densityMap(candidateIdx));
                scores = 0.6 * scores + 0.4 * densityVals;
            end
            scores = scores + 0.05 * rand(size(scores));
            [~, order] = sort(scores, 'descend');
            selectionCount = min(voxelsToAdjust, numel(order));
            chosen = candidateIdx(order(1:selectionCount));
            model(chosen) = true;
        else
            boundaryMask = model & (neighborCount < sum(neighborKernel(:)));
            candidateIdx = find(boundaryMask);
            if isempty(candidateIdx)
                candidateIdx = find(model);
            end
            if isempty(candidateIdx)
                break;
            end
            scores = normalizeScores(neighborCount(candidateIdx));
            if ~isempty(densityMap)
                densityVals = normalizeScores(densityMap(candidateIdx));
                scores = 0.6 * scores + 0.4 * (1 - densityVals);
            end
            scores = scores + 0.05 * rand(size(scores));
            [~, order] = sort(scores, 'ascend');
            selectionCount = min(voxelsToAdjust, numel(order));
            chosen = candidateIdx(order(1:selectionCount));
            model(chosen) = false;
        end
    end
    % 精细兜底：直接翻转最合适的体素以达到目标孔隙率
    finalDiff = targetPorosity - mean(model(:));
    if abs(finalDiff) > tolerance
        neighborCount = convn(double(model), neighborKernel, 'same');
        voxelsToAdjust = max(1, round(abs(finalDiff) * totalVoxels));
        if finalDiff > 0
            candidateIdx = find(~model);
            if isempty(candidateIdx)
                return;
            end
            scores = normalizeScores(neighborCount(candidateIdx));
            if ~isempty(densityMap)
                scores = 0.5 * scores + 0.5 * normalizeScores(densityMap(candidateIdx));
            end
            [~, order] = sort(scores, 'descend');
            chosen = candidateIdx(order(1:min(voxelsToAdjust, numel(order))));
            model(chosen) = true;
        else
            candidateIdx = find(model);
            if isempty(candidateIdx)
                return;
            end
            scores = normalizeScores(neighborCount(candidateIdx));
            if ~isempty(densityMap)
                scores = 0.5 * scores + 0.5 * (1 - normalizeScores(densityMap(candidateIdx)));
            end
            [~, order] = sort(scores, 'ascend');
            chosen = candidateIdx(order(1:min(voxelsToAdjust, numel(order))));
            model(chosen) = false;
        end
    end
end
function values = normalizeScores(values)
    if isempty(values)
        return;
    end
    values = double(values);
    vmin = min(values);
    vmax = max(values);
    if vmax > vmin
        values = (values - vmin) ./ (vmax - vmin + eps);
    else
        values = zeros(size(values));
    end
end
function model = refineInitialModelWithSpatialMCMC(model, targetSpatial, targetMorph, targetPorosity, optParams)
    % 使用MCMC在初始化阶段细化模型，使空间特征与原始模型更匹配
    maxIterations = 200;
    temperature = 0.5;
    coolingRate = 0.97;
    bestModel = model;
    currentEnergy = computeInitialMCMCEnergy(model, targetSpatial, targetMorph, targetPorosity, optParams);
    bestEnergy = currentEnergy;
    for iter = 1:maxIterations
        move = proposeClusterNeighborhoodMove(model, targetPorosity, optParams);
        if isempty(move.linearIdx)
            continue;
        end
        candidateModel = applyClusterNeighborhoodMove(model, move);
        candidateEnergy = computeInitialMCMCEnergy(candidateModel, targetSpatial, targetMorph, targetPorosity, optParams);
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
function energy = computeInitialMCMCEnergy(model, targetSpatial, targetMorph, targetPorosity, optParams)
    % 计算初始化阶段MCMC的能量函数，强调与统一能量一致
    if nargin < 5 || isempty(optParams)
        optParams = struct();
    end
    tempOpt = optParams;
    tempOpt.spatialFeatures = targetSpatial;
    tempOpt.morphologyFeatures = targetMorph;
    tempOpt.targetPorosity = targetPorosity;
    if ~isfield(tempOpt, 'multiScaleSpatialFeatures') || isempty(tempOpt.multiScaleSpatialFeatures)
        tempOpt.multiScaleSpatialFeatures = optParams.multiScaleSpatialFeatures;
    end
    weights = getEnergyWeights(tempOpt, tempOpt.energyWeights);
    energy = computeUnifiedEnergySnapshot(model, tempOpt, weights);
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
function mcmcState = initializeComprehensiveMCMCState(model, optParams, weights, parallelHint)
    % 初始化综合MCMC状态
    if nargin < 4
        parallelHint = [];
    end
    mcmcState = struct();
    mcmcState.model = model;
    mcmcState.bestModel = model;
    mcmcState.boundaryUpdated = false;
    mcmcState.boundaryContext = struct();
    mcmcState.temperature = 0.4;
    % 存储优化参数和权重
    mcmcState.optParams = optParams;
    mcmcState.weights = weights;
    % 计算初始特征与能量（支持并行）
    mcmcState = updateAllFeatures(mcmcState, parallelHint);
    mcmcState.bestEnergy = mcmcState.currentEnergy;
    % 初始化匹配度跟踪
    mcmcState.currentMorphMatch = calculateMorphologyMatch(mcmcState.morphologyFeatures, ...
        mcmcState.optParams.morphologyFeatures);
    mcmcState.currentSpatialMatch = calculateSpatialMatch(mcmcState.spatialFeatures, ...
        mcmcState.optParams.spatialFeatures);
    mcmcState.bestMorphMatch = mcmcState.currentMorphMatch;
    mcmcState.bestSpatialMatch = mcmcState.currentSpatialMatch;
    mcmcState.bestMatchModel = model;
    mcmcState.bestShapeModel = model;
    mcmcState.bestShapeScore = computeCompositeShapeScore(mcmcState);
    % 允许配置回退间隔（保持默认500步，但可在 config 中关闭）
    mcmcState.checkpointInterval = 500;
    mcmcState.lastCheckpointIter = 0;
    mcmcState.checkpointBaselineMorph = mcmcState.bestMorphMatch;
    mcmcState.checkpointBaselineSpatial = mcmcState.bestSpatialMatch;
    % 性能跟踪
    mcmcState.acceptanceRate = 0.3;
    mcmcState.moveHistory = zeros(1, 10);
    mcmcState.energyTrend = zeros(1, 10);
    
    % 优化历史
    mcmcState.optimizationHistory = struct();
    mcmcState.optimizationHistory.morphologyMatches = [];
    mcmcState.optimizationHistory.spatialMatches = [];
    mcmcState.optimizationHistory.iterations = [];
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
            currentWeights.morphology = baseWeights.morphology * 2.1;
            currentWeights.shapePreservation = baseWeights.shapePreservation * 1.9;
            currentWeights.directOverlap = baseWeights.directOverlap * 1.3; % 在早期直接向原始形态靠拢
            currentWeights.spatial = baseWeights.spatial * 0.6;
            currentWeights.multiScale = baseWeights.multiScale * 0.6;
            currentWeights.cluster = baseWeights.cluster * 0.85; % 降低大尺度簇操作权重
        case 'spatial'
            % 空间优化阶段
            currentWeights.spatial = baseWeights.spatial * 1.5;
            currentWeights.multiScale = baseWeights.multiScale * 1.5;
            currentWeights.morphology = baseWeights.morphology * 0.7;
            currentWeights.shapePreservation = baseWeights.shapePreservation * 0.7;
            currentWeights.directOverlap = baseWeights.directOverlap * 1.1; % 仍保持一定位置一致性
        case 'balanced'
            % 平衡阶段，使用原始权重
            % 可以稍微增强连通性和结构一致性
            currentWeights.connectivity = baseWeights.connectivity * 1.2;
            currentWeights.structureCoherence = baseWeights.structureCoherence * 1.2;
            currentWeights.directOverlap = baseWeights.directOverlap; % 回归默认
    end
end
%% ========== 能量计算函数 ==========
function energy = calculateComprehensiveEnergy(mcmcState, parallelHint) %#ok<INUSD>
    % 计算综合能量函数（保持接口以兼容旧调用）
    weights = mcmcState.weights;
    energy = computeUnifiedEnergySnapshot(mcmcState.model, ...
        mcmcState.optParams, weights, mcmcState.features, ...
        mcmcState.spatialFeatures, mcmcState.morphologyFeatures, ...
        mcmcState.multiScaleSpatialFeatures);
end
function energy = calculatePorosityEnergy(clusterFeatures, optParams)
    % 孔隙率能量仅依赖当前孔隙率与 targetPorosity 的偏差
    if isempty(optParams) || ~isfield(optParams, 'targetPorosity')
        energy = 0;
        return;
    end
    targetPorosity = optParams.targetPorosity;
    if isempty(targetPorosity) || ~isfinite(targetPorosity)
        energy = 0;
        return;
    end
    modelSize = optParams.modelSize;
    if isfield(clusterFeatures, 'totalVoxels') && clusterFeatures.totalVoxels > 0
        totalVoxels = clusterFeatures.totalVoxels;
    elseif isfield(clusterFeatures, 'sizes') && ~isempty(clusterFeatures.sizes)
        totalVoxels = sum(clusterFeatures.sizes);
    else
        totalVoxels = 0;
    end
    currentPorosity = totalVoxels / prod(modelSize);
    diff = currentPorosity - targetPorosity;
    energy = diff.^2 / max(targetPorosity * (1 - targetPorosity), eps);
end
function model = enforcePorosityHardConstraint(model, targetPorosity, maxIters)
    % 在体素精度上强制满足目标孔隙率，优先调整边界体素
    if nargin < 3
        maxIters = 20;
    end
    targetVoxels = round(targetPorosity * numel(model));
    iter = 0;
    while iter < maxIters
        currentVoxels = nnz(model);
        diff = targetVoxels - currentVoxels;
        if diff == 0
            break;
        end
        context = computeBoundaryMasks(model);
        if diff > 0
            candidates = context.solidIdx;
            addCount = min(abs(diff), numel(candidates));
            if addCount == 0
                break;
            end
            chosen = selectLeastDisruptiveVoxels(model, candidates, addCount, true);
            model(chosen) = true;
        else
            candidates = context.poreIdx;
            removeCount = min(abs(diff), numel(candidates));
            if removeCount == 0
                break;
            end
            chosen = selectLeastDisruptiveVoxels(model, candidates, removeCount, false);
            model(chosen) = false;
        end
        iter = iter + 1;
    end
end
function chosen = selectLeastDisruptiveVoxels(model, candidates, count, toPore)
    kernel = zeros(3,3,3);
    kernel(2,2,1) = 1; kernel(2,2,3) = 1;
    kernel(2,1,2) = 1; kernel(2,3,2) = 1;
    kernel(1,2,2) = 1; kernel(3,2,2) = 1;
    neighborPore = convn(double(model), kernel, 'same');
    scores = neighborPore(candidates);
    if toPore
        % 优先选择周围孔隙更少的固体体素，避免大幅扩大簇
        [~, order] = sort(scores, 'ascend');
    else
        % 优先收缩外缘体素
        [~, order] = sort(scores, 'descend');
    end
    order = order(1:min(count, numel(order)));
    chosen = candidates(order);
end
function E_cluster = calculateClusterEnergy(features, optParams)
    % 计算簇能量：显式使用自适应目标，抑制超级大簇
    if features.numClusters == 0
        E_cluster = 5.0;
        return;
    end
    clusterTarget = [];
    if isfield(optParams, 'clusterTarget')
        clusterTarget = optParams.clusterTarget;
    end
    sizes = double(features.sizes(:));
    if isempty(sizes)
        E_cluster = 3.0;
        return;
    end
    if isempty(clusterTarget) || ~isfield(clusterTarget, 'numClusters')
        targetCount = features.numClusters;
        targetHistEdges = linspace(min(sizes), max(sizes), 10);
        targetHist = histcounts(sizes, targetHistEdges, 'Normalization', 'probability');
        clusterTarget = struct('numClusters', targetCount, 'histEdges', targetHistEdges, ...
            'histogram', targetHist, 'maxSizeTarget', max(sizes), 'meanSize', mean(sizes), ...
            'sizeArray', sizes);
    end
    N_target = max(clusterTarget.numClusters, 1);
    Ndiff = ((features.numClusters - N_target) / N_target).^2;
    sizeMean = mean(sizes);
    targetMean = max(clusterTarget.meanSize, 1);
    meanDiff = ((sizeMean - targetMean) / targetMean).^2;
    targetVar = var(double(clusterTarget.sizeArray));
    currentVar = var(sizes);
    varDiff = ((currentVar - targetVar) / max(targetVar + 1, 1)).^2;
    currentHist = histcounts(sizes, clusterTarget.histEdges, 'Normalization', 'probability');
    if numel(currentHist) ~= numel(clusterTarget.histogram)
        currentHist = interp1(linspace(0,1,numel(currentHist)), currentHist, ...
            linspace(0,1,numel(clusterTarget.histogram)), 'linear', 'extrap');
        currentHist(isnan(currentHist)) = 0;
    end
    histDiff = sum((currentHist - clusterTarget.histogram).^2);
    targetMax = max(clusterTarget.maxSize, 1);
    exceedRatio = max(0, max(sizes) / (1.1 * targetMax) - 1);
    maxPenalty = exceedRatio.^2;
    E_cluster = 0.35 * Ndiff + 0.2 * meanDiff + 0.15 * varDiff + 0.15 * histDiff + 0.15 * maxPenalty * 8;
end
function metrics = computeIslandMetrics(model)
    % 统计孔隙/固体的小岛数量与体积（使用连通域）
    threshold = 10;
    CCpore = bwconncomp(model, 26);
    CCsolid = bwconncomp(~model, 26);
    poreSizes = cellfun(@numel, CCpore.PixelIdxList);
    solidSizes = cellfun(@numel, CCsolid.PixelIdxList);
    poreIslands = poreSizes(poreSizes < threshold);
    solidIslands = solidSizes(solidSizes < threshold);
    metrics = struct();
    metrics.numPoreIslands = numel(poreIslands);
    metrics.numSolidIslands = numel(solidIslands);
    metrics.poreIslandVolume = sum(poreIslands);
    metrics.solidIslandVolume = sum(solidIslands);
end
function E_island = calculateIslandEnergy(model)
    metrics = computeIslandMetrics(model);
    volume = numel(model);
    smallIslandPenalty = (metrics.numPoreIslands + metrics.numSolidIslands) / max(volume, 1);
    volumePenalty = (metrics.poreIslandVolume + metrics.solidIslandVolume) / max(volume, 1);
    E_island = 8 * smallIslandPenalty + 2 * volumePenalty;
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
    % 计算连通性能量：突出欧拉数与最大连通体比例的双重偏差
    targetConn = optParams.spatialFeatures.connectivity;
    if ~isfield(spatialFeatures, 'connectivity')
        E_connectivity = 2.0;
        return;
    end
    currentConn = spatialFeatures.connectivity;
    if ~isfield(currentConn, 'eulerNumber') || ~isfield(targetConn, 'eulerNumber')
        E_connectivity = 2.0;
        return;
    end
    eulerDiff = ((currentConn.eulerNumber - targetConn.eulerNumber) / ...
        max(abs(targetConn.eulerNumber), 1)) ^ 2;
    currentLmax = 0; targetLmax = 0;
    if isfield(currentConn, 'largestComponentRatio')
        currentLmax = currentConn.largestComponentRatio;
    end
    if isfield(targetConn, 'largestComponentRatio')
        targetLmax = targetConn.largestComponentRatio;
    end
    lmaxDiff = ((currentLmax - targetLmax) / max(targetLmax, 1e-3)) ^ 2;
    wE = 0.5; wL = 0.5;
    E_connectivity = wE * eulerDiff + wL * lmaxDiff;
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
function E_direct = calculateDirectOverlapEnergy(model, optParams)
    % 直接重叠能量：鼓励生成模型在空间上与原始模型更一致
    if ~isfield(optParams, 'referenceModel') || isempty(optParams.referenceModel) || ...
            ~isequal(size(optParams.referenceModel), size(model))
        E_direct = 0; % 无参考时忽略该项
        return;
    end
    refModel = logical(optParams.referenceModel);
    binModel = logical(model);
    overlap = mean(binModel(:) == refModel(:));
    % 直接以重叠比例转换为能量，避免重复惩罚
    E_direct = 1 - overlap;
end
function weights = getDefaultEnergyWeights()
    % 提供一个统一的默认能量权重集合，确保各阶段评估一致
    weights = struct();
    weights.cluster = 1.6;
    weights.porosity = 4.0;
    weights.morphology = 11.0;
    weights.spatial = 9.0;
    weights.connectivity = 7.0;
    weights.multiScale = 3.0;
    weights.shapePreservation = 5.0;
    weights.structureCoherence = 4.0;
    weights.island = 8.0;
    weights.directOverlap = 3.5; % 与原始模型直接重叠项（提高权重以限制大幅偏移）
end
function weights = initializeNormalizedEnergyWeights(components, targetRatios)
    % 根据初始能量值自动归一化权重，使各分量贡献接近目标比例
    baseWeights = getDefaultEnergyWeights();
    weights = baseWeights;
    fn = fieldnames(baseWeights);
    epsVal = 1e-6;
    for i = 1:numel(fn)
        field = fn{i};
        if isfield(components, field)
            compVal = max(components.(field), epsVal);
        else
            compVal = 1.0;
        end
        if isfield(targetRatios, field)
            targetRatio = targetRatios.(field);
        else
            targetRatio = 0.02; % 未指定时给出极小占比
        end
        weights.(field) = targetRatio / compVal;
    end
    % 调整尺度，使孔隙率权重处于强化区间
    scaleFactor = 25.0 / max(weights.porosity, epsVal);
    weights = structfun(@(v) v * scaleFactor, weights, 'UniformOutput', false);
    weights = cell2struct(struct2cell(weights), fn, 1);
    % 限定孔隙率权重在 [20, 40] 之间
    clampFactor = min(max(20 / weights.porosity, 1), 40 / weights.porosity);
    weights = structfun(@(v) v * clampFactor, weights, 'UniformOutput', false);
    weights = cell2struct(struct2cell(weights), fn, 1);
end
function weights = getEnergyWeights(optParams, defaultWeights)
    % 获取能量权重，优先使用optParams中的缓存
    if nargin < 2 || isempty(defaultWeights)
        defaultWeights = getDefaultEnergyWeights();
    end
    weights = defaultWeights;
    if nargin >= 1 && isstruct(optParams) && isfield(optParams, 'energyWeights') && ...
            ~isempty(optParams.energyWeights)
        userWeights = optParams.energyWeights;
        fns = fieldnames(defaultWeights);
        for i = 1:numel(fns)
            fn = fns{i};
            if isfield(userWeights, fn) && ~isempty(userWeights.(fn))
                weights.(fn) = userWeights.(fn);
            end
        end
    end
end
function [energy, components, featurePack] = computeUnifiedEnergySnapshot(model, optParams, ...
        weights, clusterFeatures, spatialFeatures, morphologyFeatures, multiScaleFeatures)
    % 计算统一的总能量，用于初始化与MCMC阶段的对齐
    if nargin < 3 || isempty(weights)
        weights = getEnergyWeights(optParams);
    end
    if nargin < 4 || isempty(clusterFeatures)
        clusterFeatures = extractEfficientClusterFeatures(model);
    end
    if nargin < 5 || isempty(spatialFeatures)
        spatialFeatures = computeEnhancedSpatialFeatures(model);
    end
    if nargin < 6 || isempty(morphologyFeatures)
        morphologyFeatures = computeDetailedMorphologyFeatures(model);
    end
    if nargin < 7 || isempty(multiScaleFeatures)
        multiScaleFeatures = computeMultiScaleSpatialFeatures(model);
    end
    components = evaluateEnergyComponents(model, clusterFeatures, spatialFeatures, ...
        morphologyFeatures, multiScaleFeatures, optParams);
    energy = weights.porosity * components.porosity + ...
        weights.cluster * components.cluster + ...
        weights.morphology * components.morphology + ...
        weights.spatial * components.spatial + ...
        weights.connectivity * components.connectivity + ...
        weights.multiScale * components.multiScale + ...
        weights.shapePreservation * components.shapePreservation + ...
        weights.structureCoherence * components.structureCoherence + ...
        weights.island * components.island + ...
        weights.directOverlap * components.directOverlap; % 新增直接重叠项，鼓励与原始模型空间位置一致
    if nargout > 2
        featurePack = struct();
        featurePack.features = clusterFeatures;
        featurePack.spatial = spatialFeatures;
        featurePack.morphology = morphologyFeatures;
        featurePack.multiScale = multiScaleFeatures;
    end
end
function components = evaluateEnergyComponents(model, clusterFeatures, spatialFeatures, ...
        morphologyFeatures, multiScaleFeatures, optParams)
    % 将所有能量分量归拢到统一的结构中
    if nargin < 6 || isempty(optParams)
        optParams = struct();
    end
    components = struct();
    components.porosity = calculatePorosityEnergy(clusterFeatures, optParams);
    components.cluster = calculateClusterEnergy(clusterFeatures, optParams);
    components.morphology = calculateMorphologyEnergy(morphologyFeatures, optParams);
    components.spatial = calculateSpatialEnergy(spatialFeatures, optParams);
    components.connectivity = calculateConnectivityEnergy(spatialFeatures, optParams);
    components.multiScale = calculateMultiScaleEnergy(multiScaleFeatures, optParams);
    components.shapePreservation = calculateShapePreservationEnergy(clusterFeatures, ...
        morphologyFeatures, optParams);
    components.structureCoherence = calculateStructureCoherenceEnergy(struct('model', model));
    components.island = calculateIslandEnergy(model);
    components.directOverlap = calculateDirectOverlapEnergy(model, optParams); % 直接重叠/差异项
end
%% ========== 特征更新和约束函数 ==========
function mcmcState = updateAllFeatures(mcmcState, parallelHint)
    % 更新所有特征，支持并行特征提取与能量评估
    if nargin < 2
        parallelHint = [];
    end
    model = mcmcState.model;
    if isempty(model)
        return;
    end
    % 并行特征提取在部分环境下可能导致 MATLAB 异常退出（尤其是线程池在大体积
    % 数据上频繁启动时）。通过失败后永久回退到串行路径，避免闪退。
    persistent featureParallelDisabled failureCount
    if isempty(featureParallelDisabled)
        featureParallelDisabled = false;
    end
    if isempty(failureCount)
        failureCount = 0;
    end
    if isempty(parallelHint)
        parallelEnabled = shouldUseParallel(8, numel(model));
    else
        parallelEnabled = logical(parallelHint);
        if parallelEnabled
            pool = [];
            try
                pool = gcp('nocreate');
            catch
                pool = [];
            end
            if isempty(pool)
                parallelEnabled = shouldUseParallel(8, numel(model));
            end
        end
    end
    if featureParallelDisabled && failureCount > 2
        parallelEnabled = false; % 只有连续失败多次才彻底退回串行
    end
    if parallelEnabled
        try
            featureResults = cell(5, 1);
            parfor idx = 1:5
                switch idx
                    case 1
                        featureResults{idx} = extractEfficientClusterFeatures(model);
                    case 2
                        featureResults{idx} = computeEnhancedSpatialFeatures(model);
                    case 3
                        featureResults{idx} = computeDetailedMorphologyFeatures(model);
                    case 4
                        featureResults{idx} = computeMultiScaleSpatialFeatures(model);
                    case 5
                        featureResults{idx} = calculateDirectOverlapEnergy(model, mcmcState.optParams); % 轻量任务增加并行粒度
                end
            end
            mcmcState.features = featureResults{1};
            mcmcState.spatialFeatures = featureResults{2};
            mcmcState.morphologyFeatures = featureResults{3};
            mcmcState.multiScaleSpatialFeatures = featureResults{4};
            mcmcState.directOverlapEnergy = featureResults{5};
            failureCount = 0;
        catch ME
             warning(ME.identifier, '并行特征提取失败，记录并暂退串行：%s', ME.message);
            failureCount = failureCount + 1;
            featureParallelDisabled = failureCount > 2;
            parallelEnabled = false;
        end
        
    else
        mcmcState.features = extractEfficientClusterFeatures(model);
        mcmcState.spatialFeatures = computeEnhancedSpatialFeatures(model);
        mcmcState.morphologyFeatures = computeDetailedMorphologyFeatures(model);
        mcmcState.multiScaleSpatialFeatures = computeMultiScaleSpatialFeatures(model);
        mcmcState.directOverlapEnergy = calculateDirectOverlapEnergy(model, mcmcState.optParams);
    end
    if featureParallelDisabled && ~parallelEnabled
        % 如果上面因异常禁用并行，确保串行特征在同一轮已经计算
        mcmcState.features = extractEfficientClusterFeatures(model);
        mcmcState.spatialFeatures = computeEnhancedSpatialFeatures(model);
        mcmcState.morphologyFeatures = computeDetailedMorphologyFeatures(model);
        mcmcState.multiScaleSpatialFeatures = computeMultiScaleSpatialFeatures(model);
        mcmcState.directOverlapEnergy = calculateDirectOverlapEnergy(model, mcmcState.optParams);
    end
    [mcmcState.currentEnergy, componentStruct] = computeUnifiedEnergySnapshot( ...
        mcmcState.model, mcmcState.optParams, mcmcState.weights, ...
        mcmcState.features, mcmcState.spatialFeatures, ...
        mcmcState.morphologyFeatures, mcmcState.multiScaleSpatialFeatures);
    mcmcState.energyComponents = componentStruct;
    % 如果能量有所改进，更新最佳记录
    if isfield(mcmcState, 'bestEnergy')
        if mcmcState.currentEnergy < mcmcState.bestEnergy
            mcmcState.bestEnergy = mcmcState.currentEnergy;
            mcmcState.bestModel = mcmcState.model;
        end
    end
    % 计算综合形态得分并更新形态最佳模型
    if ~isfield(mcmcState, 'bestShapeScore') || isempty(mcmcState.bestShapeScore)
        mcmcState.bestShapeScore = -inf;
    end
    shapeScore = computeCompositeShapeScore(mcmcState);
    if shapeScore > mcmcState.bestShapeScore
        mcmcState.bestShapeScore = shapeScore;
        mcmcState.bestShapeModel = mcmcState.model;
    end
    mcmcState.boundaryUpdated = false;
end
function score = computeCompositeShapeScore(mcmcState)
    % 综合考虑形态、空间匹配与直接重叠的一致性得分（分数越高越好）
    alpha = 0.25; beta = 0.25; gamma = 0.5;
    if isfield(mcmcState, 'optParams') && isfield(mcmcState.optParams, 'shapeFirstMode') && mcmcState.optParams.shapeFirstMode
        alpha = 0.2; beta = 0.2; gamma = 0.6;
    end
    overlapScore = 1 - mcmcState.directOverlapEnergy;
    score = alpha * mcmcState.currentMorphMatch + beta * mcmcState.currentSpatialMatch + gamma * overlapScore;
end
function mcmcState = enforceComprehensiveConstraints(mcmcState, optParams, phase, parallelHint)
    % 强制执行综合约束
    if nargin < 4
        parallelHint = [];
    end
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
    % 定期硬性纠正孔隙率，防止持续漂移
    mcmcState.model = enforcePorosityClusterTargets(mcmcState.model, optParams, optParams.referenceDensityMap);
    % 约束后立即刷新特征和能量，保证评估指标可靠
    mcmcState = updateAllFeatures(mcmcState, parallelHint);
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
                        % 减少球形度，使用三维各向异性膨胀
                        dir = randn(1, 3);
                        se = createAnisotropicStructuringElement(dir, [3, 1, 1]);
                        adjusted = imdilate(clusterMask, se);
                        adjusted = imerode(adjusted, strel('sphere', 1));
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
    % 使用软约束调整簇大小，避免强制固定范围
    CC = bwconncomp(model, 26);
    if CC.NumObjects == 0
        return;
    end
    sizes = cellfun(@numel, CC.PixelIdxList);
    modified = false;
    [lowerBound, upperBound] = getAdaptiveClusterBounds(optParams);
    if isempty(lowerBound) || isempty(upperBound) || upperBound <= 0
        lowerBound = max(1, round(prctile(double(sizes), 5)));
        upperBound = max(lowerBound + 1, round(prctile(double(sizes), 95)));
    end
    % 放宽阈值，允许 40% 的偏差
    shrinkThreshold = max(upperBound * 1.4, upperBound + max(20, round(0.1 * upperBound)));
    removeThreshold = max(1, floor(lowerBound * 0.5));
    growThreshold = max(lowerBound, floor(lowerBound * 0.9));
    % 处理极大簇（采用温和收缩）
    largeClusters = find(sizes > shrinkThreshold);
    for i = 1:length(largeClusters)
        idx = largeClusters(i);
        model = reduceOversizedCluster(model, CC.PixelIdxList{idx}, optParams);
        modified = true;
    end
    % 删除极小簇或尝试生长
    tinyClusters = find(sizes < removeThreshold);
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
    if modified
        CC = bwconncomp(model, 26);
        sizes = cellfun(@numel, CC.PixelIdxList);
    end
    % 尝试合并接近的小簇，鼓励形成稳定结构
    smallClusters = find(sizes >= removeThreshold & sizes < growThreshold);
    if length(smallClusters) >= 2
        for i = 1:length(smallClusters)-1
            for j = i+1:length(smallClusters)
                idx1 = smallClusters(i);
                idx2 = smallClusters(j);
                [dist, ~, ~] = findClosestPoints(model, CC.PixelIdxList{idx1}, CC.PixelIdxList{idx2});
                if dist < 5
                    model = connectClusters(model, CC.PixelIdxList{idx1}, CC.PixelIdxList{idx2});
                    modified = true;
                    break;
                end
            end
            if modified
                break;
            end
        end
    end
end
function model = enforceTargetClusterCount(model, optParams)
    % 对簇数量施加软约束，仅在显著偏差时纠偏
    preserveSmall = isfield(optParams, 'preserveSmallPores') && optParams.preserveSmallPores;
    targetCount = [];
    if isfield(optParams, 'clusterReference') && ~isempty(optParams.clusterReference)
        targetCount = optParams.clusterReference.numClusters;
    end
    [minCount, maxCount] = getAdaptiveClusterCountBounds(optParams);
    if minCount == 0 && isinf(maxCount)
        return;
    end
    CC = bwconncomp(model, 26);
    currentCount = CC.NumObjects;
    if currentCount == 0
        model = seedRandomCluster(model, optParams);
        return;
    end
    if ~isempty(targetCount) && abs(currentCount - targetCount) <= 0.3 * max(targetCount, 1)
        % 小偏差交给能量函数处理
        return;
    end
    sizes = cellfun(@numel, CC.PixelIdxList);
    if currentCount < minCount
        deficit = minCount - currentCount;
        maxAdjust = max(1, ceil(deficit * 0.5));
        for i = 1:maxAdjust
            [model, added] = addRepresentativeCluster(model, CC, optParams);
            if ~added && preserveSmall
                model = splitOversizedCluster(model, CC);
            elseif ~added
                model = seedRandomCluster(model, optParams);
            end
            CC = bwconncomp(model, 26);
        end
    elseif currentCount > maxCount
        excess = currentCount - maxCount;
        maxAdjust = max(1, ceil(excess * 0.5));
        for i = 1:maxAdjust
            CC = bwconncomp(model, 26);
            if CC.NumObjects <= maxCount
                break;
            end
            if preserveSmall
                model = mergeLargestClusters(model, CC, sizes);
                model = repairThinChannels(model);
            else
                model = mergeClosestClusters(model, CC);
            end
            sizes = cellfun(@numel, CC.PixelIdxList);
        end
    end
end
function model = splitOversizedCluster(model, CC)
    % 在最大簇内切开薄层，优先通过“簇切分”增加簇数量
    if CC.NumObjects == 0
        return;
    end
    [~, idx] = max(cellfun(@numel, CC.PixelIdxList));
    mask = false(size(model));
    mask(CC.PixelIdxList{idx}) = true;
    bbox = regionprops3(mask, 'BoundingBox');
    if isempty(bbox)
        return;
    end
    bb = bbox.BoundingBox(1, :);
    center = round(bb(1:3) + bb(4:6) / 2);
    planeAxis = randi(3);
    sliceMask = false(size(model));
    switch planeAxis
        case 1
            sliceMask(center(2), :, :) = true;
        case 2
            sliceMask(:, center(1), :) = true;
        case 3
            sliceMask(:, :, center(3)) = true;
    end
    sliceMask = imdilate(sliceMask, strel('sphere', 1));
    removal = mask & sliceMask;
    model(removal) = false;
end
function model = mergeLargestClusters(model, CC, sizes)
    % 仅在较大簇之间建立窄通道，减少簇数量同时避免吞并小孔隙
    if nargin < 3 || isempty(sizes)
        sizes = cellfun(@numel, CC.PixelIdxList);
    end
    [~, order] = sort(sizes, 'descend');
    if numel(order) < 2
        return;
    end
    idxA = order(1);
    idxB = order(2);
    maskA = false(size(model)); maskA(CC.PixelIdxList{idxA}) = true;
    maskB = false(size(model)); maskB(CC.PixelIdxList{idxB}) = true;
    distMapA = bwdist(~maskA);
    distMapB = bwdist(~maskB);
    connection = (distMapA + distMapB) == min(distMapA(:) + distMapB(:));
    bridge = imdilate(connection, strel('sphere', 1));
    model(bridge) = true;
end
function model = repairThinChannels(model)
    % 修复因局部操作导致的细通道断裂，减少最大连通簇碎裂
    skeleton = bwskel(model);
    endpoints = bwmorph3(skeleton, 'endpoints');
    if nnz(endpoints) < 2
        return;
    end
    [x, y, z] = ind2sub(size(model), find(endpoints));
    pts = [x, y, z];
    if size(pts, 1) < 2
        return;
    end
    idx = randperm(size(pts, 1), min(2, size(pts, 1)));
    p1 = pts(idx(1), :); p2 = pts(idx(end), :);
    lineMask = false(size(model));
    numSteps = max(abs(p2 - p1)) + 1;
    for t = linspace(0, 1, numSteps)
        pos = round(p1 * (1 - t) + p2 * t);
        lineMask(pos(1), pos(2), pos(3)) = true;
    end
    lineMask = imdilate(lineMask, strel('sphere', 1));
    model(lineMask) = true;
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
    [lowerBound, ~] = getAdaptiveClusterBounds(optParams);
    if isempty(lowerBound) || lowerBound <= 0
        lowerBound = 30;
    end
    radius = max(2, round(lowerBound^(1/3)));
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
     % 使用整数偏移，避免非整数索引导致错误
    hx = floor((sx - 1) / 2);
    hy = floor((sy - 1) / 2);
    hz = floor((sz - 1) / 2);
    for i = 1:sx
        for j = 1:sy
            for k = 1:sz
                x = cx - hx + (i - 1);
                y = cy - hy + (j - 1);
                z = cz - hz + (k - 1);
                
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
    % 使用地质距离路径连接两个簇，避免直线穿墙
    volumeSize = size(model);
    seed1 = false(volumeSize);
    seed1(cluster1) = true;
    seed2 = false(volumeSize);
    seed2(cluster2) = true;
    contactRegion1 = imdilate(seed1, strel('sphere', 2)) & ~model;
    contactRegion2 = imdilate(seed2, strel('sphere', 2));
    if ~any(contactRegion1(:))
        contactRegion1 = seed1;
    end
    costVolume = ones(volumeSize, 'double');
    boundaryMask = bwperim(model, 26);
    costVolume(~model) = 2.5;
    costVolume(boundaryMask) = 1.0;
    if exist('graydist', 'file') == 2
        try
            distMap = graydist(costVolume, contactRegion1, 'quasi-euclidean');
        catch
            distMap = graydist(costVolume, contactRegion1);
        end
    else
        try
            distMap = bwdistgeodesic(true(volumeSize), contactRegion1, 'quasi-euclidean');
        catch
            distMap = bwdistgeodesic(true(volumeSize), contactRegion1);
        end
        distMap = distMap .* costVolume;
    end
    targetCandidates = find(contactRegion2);
    if isempty(targetCandidates)
        targetCandidates = cluster2;
    end
    [~, minIdx] = min(distMap(targetCandidates));
    if isempty(minIdx) || ~isfinite(minIdx)
        return;
    end
    targetLinear = targetCandidates(minIdx);
    path = traceGeodesicPath(distMap, targetLinear);
    if isempty(path)
        return;
    end
    model(path) = true;
    model = imdilate(model, strel('sphere', 1));
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
function boundaryStruct = computeBoundaryMasks(model)
    poreMask = model;
    solidMask = ~model;
    poreBoundary = poreMask & hasNeighborOfPhase(model, 0);
    solidBoundary = solidMask & hasNeighborOfPhase(model, 1);
    boundaryStruct.poreIdx = find(poreBoundary);
    boundaryStruct.solidIdx = find(solidBoundary);
    boundaryStruct.poreBoundaryMask = poreBoundary;
    boundaryStruct.solidBoundaryMask = solidBoundary;
    boundaryStruct.poreMask = poreMask;
    boundaryStruct.solidMask = solidMask;
end
function mask = hasNeighborOfPhase(model, phase)
    kernel = zeros(3,3,3);
    kernel(2,2,1) = 1; kernel(2,2,3) = 1;
    kernel(2,1,2) = 1; kernel(2,3,2) = 1;
    kernel(1,2,2) = 1; kernel(3,2,2) = 1;
    if phase == 1
        neighborCount = convn(double(model), kernel, 'same');
        mask = neighborCount > 0;
    else
        neighborCount = convn(double(~model), kernel, 'same');
        mask = neighborCount > 0;
    end
end
function mcmcState = refreshBoundaryContext(mcmcState)
    % 为当前模型刷新边界掩膜与索引，供边界驱动的移动使用
    if ~isfield(mcmcState, 'boundaryUpdated') || ~mcmcState.boundaryUpdated
        mcmcState.boundaryContext = computeBoundaryMasks(mcmcState.model);
        mcmcState.boundaryUpdated = true;
    end
end
%% ========== 移动策略和生成函数 ==========
function strategy = selectComprehensiveAdaptiveMoveStrategy(iter, maxIter, mcmcState, phase)
    % 选择综合自适应移动策略
    progress = iter / maxIter;
    
    switch phase
        case 'morphology'
            % 形态优化阶段策略
            strategies = {'boundary_exchange', 'boundary_birthdeath', 'boundary_smooth', 'local_shape'};
            weights = [0.55, 0.25, 0.1, 0.1];
        case 'spatial'
            % 空间优化阶段策略
            strategies = {'boundary_exchange', 'spatial_aware', 'boundary_birthdeath', 'anisotropy_adjust'};
            weights = [0.45, 0.25, 0.2, 0.1];
        case 'balanced'
            % 平衡阶段策略
            strategies = {'boundary_exchange', 'boundary_birthdeath', 'fine_tune', 'structure_coherence'};
            weights = [0.4, 0.35, 0.15, 0.1];
            
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
        % 接受率太高，使用更激进的策略（形态阶段仍避免大尺度破坏）
        if strcmp(phase, 'morphology')
            strategies = [strategies, {'local_shape'}];
            weights = [weights * 0.85, 0.15];
        else
            strategies = [strategies, {'large_scale'}];
            weights = [weights * 0.8, 0.2];
        end
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
function [moves, moveTypes] = generateComprehensiveBatchMoves(mcmcState, batchSize, optParams, lookupTables, parallelEnabled, strategy)
    % 生成以边界为主的批量移动，维持孔隙率稳定
    if nargin < 6 || isempty(strategy)
        strategy = 'boundary_exchange';
    end
    model = mcmcState.model;
    moves = cell(batchSize, 1);
    moveTypes = cell(batchSize, 1);
    if batchSize <= 0
        return;
    end
    if nargin < 5 || isempty(parallelEnabled)
        parallelEnabled = shouldUseParallel(batchSize, numel(model));
    end
    if ~isfield(mcmcState, 'boundaryUpdated') || ~mcmcState.boundaryUpdated
        mcmcState.boundaryContext = computeBoundaryMasks(model);
        mcmcState.boundaryUpdated = true;
    end
    boundaryContext = mcmcState.boundaryContext;
    porosityDiff = mean(model(:)) - optParams.targetPorosity;
    typeWeights = computeBoundaryMoveWeights(strategy, porosityDiff);
    if parallelEnabled
        parfor i = 1:batchSize
           [moves{i}, moveTypes{i}] = generateSingleBoundaryMove(model, boundaryContext, ...
                optParams, lookupTables, typeWeights, porosityDiff);
        end
    else
        for i = 1:batchSize
           [moves{i}, moveTypes{i}] = generateSingleBoundaryMove(model, boundaryContext, ...
                optParams, lookupTables, typeWeights, porosityDiff);
        end
    end
end
function weights = computeBoundaryMoveWeights(strategy, porosityDiff)
    base = struct('exchange', 0.72, 'birthdeath', 0.23, 'random', 0.05);
    if abs(porosityDiff) > 0.01
        base.birthdeath = base.birthdeath + 0.07;
        base.exchange = base.exchange - 0.04;
    end
    if strcmp(strategy, 'spatial') || strcmp(strategy, 'spatial_aware')
        base.exchange = base.exchange + 0.05;
        base.birthdeath = base.birthdeath - 0.03;
    end
    weights = [max(base.exchange, 0), max(base.birthdeath, 0), max(base.random, 0)];
    weights = weights / sum(weights);
end
function [move, moveType] = generateSingleBoundaryMove(model, boundaryContext, optParams, lookupTables, weights, porosityDiff)
    r = rand();
    cum = cumsum(weights);
    if r <= cum(1)
        moveType = 'boundary_exchange';
        move = generateBoundaryExchangeMove(model, boundaryContext);
    elseif r <= cum(2)
        moveType = 'boundary_birthdeath';
        move = generateBoundaryBirthDeathMove(model, boundaryContext, optParams.targetPorosity);
    else
        moveType = 'global_random';
        move = generateGlobalExplorationMove(model, boundaryContext, lookupTables);
    end
    move = enforceTopologyAwareMove(move, model);
end
function move = generateBoundaryExchangeMove(model, boundaryContext)
    move = struct('linearIdx', [], 'oldValues', [], 'newValues', []);
    if isempty(boundaryContext.poreIdx) || isempty(boundaryContext.solidIdx)
        move = generateLocalMove(model, 2);
        return;
    end
    poreIdx = boundaryContext.poreIdx(randi(numel(boundaryContext.poreIdx)));
    [px, py, pz] = ind2sub(size(model), poreIdx);
    [sx, sy, sz] = ind2sub(size(model), boundaryContext.solidIdx);
    dist = sqrt((sx - px).^2 + (sy - py).^2 + (sz - pz).^2);
    candidates = boundaryContext.solidIdx(dist <= 2);
    if isempty(candidates)
        move = generateLocalMove(model, 2);
        return;
    end
    solidIdx = candidates(randi(numel(candidates)));
    move.linearIdx = [poreIdx; solidIdx];
    move.oldValues = [true; false];
    move.newValues = [false; true];
end
function move = generateBoundaryBirthDeathMove(model, boundaryContext, targetPorosity)
    move = struct('linearIdx', [], 'oldValues', [], 'newValues', []);
    currentPorosity = mean(model(:));
    increase = currentPorosity < targetPorosity;
    if increase
        sourceSet = boundaryContext.solidIdx;
        newVal = true;
        oldVal = false;
    else
        sourceSet = boundaryContext.poreIdx;
        newVal = false;
        oldVal = true;
    end
    if isempty(sourceSet)
        move = generateLocalMove(model, 2);
        return;
    end
    nFlip = min(max(1, round(0.001 * numel(model))), numel(sourceSet));
    chosen = sourceSet(randperm(numel(sourceSet), nFlip));
    move.linearIdx = chosen(:);
    move.oldValues = repmat(oldVal, numel(chosen), 1);
    move.newValues = repmat(newVal, numel(chosen), 1);
end
function move = generateGlobalExplorationMove(model, boundaryContext, lookupTables)
    %#ok<INUSD>
    move = struct('linearIdx', [], 'oldValues', [], 'newValues', []);
    [nx, ny, nz] = size(model);
    if isfield(boundaryContext, 'poreIdx') && ~isempty(boundaryContext.poreIdx)
        centerIdx = boundaryContext.poreIdx(randi(numel(boundaryContext.poreIdx)));
    elseif isfield(boundaryContext, 'solidIdx') && ~isempty(boundaryContext.solidIdx)
        centerIdx = boundaryContext.solidIdx(randi(numel(boundaryContext.solidIdx)));
    else
        centerIdx = randi(numel(model));
    end
    [cx, cy, cz] = ind2sub(size(model), centerIdx);
    center = [cx, cy, cz];
    radius = 1;
    [xRange, yRange, zRange] = ndgrid( ...
        max(1, center(1)-radius):min(nx, center(1)+radius), ...
        max(1, center(2)-radius):min(ny, center(2)+radius), ...
        max(1, center(3)-radius):min(nz, center(3)+radius));
    localIdx = sub2ind(size(model), xRange(:), yRange(:), zRange(:));
    localValues = model(localIdx);
    poreIdx = localIdx(localValues);
    solidIdx = localIdx(~localValues);
    if isempty(poreIdx) || isempty(solidIdx)
        move.linearIdx = localIdx(1:min(numel(localIdx), 2));
        move.oldValues = localValues(1:min(numel(localIdx), 2));
        move.newValues = ~move.oldValues;
        return;
    end
    nSwap = min(ceil(numel(localIdx) * 0.15), min(numel(poreIdx), numel(solidIdx)));
    poreChosen = poreIdx(randperm(numel(poreIdx), nSwap));
    solidChosen = solidIdx(randperm(numel(solidIdx), nSwap));
    move.linearIdx = [poreChosen; solidChosen];
    move.oldValues = [true(nSwap,1); false(nSwap,1)];
    move.newValues = [false(nSwap,1); true(nSwap,1)];
end
%% ========== 具体移动生成函数 ==========
function move = generateLocalMove(model, radius)
    % 生成围绕孔隙簇的局部移动，优先在簇边界附近调整
    if nargin < 2
        radius = 2; % 限制在1~2体素边界层内
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
    % 生成边界平滑移动（向量化实现）
    move = struct();
    stats = precomputeLocalMoveStatistics(model);
    boundaryMask = stats.boundaryMask;
    boundaryIdx = find(boundaryMask);
    if isempty(boundaryIdx)
        move = generateLocalMove(model, 3);
        return;
    end
    neighborCount = double(stats.neighborCount6(boundaryMask));
    neighborSum = double(stats.neighborSum6(boundaryMask));
    centerValues = model(boundaryIdx);
    diffCounts = neighborSum;
    if any(centerValues)
        filledMask = centerValues;
        diffCounts(filledMask) = neighborCount(filledMask) - neighborSum(filledMask);
    end
    roughness = diffCounts ./ max(neighborCount, 1);
    nSmooth = min(5, numel(boundaryIdx));
    [~, order] = sort(roughness, 'descend');
    selected = order(1:nSmooth);
    candidateIdx = boundaryIdx(selected);
    majorityValues = neighborSum(selected) >= (neighborCount(selected) / 2);
    oldValues = centerValues(selected);
    changeMask = oldValues ~= majorityValues;
    if ~any(changeMask)
        move = generateLocalMove(model, 3);
        return;
    end
    move.linearIdx = candidateIdx(changeMask);
    move.oldValues = oldValues(changeMask);
    move.newValues = majorityValues(changeMask);
    move.oldValues = move.oldValues(:);
    move.newValues = move.newValues(:);
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
    [lowerBound, upperBound] = getAdaptiveClusterBounds(optParams);
    if isempty(lowerBound) || isempty(upperBound)
        lowerBound = prctile(double(sizes), 20);
        upperBound = prctile(double(sizes), 80);
    end
    midSizeClusters = find(sizes > lowerBound & sizes < upperBound);
    
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
    [lowerBound, upperBound] = getAdaptiveClusterBounds(optParams);
    if isempty(lowerBound) || isempty(upperBound)
        lowerBound = prctile(double(sizes), 15);
        upperBound = prctile(double(sizes), 85);
    end
    midSizeClusters = find(sizes > lowerBound & sizes < upperBound);
    
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
    % 生成与全局能量一致的空间相关性移动
    move = struct('linearIdx', [], 'oldValues', [], 'newValues', []);
    weights = getEnergyWeights(optParams, optParams.energyWeights);
    targetDensityMap = computeTargetDensityMap(optParams.spatialFeatures, optParams, size(model));
    kernel = ones(5, 5, 5, 'double');
    kernel = kernel / sum(kernel(:));
    localDensity = convn(double(model), kernel, 'same');
    densityError = localDensity - targetDensityMap;
    boundaryMask = bwperim(model, 26);
    surfaceBand = imdilate(boundaryMask, strel('sphere', 2));
    addCandidates = find(surfaceBand & ~model);
    removeCandidates = find(surfaceBand & model);
    if isempty(addCandidates) && isempty(removeCandidates)
        move = generateLocalMove(model, 4);
        return;
    end
    [baselineEnergy, ~, ~] = computeUnifiedEnergySnapshot(model, optParams, weights);
    bestEnergy = baselineEnergy;
    bestMove = move;
    maxCandidates = 4;
    if ~isempty(addCandidates)
        [~, order] = sort(densityError(addCandidates), 'ascend');
        addCandidates = addCandidates(order(1:min(maxCandidates, numel(order))));
        for idx = addCandidates(:)'
            candidateMove = createSpatialAdjustmentMove(model, idx, 'add', densityError, optParams);
            candidateMove = enforceTopologyAwareMove(candidateMove, model);
            if isempty(candidateMove.linearIdx)
                continue;
            end
            candidateModel = applyMoveToModel(model, candidateMove);
            candidateEnergy = computeUnifiedEnergySnapshot(candidateModel, optParams, weights);
            if candidateEnergy < bestEnergy
                bestEnergy = candidateEnergy;
                bestMove = candidateMove;
            end
        end
    end
    if ~isempty(removeCandidates)
        [~, order] = sort(densityError(removeCandidates), 'descend');
        removeCandidates = removeCandidates(order(1:min(maxCandidates, numel(order))));
        for idx = removeCandidates(:)'
            candidateMove = createSpatialAdjustmentMove(model, idx, 'remove', densityError, optParams);
            candidateMove = enforceTopologyAwareMove(candidateMove, model);
            if isempty(candidateMove.linearIdx)
                continue;
            end
            candidateModel = applyMoveToModel(model, candidateMove);
            candidateEnergy = computeUnifiedEnergySnapshot(candidateModel, optParams, weights);
            if candidateEnergy < bestEnergy
                bestEnergy = candidateEnergy;
                bestMove = candidateMove;
            end
        end
    end
    if isempty(bestMove.linearIdx)
        move = generateLocalMove(model, 4);
    else
        move = bestMove;
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
    [lowerBound, ~] = getAdaptiveClusterBounds(optParams);
    if isempty(lowerBound)
        lowerBound = prctile(double(sizes), 20);
    end
    if CC.NumObjects >= 2 && sortedSizes(2) > lowerBound
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
%% ========== 拓扑与局部调整辅助函数 ==========
function targetDensityMap = computeTargetDensityMap(targetSpatial, optParams, modelSize)
    % 构造目标密度图，优先使用参考密度图并保持尺度一致
    basePorosity = 0.5;
    if isfield(optParams, 'targetPorosity') && ~isempty(optParams.targetPorosity)
        basePorosity = optParams.targetPorosity;
    end
    targetDensityMap = ones(modelSize, 'double') * basePorosity;
    if nargin < 3 || isempty(modelSize)
        modelSize = size(optParams.originalBinaryModel);
    end
    if isfield(optParams, 'referenceDensityMap') && ~isempty(optParams.referenceDensityMap)
        refMap = optParams.referenceDensityMap;
        if ~isequal(size(refMap), modelSize)
            refMap = resizeVolume(refMap, modelSize);
        end
        targetDensityMap = double(refMap);
    elseif isstruct(targetSpatial) && isfield(targetSpatial, 'localDensityMap') && ...
            ~isempty(targetSpatial.localDensityMap)
        refMap = targetSpatial.localDensityMap;
        if ~isequal(size(refMap), modelSize)
            refMap = resizeVolume(refMap, modelSize);
        end
        targetDensityMap = double(refMap);
    end
    maxVal = max(targetDensityMap(:));
    if maxVal > 0
        targetDensityMap = targetDensityMap / maxVal;
    end
end
function move = createSpatialAdjustmentMove(model, centerIdx, mode, densityError, optParams)
    % 基于局部密度误差生成拓扑安全的候选移动
    if nargin < 5
        optParams = struct();
    end
    move = struct('linearIdx', [], 'oldValues', [], 'newValues', []);
    radius = 2;
    neighborhood = getNeighborhoodIndices(size(model), centerIdx, radius);
    if isempty(neighborhood)
        return;
    end
    switch mode
        case 'add'
            candidateIdx = neighborhood(~model(neighborhood));
            if isempty(candidateIdx)
                return;
            end
            scores = densityError(candidateIdx);
            [~, order] = sort(scores, 'ascend');
            nSelect = min(6, numel(order));
            selected = candidateIdx(order(1:nSelect));
            move.linearIdx = selected(:);
            move.oldValues = false(numel(selected), 1);
            move.newValues = true(numel(selected), 1);
        case 'remove'
            candidateIdx = neighborhood(model(neighborhood));
            if isempty(candidateIdx)
                return;
            end
            scores = densityError(candidateIdx);
            [~, order] = sort(scores, 'descend');
            nSelect = min(6, numel(order));
            selected = candidateIdx(order(1:nSelect));
            move.linearIdx = selected(:);
            move.oldValues = true(numel(selected), 1);
            move.newValues = false(numel(selected), 1);
        otherwise
            return;
    end
end
function idxList = getNeighborhoodIndices(modelSize, centerIdx, radius)
    % 获取给定半径内的邻域索引，使用球形邻域
    [cx, cy, cz] = ind2sub(modelSize, centerIdx);
    xRange = max(1, cx-radius):min(modelSize(1), cx+radius);
    yRange = max(1, cy-radius):min(modelSize(2), cy+radius);
    zRange = max(1, cz-radius):min(modelSize(3), cz+radius);
    idxList = [];
    for x = xRange
        for y = yRange
            for z = zRange
                if sqrt((x-cx)^2 + (y-cy)^2 + (z-cz)^2) <= radius + 0.01
                    idxList(end+1, 1) = sub2ind(modelSize, x, y, z); %#ok<AGROW>
                end
            end
        end
    end
    idxList = unique(idxList, 'stable');
end
function model = applyMoveToModel(model, move)
    % 将移动应用到模型上
    if isempty(move) || ~isfield(move, 'linearIdx') || isempty(move.linearIdx)
        return;
    end
    model(move.linearIdx) = logical(move.newValues);
end
function move = enforceTopologyAwareMove(move, model)
    % 过滤不满足拓扑安全约束的移动
    if isempty(move) || ~isfield(move, 'linearIdx') || isempty(move.linearIdx)
        return;
    end
    additions = move.linearIdx(move.newValues & ~move.oldValues);
    removals = move.linearIdx(move.oldValues & ~move.newValues);
    boundary = bwperim(model, 26);
    surfaceBand = imdilate(boundary, strel('sphere', 1));
    validAdd = false(size(additions));
    for i = 1:numel(additions)
        idx = additions(i);
        if surfaceBand(idx) && isAdditionTopologySafe(model, idx)
            validAdd(i) = true;
        end
    end
    validRemove = false(size(removals));
    for i = 1:numel(removals)
        idx = removals(i);
        if boundary(idx) && isRemovalTopologySafe(model, idx)
            validRemove(i) = true;
        end
    end
    additions = additions(validAdd);
    removals = removals(validRemove);
    newLinearIdx = [additions(:); removals(:)];
    newOldValues = [false(numel(additions), 1); true(numel(removals), 1)];
    newNewValues = [true(numel(additions), 1); false(numel(removals), 1)];
    if isempty(newLinearIdx)
        move.linearIdx = [];
        move.oldValues = [];
        move.newValues = [];
        return;
    end
    [uniqueIdx, order] = unique(newLinearIdx, 'stable');
    move.linearIdx = uniqueIdx;
    move.oldValues = newOldValues(order);
    move.newValues = newNewValues(order);
end
function safe = isAdditionTopologySafe(model, idx)
    % 添加体素时确保连接到现有表面且不会形成孤立簇
    [patch, cx, cy, cz] = extractLocalPatch(model, idx, 1);
    patch(cx, cy, cz) = false;
    neighborCount = sum(patch(:));
    if neighborCount < 2
        safe = false;
        return;
    end
    newPatch = patch;
    newPatch(cx, cy, cz) = true;
    CC = bwconncomp(newPatch, 26);
    safe = CC.NumObjects == 1;
end
function safe = isRemovalTopologySafe(model, idx)
    % 移除体素时确保不切断主通道
    [patch, cx, cy, cz] = extractLocalPatch(model, idx, 2);
    if ~patch(cx, cy, cz)
        safe = false;
        return;
    end
    patch(cx, cy, cz) = false;
    neighborCount = sum(patch(:));
    if neighborCount == 0
        safe = true;
        return;
    end
    if neighborCount < 2
        safe = false;
        return;
    end
    CC = bwconncomp(patch, 26);
    safe = CC.NumObjects == 1;
end
function [patch, cx, cy, cz] = extractLocalPatch(model, idx, radius)
    % 提取包含目标索引的局部补丁
    [x, y, z] = ind2sub(size(model), idx);
    x1 = max(1, x-radius); x2 = min(size(model, 1), x+radius);
    y1 = max(1, y-radius); y2 = min(size(model, 2), y+radius);
    z1 = max(1, z-radius); z2 = min(size(model, 3), z+radius);
    patch = model(x1:x2, y1:y2, z1:z2);
    cx = x - x1 + 1;
    cy = y - y1 + 1;
    cz = z - z1 + 1;
end
function se = createAnisotropicStructuringElement(direction, halfLengths)
    % 创建一个各向异性的三维结构元素
    if nargin < 2 || isempty(halfLengths)
        halfLengths = [2, 1, 1];
    end
    if numel(halfLengths) == 1
        halfLengths = repmat(halfLengths, 1, 3);
    elseif numel(halfLengths) == 2
        halfLengths = [halfLengths, 1];
    end
    direction = direction(:)';
    if all(direction == 0)
        direction = [1, 0, 0];
    end
    direction = abs(direction);
    if sum(direction) == 0
        direction = [1, 0, 0];
    end
    [~, axisOrder] = sort(direction, 'descend');
    radii = ones(1, 3);
    for i = 1:3
        idx = axisOrder(i);
        radii(idx) = max(1, round(halfLengths(min(i, numel(halfLengths)))));
    end
    rx = radii(1); ry = radii(2); rz = radii(3);
    [X, Y, Z] = ndgrid(-rx:rx, -ry:ry, -rz:rz);
    mask = (X.^2/(rx+eps)^2 + Y.^2/(ry+eps)^2 + Z.^2/(rz+eps)^2) <= 1;
    se = strel('arbitrary', mask);
end
function path = traceGeodesicPath(distMap, startIdx)
    % 沿距离图追踪最短路径
    path = startIdx;
    currentIdx = startIdx;
    sz = size(distMap);
    visited = false(numel(distMap), 1);
    while true
        if ~isfinite(distMap(currentIdx)) || distMap(currentIdx) <= 0
            break;
        end
        visited(currentIdx) = true;
        [x, y, z] = ind2sub(sz, currentIdx);
        bestIdx = currentIdx;
        bestVal = distMap(currentIdx);
        for dx = -1:1
            for dy = -1:1
                for dz = -1:1
                    if dx == 0 && dy == 0 && dz == 0
                        continue;
                    end
                    nx = x + dx; ny = y + dy; nz = z + dz;
                    if nx >= 1 && nx <= sz(1) && ny >= 1 && ny <= sz(2) && nz >= 1 && nz <= sz(3)
                        neighborIdx = sub2ind(sz, nx, ny, nz);
                        neighborVal = distMap(neighborIdx);
                        if neighborVal < bestVal && isfinite(neighborVal)
                            bestVal = neighborVal;
                            bestIdx = neighborIdx;
                        end
                    end
                end
            end
        end
        if bestIdx == currentIdx || visited(bestIdx)
            break;
        end
        path(end+1, 1) = bestIdx; %#ok<AGROW>
        currentIdx = bestIdx;
    end
    path = unique(path, 'stable');
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
    [lowerBound, upperBound] = getAdaptiveClusterBounds(optParams);
    if isempty(lowerBound)
        lowerBound = prctile(double(sizes), 20);
    end
    if isempty(upperBound)
        upperBound = prctile(double(sizes), 80);
    end
    if any(sizes > upperBound)
        % 缩小过大的簇
        largeClusterIdx = find(sizes > upperBound, 1);
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
        
    elseif any(sizes < lowerBound)
        % 扩大过小的簇
        smallClusterIdx = find(sizes < lowerBound, 1);
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
    moveTypes, lookupTables, phase, parallelEnabled)
    % 评估综合批量移动
    nMoves = length(moves);
    deltaEnergies = zeros(nMoves, 1);
    if nMoves == 0
        return;
    end
    persistent parallelBatchSupported parallelFailureWarned
    if isempty(parallelBatchSupported)
        parallelBatchSupported = true;
    end
    if isempty(parallelFailureWarned)
        parallelFailureWarned = false;
    end
    if nargin < 6 || isempty(parallelEnabled)
        parallelEnabled = shouldUseParallel(nMoves, numel(mcmcState.model));
    end
    localStats = precomputeLocalMoveStatistics(mcmcState.model);
    evalState = prepareMoveEvaluationState(mcmcState);
    useParallel = parallelEnabled && nMoves > 1 && parallelBatchSupported;
    parallelSucceeded = false;
    if useParallel
        try
            parfor i = 1:nMoves
                deltaEnergies(i) = safeEvaluateComprehensiveMove(evalState, moves{i}, ...
                    moveTypes{i}, lookupTables, phase, i, localStats);
            end
            parallelSucceeded = true;
        catch ME
            parallelBatchSupported = false;
            if ~parallelFailureWarned
               warning(ME.identifier, '评估移动 %d 时出错: %s', moveIndex, ME.message);
                parallelFailureWarned = true;
            end
            parallelSucceeded = false;
        end
    end
    if ~parallelSucceeded
        for i = 1:nMoves
            deltaEnergies(i) = safeEvaluateComprehensiveMove(evalState, moves{i}, ...
                moveTypes{i}, lookupTables, phase, i, localStats);
        end
    end
end
function evalState = prepareMoveEvaluationState(mcmcState)
    % 为批量能量评估准备精简状态，避免并行传递不支持的字段
    evalState = struct();
    evalState.model = mcmcState.model;
    evalState.optParams = struct();
    if isfield(mcmcState, 'optParams') && ~isempty(mcmcState.optParams)
        evalState.optParams = mcmcState.optParams;
    end
    evalState.weights = mcmcState.weights;
    if isfield(mcmcState, 'spatialFeatures')
        evalState.spatialFeatures = mcmcState.spatialFeatures;
    else
        evalState.spatialFeatures = struct();
    end
    if isfield(mcmcState, 'morphologyFeatures')
        evalState.morphologyFeatures = mcmcState.morphologyFeatures;
    else
        evalState.morphologyFeatures = struct();
    end
    if isfield(mcmcState, 'multiScaleSpatialFeatures')
        evalState.multiScaleSpatialFeatures = mcmcState.multiScaleSpatialFeatures;
    else
        evalState.multiScaleSpatialFeatures = struct();
    end
end
function deltaE = safeEvaluateComprehensiveMove(mcmcState, move, moveType, ...
    lookupTables, phase, moveIndex, localStats)
    % 并行安全的能量评估包装器
    if nargin < 7
        localStats = [];
    end
    try
        deltaE = evaluateComprehensiveMoveDelta(mcmcState, move, moveType, ...
            lookupTables, phase, localStats);
    catch ME
        warning('评估移动 %d 时出错: %s', moveIndex, ME.message);
        deltaE = evaluateSimpleMoveDelta(mcmcState, move);
    end
    if ~isfinite(deltaE)
        deltaE = evaluateSimpleMoveDelta(mcmcState, move);
    end
end
function localStats = precomputeLocalMoveStatistics(model)
    % 为批量能量评估预计算局部统计量
    persistent kernel6
    if isempty(kernel6)
        kernel6 = single(zeros(3, 3, 3));
        kernel6(2, 2, 1) = 1;
        kernel6(2, 2, 3) = 1;
        kernel6(2, 1, 2) = 1;
        kernel6(2, 3, 2) = 1;
        kernel6(1, 2, 2) = 1;
        kernel6(3, 2, 2) = 1;
    end
    localStats = struct();
    localStats.boundaryMask = bwperim(model, 26);
    localStats.neighborSum6 = convn(single(model), kernel6, 'same');
    localStats.neighborCount6 = convn(ones(size(model), 'single'), kernel6, 'same');
end
function deltaE = evaluateComprehensiveMoveDelta(mcmcState, move, moveType, ...
    lookupTables, phase, localStats)
    % 评估综合移动的能量变化
    % 基础能量变化
    baseDE = evaluateBaseMoveDelta(mcmcState, move, localStats);
    % 当前特征匹配度
    targetSpatial = mcmcState.optParams.spatialFeatures;
    targetMorph = mcmcState.optParams.morphologyFeatures;
    spatialMatch = calculateSpatialMatch(mcmcState.spatialFeatures, targetSpatial);
    morphMatch = calculateMorphologyMatch(mcmcState.morphologyFeatures, targetMorph);
    spatialDeficit = max(0, 0.97 - spatialMatch);
    morphDeficit = max(0, 0.97 - morphMatch);
    multiScaleDeficit = 0;
    if isfield(mcmcState.optParams, 'multiScaleSpatialFeatures') && ...
            ~isempty(mcmcState.optParams.multiScaleSpatialFeatures)
        multiScaleMatch = calculateMultiScaleMatch(mcmcState.multiScaleSpatialFeatures, ...
            mcmcState.optParams.multiScaleSpatialFeatures);
        multiScaleDeficit = max(0, 0.95 - multiScaleMatch);
    end
    % 根据移动类型和阶段调整
    switch moveType
        case {'morphology_preserving', 'local_shape', 'boundary_smooth', ...
                'cluster_shape', 'fine_morphology'}
            % 形态相关移动
            phaseBoost = 0.9;
            if strcmp(phase, 'morphology')
                phaseBoost = 1.5;
            elseif strcmp(phase, 'balanced')
                phaseBoost = 1.1;
            end
            if morphDeficit > 0
                shapePenalty = 3.8;
                if strcmp(phase, 'morphology')
                    shapePenalty = 4.5; % 形态阶段提高形状恶化惩罚
                end
                deltaE = baseDE - phaseBoost * (0.3 + shapePenalty * morphDeficit);
            else
                deltaE = baseDE + 0.08 * phaseBoost;
            end
        case {'spatial_aware', 'spatial_correlation', 'anisotropy_adjust', ...
                'connectivity_enhance'}
            % 空间相关移动
            phaseBoost = 0.9;
            if strcmp(phase, 'spatial')
                phaseBoost = 1.5;
            elseif strcmp(phase, 'balanced')
                phaseBoost = 1.1;
            end
            if spatialDeficit > 0
                deltaE = baseDE - phaseBoost * (0.25 + 3.0 * spatialDeficit + 1.2 * multiScaleDeficit);
            else
                deltaE = baseDE + 0.1 * phaseBoost;
            end
        case 'structure_coherence'
            % 结构连贯性移动在所有阶段都有益
            combinedDeficit = max([morphDeficit, spatialDeficit, multiScaleDeficit]);
            deltaE = baseDE - 0.35 - 2.5 * combinedDeficit;
        case 'large_scale'
            % 大规模移动需要谨慎
            deltaE = baseDE + 0.3 + max(0, 0.6 - (morphDeficit + spatialDeficit));
        otherwise
            deltaE = baseDE;
    end
    % 添加随机扰动避免局部最优
    deltaE = deltaE * (0.98 + 0.04 * randn());
end
function deltaE = evaluateBaseMoveDelta(mcmcState, move, localStats)
    % 评估基础移动能量变化
    if isempty(move.linearIdx)
        deltaE = 0;
        return;
    end
    if nargin < 3 || isempty(localStats)
        localStats = precomputeLocalMoveStatistics(mcmcState.model);
    end
    % 快速估计能量变化
    idx = move.linearIdx(:);
    nChanged = numel(idx);
    modelSize = numel(mcmcState.model);
    newValues = logical(move.newValues(:));
    oldValues = logical(move.oldValues(:));
    % 孔隙率变化
    nFlipped = sum(newValues) - sum(oldValues);
    deltaPorosity = nFlipped / modelSize;
    targetPorosity = mcmcState.optParams.targetPorosity;
    currentPorosity = mean(mcmcState.model(:));
    newPorosity = currentPorosity + deltaPorosity;
    currentError = abs(currentPorosity - targetPorosity);
    newError = abs(newPorosity - targetPorosity);
    deltaE_porosity = (newError - currentError) * mcmcState.weights.porosity;
    % 显式惩罚孔隙率进一步恶化的移动
    porosityErrorIncrease = newError - currentError;
    if porosityErrorIncrease > 0.01
        deltaE_porosity = deltaE_porosity + mcmcState.weights.porosity * 3.5 * porosityErrorIncrease;
    end
    if currentPorosity < targetPorosity - 0.03 && deltaPorosity < 0
        % 当前孔隙率偏低仍在减少，强烈惩罚
        deltaE_porosity = deltaE_porosity + mcmcState.weights.porosity * (5 + 10 * abs(deltaPorosity));
    end
    % 利用预计算的局部统计量估计表面积变化
    neighborCount = double(localStats.neighborCount6(idx));
    neighborSum = double(localStats.neighborSum6(idx));
    currentValues = double(mcmcState.model(idx));
    currentMismatched = currentValues .* (neighborCount - neighborSum) + ...
        (1 - currentValues) .* neighborSum;
    newMismatched = double(newValues) .* (neighborCount - neighborSum) + ...
        double(~newValues) .* neighborSum;
    surfaceChangeEstimate = sum(newMismatched - currentMismatched);
    normalization = max(1, sum(neighborCount));
    surfaceRatio = surfaceChangeEstimate / normalization;
    deltaE_surface = surfaceRatio * mcmcState.weights.morphology;
    % 簇大小变化的粗略估计
    clusterImpact = 0;
    if nChanged > 20
        clusterImpact = 0.1; % 大变化可能破坏簇结构
    elseif nChanged > 10
        clusterImpact = 0.05;
    end
    deltaE_cluster = clusterImpact * mcmcState.weights.cluster;
    % 空间相关性影响（保持与旧版本一致的启发式）
    if nChanged > 1
        [xs, ys, zs] = ind2sub(size(mcmcState.model), idx);
        spatialSpread = std(xs) + std(ys) + std(zs);
        normalizedSpread = spatialSpread / (size(mcmcState.model, 1) + ...
            size(mcmcState.model, 2) + size(mcmcState.model, 3));
        deltaE_spatial = normalizedSpread * mcmcState.weights.spatial * 0.1;
    else
        deltaE_spatial = 0;
    end
    penalty = estimateClusterImpactPenalty(mcmcState, idx, newValues, localStats);
    % 总能量变化
    deltaE = deltaE_porosity + deltaE_surface + deltaE_cluster + deltaE_spatial + penalty;
end
function penalty = estimateClusterImpactPenalty(mcmcState, idx, newValues, localStats)
    % 估计移动对簇数量和最大簇的影响，并根据约束给出惩罚
    penalty = 0;
    if isempty(idx)
        return;
    end
    modelSize = size(mcmcState.model);
    [xs, ys, zs] = ind2sub(modelSize, idx);
    xr = max(min(xs)-2,1):min(max(xs)+2, modelSize(1));
    yr = max(min(ys)-2,1):min(max(ys)+2, modelSize(2));
    zr = max(min(zs)-2,1):min(max(zs)+2, modelSize(3));
    subCurrent = mcmcState.model(xr, yr, zr);
    subNew = subCurrent;
    localIdx = sub2ind(size(subCurrent), xs - xr(1) + 1, ys - yr(1) + 1, zs - zr(1) + 1);
    subNew(localIdx) = newValues;
    CCbefore = bwconncomp(subCurrent, 26);
    CCafter = bwconncomp(subNew, 26);
    deltaClusters = CCafter.NumObjects - CCbefore.NumObjects;
    baseCount = mcmcState.features.numClusters;
    if isfield(mcmcState.optParams, 'targetClusterCount') && ~isempty(mcmcState.optParams.targetClusterCount)
        estCount = baseCount + deltaClusters;
        tgt = mcmcState.optParams.targetClusterCount;
        if abs(estCount - tgt) > 0.2 * max(1, tgt)
            penalty = penalty + 1e3 + mcmcState.weights.cluster * abs(estCount - tgt);
        end
    end
    maxBeforeLocal = 0; maxAfterLocal = 0;
    if ~isempty(CCbefore.PixelIdxList)
        maxBeforeLocal = max(cellfun(@numel, CCbefore.PixelIdxList));
    end
    if ~isempty(CCafter.PixelIdxList)
        maxAfterLocal = max(cellfun(@numel, CCafter.PixelIdxList));
    end
    baseMax = maxBeforeLocal;
    if isfield(mcmcState, 'features') && isfield(mcmcState.features, 'maxSize') && ~isempty(mcmcState.features.maxSize)
        baseMax = mcmcState.features.maxSize;
    end
    estMax = max([maxAfterLocal, baseMax]);
    if isfield(mcmcState.optParams, 'targetMax') && ~isempty(mcmcState.optParams.targetMax)
        if estMax > 1.2 * mcmcState.optParams.targetMax
            penalty = penalty + 1e3 + mcmcState.weights.cluster * (estMax / max(mcmcState.optParams.targetMax, 1));
        end
    end
    if isfield(mcmcState.optParams, 'preserveSmallPores') && mcmcState.optParams.preserveSmallPores
        [lowerBound, ~] = getAdaptiveClusterBounds(mcmcState.optParams);
        if ~isempty(lowerBound) && lowerBound > 0
            smallBefore = sum(cellfun(@numel, CCbefore.PixelIdxList) < 1.2 * lowerBound & ...
                cellfun(@numel, CCbefore.PixelIdxList) > 0);
            smallAfter = sum(cellfun(@numel, CCafter.PixelIdxList) < 1.2 * lowerBound & ...
                cellfun(@numel, CCafter.PixelIdxList) > 0);
            if smallAfter < smallBefore
                penalty = penalty + mcmcState.weights.cluster * (smallBefore - smallAfter) * 50;
            end
        end
    end
    % 防止未被localStats覆盖的大块修改
    if isfield(localStats, 'neighborCount6') && numel(idx) > 0.1 * numel(mcmcState.model)
        penalty = penalty + 1e4;
    end
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
    deltaEnergies, temperature, moveTypes, parallelHint)
    % 应用最佳综合移动
    if nargin < 6
        parallelHint = [];
    end
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
    [~, bestIdx] = min(validEnergies);
    bestMove = validMoves{bestIdx};
    bestType = validTypes{bestIdx};
    if isfield(bestMove, 'linearIdx') && ~isempty(bestMove.linearIdx)
        % 验证移动的有效性
        validIndices = bestMove.linearIdx > 0 & bestMove.linearIdx <= numel(mcmcState.model);
        if any(validIndices)
            validLinearIdx = bestMove.linearIdx(validIndices);
            validNewValues = bestMove.newValues(validIndices);
            % 构造候选状态并计算真实能量变化
            candidateState = mcmcState;
            candidateState.model = mcmcState.model;
            candidateState.model(validLinearIdx) = validNewValues;
            candidateState = updateAllFeatures(candidateState, parallelHint);
            actualDeltaE = candidateState.currentEnergy - mcmcState.currentEnergy;
            % Metropolis准则（基于真实能量差）
            safeTemperature = max(temperature, 1e-6);
            acceptProb = min(1, exp(-actualDeltaE / safeTemperature));
            if actualDeltaE < 0 || rand() < acceptProb
                accepted = true;
                % 记录接受的移动类型
                if ~isfield(candidateState, 'acceptedMoveTypes') || isempty(candidateState.acceptedMoveTypes)
                    candidateState.acceptedMoveTypes = struct();
                end
                if isfield(candidateState.acceptedMoveTypes, bestType)
                    candidateState.acceptedMoveTypes.(bestType) = ...
                        candidateState.acceptedMoveTypes.(bestType) + 1;
                else
                    candidateState.acceptedMoveTypes.(bestType) = 1;
                end
                candidateState.boundaryUpdated = false;
                mcmcState = candidateState;
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
    iter, phase, parallelHint)
    % 综合自适应调整
    if nargin < 5
        parallelHint = [];
    end
    
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
        mcmcState = updateAllFeatures(mcmcState, parallelHint);
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
                    model = model | sphere;
                else
                    model = model & ~sphere;
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
                halfSize = floor(regionSize/2);
                % 当模型尺寸不足以容纳区域时直接跳过本次扰动
                if nx <= 2 * halfSize || ny <= 2 * halfSize || nz <= 2 * halfSize
                    continue;
                end
                cx = randi([halfSize + 1, nx - halfSize]);
                cy = randi([halfSize + 1, ny - halfSize]);
                cz = randi([halfSize + 1, nz - halfSize]);
                
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
        se = createAnisotropicStructuringElement([1, 0, 0], [3, 1, 1]);
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
        [lowerBound, upperBound] = getAdaptiveClusterBounds(mcmcState.optParams);
        fprintf('  簇大小: [%d, %d] (自适应目标: [%d, %d])\n', ...
            mcmcState.features.sizeStats(1), mcmcState.features.sizeStats(2), ...
            round(lowerBound), round(upperBound));
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
function finalModel = comprehensivePostProcessShapePreserving(model, optParams)
    % 形态保持的温和后处理，仅在边界进行小幅清理
    fprintf('执行形态保持后处理...\n');
    targetPorosity = optParams.targetPorosity;
    targetPorosity = max(0, min(1, targetPorosity));
    preModel = model;
    preFeatures = extractEfficientClusterFeatures(model);
    refModel = optParams.referenceModel;
    % 1. 删除孤立的 1-2 体素噪声
    CC = bwconncomp(model, 26);
    sizes = cellfun(@numel, CC.PixelIdxList);
    removeIdx = find(sizes <= 2);
    for i = 1:numel(removeIdx)
        model(CC.PixelIdxList{removeIdx(i)}) = false;
    end
    % 2. 修复长度1的毛刺（仅在边界）
    boundary = bwperim(model, 26);
    neighborKernel = ones(3,3,3);
    neighborKernel(2,2,2) = 0;
    neighborCount = convn(double(model), neighborKernel, 'same');
    spurMask = model & boundary & neighborCount <= 1;
    model(spurMask) = false;
    % 3. 轻微边界平滑（不创建新连通桥）
    bandMask = imdilate(boundary, strel('sphere', 1));
    cleanProposal = model;
    filtered = medfilt3(model, [3 3 3]);
    cleanProposal(bandMask) = filtered(bandMask);
    mergedCheck = computeConnectivityMetrics(cleanProposal);
    if mergedCheck.largestComponentSize <= 1.2 * preFeatures.maxSize
        model = cleanProposal;
    end
    % 4. 孔隙率微调：仅在边界随机翻转极少量体素
    porosityDiff = targetPorosity - mean(model(:));
    if abs(porosityDiff) > 0.0005
        step = sign(porosityDiff);
        adjustMask = bandMask;
        adjustIdx = find(adjustMask);
        adjustCount = min(max(1, round(0.0002 * numel(model))), numel(adjustIdx));
        adjustIdx = adjustIdx(randperm(numel(adjustIdx), adjustCount));
        proposal = model;
        proposal(adjustIdx) = step > 0;
        % 只有当直接重叠和簇统计没有恶化时才接受
        overlapBefore = mean(model(:) == refModel(:));
        overlapAfter = mean(proposal(:) == refModel(:));
        postFeatures = extractEfficientClusterFeatures(proposal);
        if overlapAfter >= overlapBefore && ...
                postFeatures.numClusters >= 0.9 * preFeatures.numClusters && ...
                postFeatures.maxSize <= 1.1 * preFeatures.maxSize
            model = proposal;
        end
    end
    % 5. 最终质量检查，避免簇被合并
    postFeatures = extractEfficientClusterFeatures(model);
    if postFeatures.numClusters < 0.85 * max(1, preFeatures.numClusters) || ...
            postFeatures.maxSize > 1.1 * preFeatures.maxSize
        warning('形态保持后处理检测到簇统计恶化，回退处理。');
        model = preModel;
    end
    finalModel = model;
    fprintf('形态保持后处理完成。\n');
end
function finalModel = comprehensivePostProcessWithSmallPores(model, optParams)
    % 综合后处理 - 改进版，更好地匹配原始孔隙形态
    fprintf('执行综合后处理（形态保持版本）...\n');
    % 记录当前孔隙率并确定目标孔隙率
    originalPorosity = mean(model(:));
    targetPorosity = optParams.targetPorosity;
    if isempty(targetPorosity) || ~isfinite(targetPorosity)
        targetPorosity = originalPorosity;
    end
    targetPorosity = max(0, min(1, targetPorosity));
    preserveSmall = isfield(optParams, 'preserveSmallPores') && optParams.preserveSmallPores;
    preClusters = extractEfficientClusterFeatures(model);
    preConn = computeConnectivityMetrics(model);
    modelBeforePost = model;
    
    if preserveSmall
        fprintf('  1. 保守边界细节修复...\n');
        model = gentleBoundaryTouchUps(model);
        fprintf('  2. 仅移除孤立的极小孔隙(<=2体素)...\n');
        model = removeIsolatedTinyIslands(model, 2);
        fprintf('  3. 细通道修复以保持连通性...\n');
        model = repairThinChannels(model);
    else
        % 仍保留原有较强后处理流程
        fprintf('  1. 强化噪声去除...\n');
        model = enhancedNoiseRemoval(model, optParams);
        fprintf('  2. 形态学重建...\n');
        model = morphologicalReconstruction(model, optParams);
        fprintf('  3. 基于参考的形态恢复...\n');
        model = referenceBasedMorphologyRestoration(model, optParams.originalBinaryModel);
        fprintf('  4. 多尺度边界平滑...\n');
        model = multiScaleBoundarySmoothing(model);
        fprintf('  5. 结构优化...\n');
        model = structureOptimization(model, optParams);
    end
    % 温和孔隙率调整，避免破坏拓扑
    fprintf('  6. 恢复目标孔隙率...\n');
    model = smartPorosityAdjustment(model, targetPorosity, optParams);
    fprintf('  7. 最终质量检查...\n');
    model = finalQualityRefinement(model, optParams);
    model = enforceTargetClusterCount(model, optParams);
    % 后处理安全检查，避免簇数与连通性被大幅破坏
    postClusters = extractEfficientClusterFeatures(model);
    postConn = computeConnectivityMetrics(model);
    if postClusters.numClusters < 0.8 * max(1, preClusters.numClusters) || ...
            postConn.largestComponentRatio < 0.8 * max(preConn.largestComponentRatio, eps)
        warning('后处理过于激进，回退到处理前模型以保护拓扑。');
        model = modelBeforePost;
    end
    
    % 统计最终结果
    reportFinalStatistics(model, optParams);
    
    finalModel = model;
    fprintf('综合后处理完成\n');
end
function finalModel = comprehensivePostProcess(model, optParams)
    % 综合后处理 - 标准平滑版本
    fprintf('执行综合后处理（标准平滑版本）...\n');
    
    % 记录当前孔隙率并确定目标孔隙率
    originalPorosity = mean(model(:));
    targetPorosity = optParams.targetPorosity;
    if isempty(targetPorosity) || ~isfinite(targetPorosity)
        targetPorosity = originalPorosity;
    end
    targetPorosity = max(0, min(1, targetPorosity));
    
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
    model = adjustPorosity(model, targetPorosity, optParams.referenceDensityMap);
    model = enforceTargetClusterCount(model, optParams);
    
    finalModel = model;
    fprintf('综合后处理完成（平滑版本）\n');
end
%% ========== 核心后处理函数 ==========
function model = gentleBoundaryTouchUps(model)
    % 仅在孔隙边界执行轻微平滑，避免整体拓扑被重写
    boundary = bwperim(model, 26);
    boundaryRegion = imdilate(boundary, strel('sphere', 1));
    smoothed = imopen(model, strel('sphere', 1));
    model(boundaryRegion) = smoothed(boundaryRegion);
end
function model = removeIsolatedTinyIslands(model, maxSize)
    % 只移除完全孤立且体素数极小的孔隙，保护小尺度结构
    CC = bwconncomp(model, 26);
    if CC.NumObjects == 0
        return;
    end
    sizes = cellfun(@numel, CC.PixelIdxList);
    removeIdx = find(sizes <= maxSize);
    for i = 1:numel(removeIdx)
        model(CC.PixelIdxList{removeIdx(i)}) = false;
    end
end
function model = enhancedNoiseRemoval(model, optParams)
    % 强化的噪声去除
    % 移除小于目标最小值一定比例的所有小孔隙
    [lowerBound, ~] = getAdaptiveClusterBounds(optParams);
    if isempty(lowerBound) || lowerBound <= 0
        lowerBound = 20;
    end
    minThreshold = max(10, lowerBound / 5);
    
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
    [lowerBound, ~] = getAdaptiveClusterBounds(optParams);
    if isempty(lowerBound)
        lowerBound = prctile(double(sizes), 20);
    end
    mediumPoresIdx = find(sizes >= lowerBound & sizes < avgSize);
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
    [lowerBound, ~] = getAdaptiveClusterBounds(optParams);
    if isempty(lowerBound) || lowerBound <= 0
        lowerBound = prctile(double(sizes), 15);
    end
    smallPoresIdx = find(sizes < lowerBound);
    
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
    [~, upperBound] = getAdaptiveClusterBounds(optParams);
    if isempty(upperBound) || upperBound <= 0
        upperBound = prctile(double(sizes), 80);
    end
    threshold = upperBound * 2;
    se = strel('sphere', 1);
    for i = 1:length(sizes)
        if sizes(i) <= threshold
            continue;
        end
        clusterMask = false(size(model));
        clusterMask(CC.PixelIdxList{i}) = true;
        distMap = bwdist(~clusterMask);
        shrinkRatio = min(0.4, max(0.05, (sizes(i) - upperBound) / sizes(i)));
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
    [lowerBound, ~] = getAdaptiveClusterBounds(optParams);
    if isempty(lowerBound)
        lowerBound = prctile(double(sizes), 20);
    end
    if CC.NumObjects >= 2 && sortedSizes(2) > lowerBound
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
        [lowerBound, upperBound] = getAdaptiveClusterBounds(optParams);
        if isempty(lowerBound)
            lowerBound = prctile(double(sizes), 20);
        end
        if isempty(upperBound)
            upperBound = prctile(double(sizes), 80);
        end
        mediumPoresIdx = find(sizes > lowerBound & sizes < upperBound);
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
        
        [~, upperBound] = getAdaptiveClusterBounds(optParams);
        if isempty(upperBound) || upperBound <= 0
            upperBound = prctile(double(sizes), 80);
        end
        largePoresIdx = find(sizes > upperBound);
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
    [lowerBound, ~] = getAdaptiveClusterBounds(optParams);
    if isempty(lowerBound) || lowerBound <= 0
        lowerBound = 30;
    end
    threshold = max(20, lowerBound / 3);
    
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
    [lowerBound, ~] = getAdaptiveClusterBounds(optParams);
    if isempty(lowerBound) || lowerBound <= 0
        lowerBound = prctile(double(sizes), 20);
    end
    keepIdx = find(sizes >= lowerBound);
    
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
    fprintf('  模型孔隙率: %.4f (目标 %.4f)\n', mean(model(:)), optParams.targetPorosity);
    fprintf('  最小孔隙: %d 体素 | 最大孔隙: %d 体素\n', min(sizes), max(sizes));
    fprintf('  平均孔隙: %.1f 体素 | 标准差: %.1f 体素\n', mean(sizes), std(sizes));
    [lowerBound, upperBound] = getAdaptiveClusterBounds(optParams);
    if isempty(lowerBound) || isempty(upperBound) || upperBound <= 0
        lowerBound = prctile(double(sizes), 15);
        upperBound = prctile(double(sizes), 85);
    end
    smallCount = sum(sizes < lowerBound);
    mediumCount = sum(sizes >= lowerBound & sizes <= upperBound);
    largeCount = sum(sizes > upperBound);
    fprintf('\n孔隙大小分布（自适应范围 [%d, %d]）：\n', round(lowerBound), round(upperBound));
    fprintf('  小孔隙: %d (%.1f%%)\n', smallCount, 100*smallCount/CC.NumObjects);
    fprintf('  中孔隙: %d (%.1f%%)\n', mediumCount, 100*mediumCount/CC.NumObjects);
    fprintf('  大孔隙: %d (%.1f%%)\n', largeCount, 100*largeCount/CC.NumObjects);
    boundary = bwperim(model, 26);
    smoothness = 1 - sum(boundary(:)) / max(sum(model(:)), 1);
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
        [lowerBound, upperBound] = getAdaptiveClusterBounds(optParams);
        fprintf('  自适应目标范围: [%d, %d]\n', round(lowerBound), round(upperBound));
        outOfRange = sum(sizes < lowerBound) + sum(sizes > upperBound);
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
    % 综合结果展示
    fprintf('\n\n===== 综合优化结果总结 =====\n');
    finalFeatures = extractEfficientClusterFeatures(finalModel);
    finalSpatial = computeEnhancedSpatialFeatures(finalModel);
    finalMorph = computeDetailedMorphologyFeatures(finalModel);
    fprintf('\n1. 基本统计：\n');
    fprintf('  目标孔隙率: %.4f | 原始: %.4f | 最终: %.4f\n', ...
        optParams.targetPorosity, mean(originalModel(:)), mean(finalModel(:)));
    fprintf('  原始簇数: %d | 最终簇数: %d\n', ...
        originalFeatures.numClusters, finalFeatures.numClusters);
    if isfield(optParams, 'inferredSpatialSummary') && ~isempty(optParams.inferredSpatialSummary)
        fprintf('\n2. 目标空间参数（梯度推断）：\n');
        fieldsToShow = {'twoPoint_mean', 'anisotropy', 'porosityGradient', ...
            'surfaceToVolumeRatio', 'tortuosityEstimate', 'chord_meanLength', ...
            'pore_meanRadius'};
        for i = 1:numel(fieldsToShow)
            name = fieldsToShow{i};
            if isfield(optParams.inferredSpatialSummary, name)
                fprintf('  %-26s : %.4f\n', name, optParams.inferredSpatialSummary.(name));
            end
        end
    end
    fprintf('\n3. 目标匹配评估：\n');
    spatialMatch = calculateSpatialMatch(finalSpatial, optParams.spatialFeatures);
    morphologyMatch = calculateMorphologyMatch(finalMorph, optParams.morphologyFeatures);
    directOverlap = 1 - calculateDirectOverlapEnergy(finalModel, optParams);
    fprintf('  空间特征匹配度: %.3f\n', spatialMatch);
    fprintf('  形态特征匹配度: %.3f (目标保持原始形态)\n', morphologyMatch);
    fprintf('  逐体素直接重叠比例: %.3f\n', directOverlap);
    fprintf('\n4. 形态与空间特征对比：\n');
    if ~isempty(originalMorph.sphericity) && ~isempty(finalMorph.sphericity)
        fprintf('  球形度: 原始=%.3f | 最终=%.3f\n', ...
            mean(originalMorph.sphericity), mean(finalMorph.sphericity));
        fprintf('  伸长率: 原始=%.3f | 最终=%.3f\n', ...
            mean(originalMorph.elongation), mean(finalMorph.elongation));
    end
    fprintf('  各向异性: 原始=%.4f | 最终=%.4f\n', ...
        originalSpatial.anisotropy, finalSpatial.anisotropy);
    fprintf('  最大连通簇比: 原始=%.4f | 最终=%.4f\n', ...
        originalSpatial.connectivity.largestComponentRatio, ...
        finalSpatial.connectivity.largestComponentRatio);
    fprintf('\n5. 簇分布统计：\n');
    if finalFeatures.numClusters == 0
        fprintf('  最终模型无簇，可跳过簇统计。\n');
    else
        sizes = double(finalFeatures.sizes);
        fprintf('  簇大小范围: [%d, %d]\n', min(sizes), max(sizes));
        fprintf('  平均体素: %.1f (σ=%.1f)\n', mean(sizes), std(sizes));
        if isfield(optParams, 'targetClusterCount') && ~isempty(optParams.targetClusterCount)
            fprintf('  簇数量偏差: %.1f%% (目标 %d, 实际 %d)\n', ...
                100 * (finalFeatures.numClusters - optParams.targetClusterCount) / max(1, optParams.targetClusterCount), ...
                optParams.targetClusterCount, finalFeatures.numClusters);
        end
        if isfield(optParams, 'targetMax') && ~isempty(optParams.targetMax)
            fprintf('  最大簇偏差: %.1f%% (目标 %.0f, 实际 %.0f)\n', ...
                100 * (max(sizes) - optParams.targetMax) / max(1, optParams.targetMax), ...
                optParams.targetMax, max(sizes));
        end
        if isfield(optParams, 'clusterReference') && ~isempty(optParams.clusterReference)
            ref = optParams.clusterReference;
            fprintf('  参考均值/标准差: %.1f / %.1f\n', ref.meanSize, ref.stdSize);
            [lowerBound, upperBound] = getAdaptiveClusterBounds(optParams);
        else
            lowerBound = prctile(sizes, 15);
            upperBound = prctile(sizes, 85);
        end
        smallCount = sum(sizes < lowerBound);
        mediumCount = sum(sizes >= lowerBound & sizes <= upperBound);
        largeCount = sum(sizes > upperBound);
        fprintf('  自适应分类范围: [%d, %d]\n', round(lowerBound), round(upperBound));
        fprintf('    小簇: %d (%.1f%%)\n', smallCount, 100*smallCount/finalFeatures.numClusters);
        fprintf('    中簇: %d (%.1f%%)\n', mediumCount, 100*mediumCount/finalFeatures.numClusters);
        fprintf('    大簇: %d (%.1f%%)\n', largeCount, 100*largeCount/finalFeatures.numClusters);
        if exist('ref', 'var') && isfield(ref, 'histogram')
            currentHist = histcounts(log10(sizes+1), ref.histEdges, 'Normalization', 'probability');
            histDiff = sum(abs(currentHist - ref.histogram));
            fprintf('  与原始簇分布差异: %.3f\n', histDiff);
        end
    end
    fprintf('\n6. 优化性能概览：\n');
    energyHistory = performanceMonitor.energyHistory;
    energyHistory = energyHistory(energyHistory ~= 0);
    if ~isempty(energyHistory)
        fprintf('  初始能量: %.6f | 最终能量: %.6f\n', energyHistory(1), energyHistory(end));
        fprintf('  能量相对下降: %.2f%%\n', ...
            (energyHistory(1) - energyHistory(end)) / abs(energyHistory(1)) * 100);
    end
    if ~isempty(performanceMonitor.acceptanceRatio)
        fprintf('  平均接受率: %.2f%%\n', mean(performanceMonitor.acceptanceRatio) * 100);
    end
    fprintf('\n7. 结构质量指标：\n');
    boundary = bwperim(finalModel, 26);
    smoothness = 1 - sum(boundary(:)) / max(sum(finalModel(:)), 1);
    fprintf('  边界平滑度指标: %.3f\n', smoothness);
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
    midSlice = squeeze(diff(:,:,round(size(diff,3)/2)));
    imagesc(midSlice);
    caxis([-1 1]);
    colormap(subplot(2,3,3), [0 0 1; 0 1 0; 1 0 0]); % 蓝/绿/红 = 负/零/正
    colorbar;
    title('差异图（绿=一致，红/蓝=不一致）');
        
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
function targetMorph = buildMorphologyTargetForPorosity(baseMorph, refPorosity, targetPorosity)
    % 根据目标孔隙率对形态目标进行构造，兼顾形状与网络类指标
    if nargin < 3
        targetPorosity = refPorosity;
    end
    targetMorph = baseMorph;
    scale = targetPorosity / max(refPorosity, eps);
    if isfield(targetMorph, 'poreNetworkDensity')
        targetMorph.poreNetworkDensity = min(1.0, targetMorph.poreNetworkDensity * scale);
    end
    if isfield(targetMorph, 'lacunarity')
        targetMorph.lacunarity = max(0, targetMorph.lacunarity / sqrt(scale));
    end
    if isfield(targetMorph, 'skeletonFeatures') && isfield(targetMorph.skeletonFeatures, 'density')
        targetMorph.skeletonFeatures.density = ...
            min(1.0, targetMorph.skeletonFeatures.density * scale);
    end
end
function match = calculateMorphologyMatch(currentMorph, targetMorph)
    % 计算形态学特征匹配度，强调形状鲁棒性
    if isempty(currentMorph) || isempty(targetMorph)
        match = 0;
        return;
    end
    % 伸长率匹配（中位数 + 裁剪）
    elongMatch = 1;
    if isfield(currentMorph, 'elongation') && isfield(targetMorph, 'elongation') && ...
            ~isempty(currentMorph.elongation) && ~isempty(targetMorph.elongation)
        medCurElong = median(currentMorph.elongation(:));
        medTgtElong = median(targetMorph.elongation(:));
        elongDiff = abs(medCurElong - medTgtElong) / (medTgtElong + 0.1);
        elongDiff = min(1, elongDiff);
        elongMatch = 1 - elongDiff;
    end
    % 球形度匹配（中位数 + 裁剪）
    spherMatch = 1;
    if isfield(currentMorph, 'sphericity') && isfield(targetMorph, 'sphericity') && ...
            ~isempty(currentMorph.sphericity) && ~isempty(targetMorph.sphericity)
        medCurSpher = median(currentMorph.sphericity(:));
        medTgtSpher = median(targetMorph.sphericity(:));
        spherDiff = abs(medCurSpher - medTgtSpher) / 0.5;
        spherDiff = min(1, spherDiff);
        spherMatch = 1 - spherDiff;
    end
    % 网络密度匹配
    densityMatch = 1;
    if isfield(currentMorph, 'poreNetworkDensity') && isfield(targetMorph, 'poreNetworkDensity')
        densDiff = abs(currentMorph.poreNetworkDensity - targetMorph.poreNetworkDensity) / ...
            max(targetMorph.poreNetworkDensity, 0.05);
        densDiff = min(1, densDiff);
        densityMatch = 1 - densDiff;
    end
    % 综合匹配度，重视形状指标
    match = 0.4 * elongMatch + 0.4 * spherMatch + 0.2 * densityMatch;
    match = max(0, min(1, match));
end
function ensureParallelPool(config)
    % 根据配置强制或尝试启动并行池，避免迭代早期 CPU 闲置
    if nargin < 1 || ~isstruct(config)
        config = struct('forceParallelPool', false, 'desiredPoolSize', 0);
    end
    if ~isfield(config, 'forceParallelPool') || ~config.forceParallelPool
        return;
    end
    if ~isParallelToolboxAvailable()
        return;
    end
    pool = gcp('nocreate');
    if isempty(pool)
        try
            if isfield(config, 'desiredPoolSize') && config.desiredPoolSize > 0
                parpool(config.desiredPoolSize);
            else
                parpool('Threads'); % R2023a 默认线程池
            end
            fprintf('并行池已启动，用于批量特征和移动评估。\n');
        catch ME
            warning(ME.identifier, '启动并行池失败，继续串行执行：%s', ME.message);
        end
    end
end
function useParallel = shouldUseParallel(batchSize, problemSize)
    % 判断是否应启用并行计算
    if nargin < 1
        batchSize = 0;
    end
    if nargin < 2
        problemSize = 0;
    end
    useParallel = false;
    if batchSize < 4
        return;
    end
    if ~isParallelToolboxAvailable()
        return;
    end
    try
        % 避免在并行工作线程中递归启动并行
        currentTask = [];
        try
            currentTask = getCurrentTask();
        catch
            currentTask = [];
        end
        if ~isempty(currentTask)
            return;
        end
        % 仅在已有线程池时使用并行，避免反复启动导致的稳定性问题
        pool = gcp('nocreate');
        if isempty(pool)
            return; % 无池时保持串行
        end
        useParallel = ~isempty(pool);
    catch
        useParallel = false;
    end
end
function available = isParallelToolboxAvailable()
    % 检查并缓存并行工具箱的可用性，避免在不支持的环境中调用并行API
    persistent cachedAvailability
    if isempty(cachedAvailability)
        hasParpool = exist('parpool', 'file') == 2;
        hasLicense = false;
        try
            hasLicense = license('test', 'Distrib_Computing_Toolbox');
        catch
            hasLicense = false;
        end
        hasParallelPkg = false;
        try
            hasParallelPkg = ~isempty(ver('parallel'));
        catch
            hasParallelPkg = false;
        end
        cachedAvailability = hasParpool && hasLicense && hasParallelPkg;
    end
    available = cachedAvailability;
end
%% 主函数结束